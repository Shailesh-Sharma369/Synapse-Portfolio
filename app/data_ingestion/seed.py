from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.database import Base, SessionLocal, engine
from app.db.models import (
    AccrualSchedule,
    Company,
    IntercompanyTransaction,
    RevenueContract,
    TrialBalance,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

DATASET_DIR = Path(__file__).resolve().parents[2] / "dataset"


def _parse_date(value: Any) -> date | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return pd.to_datetime(value).date()


def _parse_decimal(value: Any, default: str = "0.00") -> Decimal:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return Decimal(default)
    return Decimal(str(value))


def _load_csv(name: str) -> pd.DataFrame:
    path = DATASET_DIR / name
    if not path.exists():
        logger.warning("Dataset file missing, skipping: %s", path)
        return pd.DataFrame()
    return pd.read_csv(path)


def _load_json(name: str) -> list[dict[str, Any]]:
    path = DATASET_DIR / name
    if not path.exists():
        logger.warning("Dataset file missing, skipping: %s", path)
        return []
    with path.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def seed_companies(db: Session) -> dict[str, str]:
    df = _load_csv("companies.csv")
    name_to_id: dict[str, str] = {}

    existing = {c.name: str(c.id) for c in db.scalars(select(Company)).all()}
    if existing:
        logger.info("Companies already seeded (%d). Reusing.", len(existing))
        return existing

    for _, row in df.iterrows():
        company = Company(
            name=str(row["name"]).strip(),
            industry=row.get("industry"),
            revenue_annual=_parse_decimal(row.get("revenue_annual")),
        )
        db.add(company)
    db.flush()
    for c in db.scalars(select(Company)).all():
        name_to_id[c.name] = str(c.id)
    logger.info("Seeded %d companies.", len(name_to_id))
    return name_to_id


def seed_trial_balances(db: Session, company_map: dict[str, str]) -> None:
    df = _load_csv("trial_balances.csv")
    if df.empty:
        return
    count = 0
    for _, row in df.iterrows():
        cname = str(row["company_name"]).strip()
        cid = company_map.get(cname)
        if not cid:
            logger.warning("Unknown company for TB row: %s", cname)
            continue
        tb = TrialBalance(
            company_id=cid,
            period=str(row["period"]),
            account_code=str(row["account_code"]),
            account_name=str(row["account_name"]),
            debit=_parse_decimal(row.get("debit")),
            credit=_parse_decimal(row.get("credit")),
            balance=_parse_decimal(row.get("balance")),
            account_type=str(row["account_type"]),
        )
        db.add(tb)
        count += 1
    logger.info("Seeded %d trial balance rows.", count)


def seed_intercompany(db: Session, company_map: dict[str, str]) -> None:
    df = _load_csv("intercompany_transactions.csv")
    if df.empty:
        return
    count = 0
    for _, row in df.iterrows():
        seller = company_map.get(str(row["selling_entity"]).strip())
        buyer = company_map.get(str(row["buying_entity"]).strip())
        if not seller or not buyer:
            logger.warning("Skipping IC row, unknown entity.")
            continue
        ic = IntercompanyTransaction(
            date=_parse_date(row["date"]),
            selling_entity_id=seller,
            buying_entity_id=buyer,
            description=row.get("description"),
            amount=_parse_decimal(row.get("amount")),
            gl_account=str(row["gl_account"]),
        )
        db.add(ic)
        count += 1
    logger.info("Seeded %d intercompany transactions.", count)


def seed_accruals(db: Session, company_map: dict[str, str]) -> None:
    df = _load_csv("accrual_schedules.csv")
    if df.empty:
        return
    count = 0
    for _, row in df.iterrows():
        cid = company_map.get(str(row["company_name"]).strip())
        if not cid:
            continue
        db.add(
            AccrualSchedule(
                company_id=cid,
                accrual_type=str(row["accrual_type"]),
                gl_account=str(row["gl_account"]),
                frequency=str(row["frequency"]),
                amount=_parse_decimal(row.get("amount")),
                last_booked_date=_parse_date(row.get("last_booked_date")),
            )
        )
        count += 1
    logger.info("Seeded %d accrual schedules.", count)


def seed_revenue_contracts(db: Session, company_map: dict[str, str]) -> None:
    records = _load_json("revenue_contracts.json")
    if not records:
        return
    count = 0
    for rec in records:
        cid = company_map.get(str(rec.get("company_name", "")).strip())
        if not cid:
            logger.warning("Skipping contract, unknown company: %s", rec.get("company_name"))
            continue
        db.add(
            RevenueContract(
                company_id=cid,
                customer=str(rec["customer"]),
                start_date=_parse_date(rec["start_date"]),
                end_date=_parse_date(rec["end_date"]),
                total_contract_value=_parse_decimal(rec.get("total_contract_value")),
                billing_schedule=rec.get("billing_schedule"),
                performance_obligations=rec.get("performance_obligations", []),
            )
        )
        count += 1
    logger.info("Seeded %d revenue contracts.", count)


def run_seed() -> None:
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        company_map = seed_companies(db)
        seed_trial_balances(db, company_map)
        seed_intercompany(db, company_map)
        seed_accruals(db, company_map)
        seed_revenue_contracts(db, company_map)
        db.commit()
        logger.info("✅ Seeding completed successfully.")
    except SQLAlchemyError as exc:
        db.rollback()
        logger.exception("❌ Database error during seeding: %s", exc)
        raise
    except Exception as exc:  # noqa: BLE001
        db.rollback()
        logger.exception("❌ Unexpected error during seeding: %s", exc)
        raise
    finally:
        db.close()


if __name__ == "__main__":
    run_seed()