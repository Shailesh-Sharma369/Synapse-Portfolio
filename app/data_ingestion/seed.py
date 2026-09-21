"""
Data ingestion script for the Month-End Close Orchestration System.

Reads CSV/JSON from `assignment1_data/` and inserts into PostgreSQL
via SQLAlchemy models.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from app.db.database import Base, SessionLocal, engine
from app.db.models import (
    AccrualSchedule,
    BankStatement,
    Budget,
    Company,
    IntercompanyTransaction,
    RevenueContract,
    TrialBalance,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

# app/data_ingestion/seed.py → parents[2] = project root
DATASET_DIR = Path(__file__).resolve().parents[2] / "assignment1_data"


# ---------------------------------------------------------------------------
# Defensive parsers — never raise, always return a safe value.
# ---------------------------------------------------------------------------
def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, float) and pd.isna(value):
        return True
    if isinstance(value, str) and value.strip() == "":
        return True
    return False


def _parse_date(value: Any) -> date | None:
    if _is_missing(value):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    try:
        return pd.to_datetime(value).date()
    except Exception:
        return None


def _parse_decimal(value: Any, default: str = "0.00") -> Decimal:
    if _is_missing(value):
        return Decimal(default)
    try:
        return Decimal(str(value).replace(",", "").strip())
    except (InvalidOperation, ValueError):
        return Decimal(default)


def _parse_int(value: Any, default: int = 0) -> int:
    if _is_missing(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _clean_str(value: Any, default: str = "") -> str:
    if _is_missing(value):
        return default
    return str(value).strip()


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        logger.warning("Missing file, skipping: %s", path)
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse %s: %s", path, exc)
        return []


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        logger.warning("Missing file, skipping: %s", path)
        return None
    try:
        return pd.read_csv(path)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to read %s: %s", path, exc)
        return None


def _insert_row(db: Session, obj: Any, context: str) -> bool:
    """
    Insert a single ORM object inside a savepoint so one bad row doesn't kill
    the whole batch. Logs and returns False on failure.
    """
    try:
        with db.begin_nested():
            db.add(obj)
        return True
    except Exception as exc:  # noqa: BLE001
        logger.debug("Skipping bad row [%s]: %s", context, exc)
        return False


# ---------------------------------------------------------------------------
# Seeders
# ---------------------------------------------------------------------------
def seed_companies(db: Session) -> int:
    """Load company_metadata.json → companies table."""
    records = _load_json_list(DATASET_DIR / "company_metadata.json")
    inserted = 0

    for rec in records:
        cid = _clean_str(rec.get("id"))
        if not cid:
            logger.warning("Company record missing 'id', skipping: %s", rec)
            continue
        if _insert_row(
            db,
            Company(
                id=cid,
                name=_clean_str(rec.get("name")) or cid,
                industry=_clean_str(rec.get("industry")) or None,
                revenue_annual=_parse_decimal(rec.get("revenue_annual")),
            ),
            context=f"company:{cid}",
        ):
            inserted += 1

    db.flush()
    logger.info("Seeded %d companies.", inserted)
    return inserted


def _seed_tb_folder(db: Session, folder: Path, label: str) -> tuple[int, int]:
    """
    Shared loader for trial-balance CSVs (used for both `trial_balances/`
    and `prior_year/` folders).

    Returns (inserted, skipped).
    """
    if not folder.exists():
        logger.warning("Missing folder: %s", folder)
        return 0, 0

    total, skipped = 0, 0
    for csv_file in sorted(folder.glob("*.csv")):
        df = _read_csv(csv_file)
        if df is None:
            continue

        for _, row in df.iterrows():
            obj = TrialBalance(
                company_id=_clean_str(row.get("company_id")),
                period=_clean_str(row.get("period")),
                account_code=_clean_str(row.get("account_code")),
                account_name=_clean_str(row.get("account_name")),
                debit=_parse_decimal(row.get("debit")),
                credit=_parse_decimal(row.get("credit")),
                balance=_parse_decimal(row.get("balance")),
                account_type=_clean_str(row.get("account_type")),
            )
            if _insert_row(db, obj, context=f"{label}:{csv_file.name}"):
                total += 1
            else:
                skipped += 1

    db.flush()
    return total, skipped


def seed_trial_balances(db: Session) -> None:
    """Seed current-period trial balances from `trial_balances/*.csv`."""
    total, skipped = _seed_tb_folder(db, DATASET_DIR / "trial_balances", "TB")
    logger.info("Seeded %d trial balance rows (skipped %d).", total, skipped)


def seed_prior_year(db: Session) -> None:
    """
    Seed historical trial balances from `prior_year/*.csv` into the same
    TrialBalance table. Used by the Cash Flow agent for opening balances
    on older periods (2024-11, 2024-12, 2025-01 etc.).
    """
    total, skipped = _seed_tb_folder(db, DATASET_DIR / "prior_year", "TB-PY")
    logger.info("Seeded %d prior-year trial balance rows (skipped %d).", total, skipped)


def seed_intercompany(db: Session) -> None:
    """Seed intercompany transactions from `intercompany/*.csv`."""
    folder = DATASET_DIR / "intercompany"
    if not folder.exists():
        logger.warning("Missing folder: %s", folder)
        return

    total, skipped = 0, 0
    for csv_file in sorted(folder.glob("*.csv")):
        df = _read_csv(csv_file)
        if df is None:
            continue

        for _, row in df.iterrows():
            obj = IntercompanyTransaction(
                transaction_id=_clean_str(row.get("transaction_id")),
                date=_parse_date(row.get("date")) or date.today(),
                selling_entity_id=_clean_str(row.get("selling_entity_id")),
                buying_entity_id=_clean_str(row.get("buying_entity_id")),
                description=_clean_str(row.get("description")) or None,
                amount=_parse_decimal(row.get("amount")),
                gl_account=_clean_str(row.get("gl_account")),
            )
            if _insert_row(db, obj, context=f"IC:{csv_file.name}"):
                total += 1
            else:
                skipped += 1

    db.flush()
    logger.info("Seeded %d intercompany transactions (skipped %d).", total, skipped)


def seed_budgets(db: Session) -> None:
    """Seed monthly budgets from `budgets/budgets_2026.csv`."""
    path = DATASET_DIR / "budgets" / "budgets_2026.csv"
    df = _read_csv(path)
    if df is None:
        return

    total, skipped = 0, 0
    for _, row in df.iterrows():
        obj = Budget(
            company_id=_clean_str(row.get("company_id")),
            year=_parse_int(row.get("year"), 2026),
            month=_parse_int(row.get("month"), 1),
            account_code=_clean_str(row.get("account_code")),
            account_name=_clean_str(row.get("account_name")),
            budget_amount=_parse_decimal(row.get("budget_amount")),
        )
        if _insert_row(db, obj, context=f"BUD:{path.name}"):
            total += 1
        else:
            skipped += 1

    db.flush()
    logger.info("Seeded %d budget rows (skipped %d).", total, skipped)


def seed_bank_statements(db: Session) -> None:
    """Seed bank statements from `bank_statements/*.csv`."""
    folder = DATASET_DIR / "bank_statements"
    if not folder.exists():
        logger.warning("Missing folder: %s", folder)
        return

    total, skipped = 0, 0
    for csv_file in sorted(folder.glob("*.csv")):
        df = _read_csv(csv_file)
        if df is None:
            continue

        for _, row in df.iterrows():
            obj = BankStatement(
                company_id=_clean_str(row.get("company_id")),
                date=_parse_date(row.get("date")) or date.today(),
                description=_clean_str(row.get("description")) or None,
                debit=_parse_decimal(row.get("debit")),
                credit=_parse_decimal(row.get("credit")),
                balance=_parse_decimal(row.get("balance")),
                period=_clean_str(row.get("period")),
            )
            if _insert_row(db, obj, context=f"BANK:{csv_file.name}"):
                total += 1
            else:
                skipped += 1

    db.flush()
    logger.info("Seeded %d bank statement rows (skipped %d).", total, skipped)


def seed_accruals(db: Session) -> None:
    """Seed accrual schedules from `accrual_schedules/accrual_schedules.csv`."""
    path = DATASET_DIR / "accrual_schedules" / "accrual_schedules.csv"
    df = _read_csv(path)
    if df is None:
        return

    total, skipped = 0, 0
    for _, row in df.iterrows():
        obj = AccrualSchedule(
            company_id=_clean_str(row.get("company_id")),
            accrual_type=_clean_str(row.get("accrual_type")),
            gl_account=_clean_str(row.get("gl_account")),
            frequency=_clean_str(row.get("frequency")),
            amount=_parse_decimal(row.get("amount")),
            last_booked_date=_parse_date(row.get("last_booked_date")),
        )
        if _insert_row(db, obj, context=f"ACCR:{path.name}"):
            total += 1
        else:
            skipped += 1

    db.flush()
    logger.info("Seeded %d accrual schedules (skipped %d).", total, skipped)


def seed_revenue_contracts(db: Session) -> None:
    """Seed ASC-606 revenue contracts from `contracts/*_contracts.json`."""
    folder = DATASET_DIR / "contracts"
    if not folder.exists():
        logger.warning("Missing folder: %s", folder)
        return

    total, skipped = 0, 0
    for json_file in sorted(folder.glob("*.json")):
        records = _load_json_list(json_file)
        for rec in records:
            # data uses "company" key, not "company_id"
            company_id = _clean_str(rec.get("company") or rec.get("company_id"))
            contract_id = _clean_str(rec.get("contract_id"))
            if not company_id or not contract_id:
                logger.warning("Contract missing id/company, skipping: %s", rec)
                skipped += 1
                continue

            obj = RevenueContract(
                contract_id=contract_id,
                company_id=company_id,
                customer=_clean_str(rec.get("customer")),
                start_date=_parse_date(rec.get("start_date")) or date.today(),
                end_date=_parse_date(rec.get("end_date")) or date.today(),
                total_contract_value=_parse_decimal(rec.get("total_contract_value")),
                billing_schedule=_clean_str(rec.get("billing_schedule")) or None,
                performance_obligations=rec.get("performance_obligations") or [],
            )
            if _insert_row(db, obj, context=f"CNTR:{json_file.name}"):
                total += 1
            else:
                skipped += 1

    db.flush()
    logger.info("Seeded %d revenue contracts (skipped %d).", total, skipped)


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------
def _is_already_seeded(db: Session) -> bool:
    """Returns True if `companies` already contains rows."""
    return db.scalar(select(Company.id).limit(1)) is not None


def run_seed() -> None:
    logger.info("Dataset directory: %s", DATASET_DIR)
    if not DATASET_DIR.exists():
        logger.error("Dataset directory does not exist: %s", DATASET_DIR)
        raise SystemExit(1)

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        if _is_already_seeded(db):
            logger.info("Database already contains companies — skipping seed.")
            logger.info("To re-seed from scratch: docker compose down -v && docker compose up -d --build")
            return

        seed_companies(db)
        seed_trial_balances(db)
        seed_prior_year(db)
        seed_intercompany(db)
        seed_budgets(db)
        seed_bank_statements(db)
        seed_accruals(db)
        seed_revenue_contracts(db)
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