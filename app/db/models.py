from __future__ import annotations

import uuid
from datetime import date
from decimal import Decimal
from typing import Any

from sqlalchemy import (
    JSON,
    Date,
    ForeignKey,
    Index,
    Numeric,
    String,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.database import Base


class Company(Base):
    __tablename__ = "companies"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    name: Mapped[str] = mapped_column(String(255), nullable=False, unique=True)
    industry: Mapped[str | None] = mapped_column(String(120))
    revenue_annual: Mapped[Decimal | None] = mapped_column(Numeric(18, 2))

    trial_balances: Mapped[list["TrialBalance"]] = relationship(back_populates="company", cascade="all, delete-orphan")
    accruals: Mapped[list["AccrualSchedule"]] = relationship(back_populates="company", cascade="all, delete-orphan")
    contracts: Mapped[list["RevenueContract"]] = relationship(back_populates="company", cascade="all, delete-orphan")


class TrialBalance(Base):
    __tablename__ = "trial_balances"
    __table_args__ = (
        UniqueConstraint("company_id", "period", "account_code", name="uq_tb_company_period_account"),
        Index("ix_tb_company_period", "company_id", "period"),
    )

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    period: Mapped[str] = mapped_column(String(7), nullable=False)  # YYYY-MM
    account_code: Mapped[str] = mapped_column(String(32), nullable=False)
    account_name: Mapped[str] = mapped_column(String(255), nullable=False)
    debit: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=Decimal("0.00"))
    credit: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=Decimal("0.00"))
    balance: Mapped[Decimal] = mapped_column(Numeric(18, 2), default=Decimal("0.00"))
    account_type: Mapped[str] = mapped_column(String(32), nullable=False)  # Asset/Liability/Equity/Revenue/Expense

    company: Mapped[Company] = relationship(back_populates="trial_balances")


class IntercompanyTransaction(Base):
    __tablename__ = "intercompany_transactions"
    __table_args__ = (Index("ix_ic_pair_date", "selling_entity_id", "buying_entity_id", "date"),)

    transaction_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    date: Mapped[date] = mapped_column(Date, nullable=False)
    selling_entity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"), nullable=False)
    buying_entity_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("companies.id", ondelete="RESTRICT"), nullable=False)
    description: Mapped[str | None] = mapped_column(String(500))
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    gl_account: Mapped[str] = mapped_column(String(32), nullable=False)


class AccrualSchedule(Base):
    __tablename__ = "accrual_schedules"
    __table_args__ = (Index("ix_accrual_company_type", "company_id", "accrual_type"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    accrual_type: Mapped[str] = mapped_column(String(120), nullable=False)
    gl_account: Mapped[str] = mapped_column(String(32), nullable=False)
    frequency: Mapped[str] = mapped_column(String(32), nullable=False)  # monthly / quarterly / annual
    amount: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    last_booked_date: Mapped[date | None] = mapped_column(Date)

    company: Mapped[Company] = relationship(back_populates="accruals")


class RevenueContract(Base):
    __tablename__ = "revenue_contracts"
    __table_args__ = (Index("ix_contract_company", "company_id"),)

    contract_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    company_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("companies.id", ondelete="CASCADE"), nullable=False)
    customer: Mapped[str] = mapped_column(String(255), nullable=False)
    start_date: Mapped[date] = mapped_column(Date, nullable=False)
    end_date: Mapped[date] = mapped_column(Date, nullable=False)
    total_contract_value: Mapped[Decimal] = mapped_column(Numeric(18, 2), nullable=False)
    billing_schedule: Mapped[str | None] = mapped_column(String(64))  # e.g. monthly, milestone, upfront
    performance_obligations: Mapped[list[dict[str, Any]] | None] = mapped_column(JSON)

    company: Mapped[Company] = relationship(back_populates="contracts")