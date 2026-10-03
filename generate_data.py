#!/usr/bin/env python3
"""
Synthetic portfolio data generator.

Produces realistic multi-entity financial data for demonstration and testing:
    - Trial balances (current + prior period)
    - Monthly budgets
    - Intercompany transactions
    - Bank statements
    - Accrual schedules
    - ASC 606 revenue contracts

All entities, customers, and vendors are 100% synthetic — no real company
names, no association with any real firm or dataset.

Usage:
    python generate_data.py --output ./collected_data
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
from faker import Faker

# Deterministic output — same seed = same dataset
np.random.seed(20260101)
fake = Faker()
Faker.seed(20260101)

# =============================================================================
# PORTFOLIO DEFINITION
# =============================================================================
# Eight generic entities spanning SaaS, Manufacturing, Retail, Healthcare,
# Transportation, Distribution, Professional Services, and Sustainable
# Packaging. Numeric parameters mirror industry-typical ratios.
# =============================================================================

COMPANIES = [
    {
        "id": "alpha_saas",
        "name": "Alpha SaaS Solutions",
        "industry": "SaaS",
        "revenue_annual": 45_000_000,
        "employees": 180,
        "has_inventory": False,
        "gross_margin": 0.72,
        "growth_rate": 0.35,
    },
    {
        "id": "beta_manufacturing",
        "name": "Beta Manufacturing Co",
        "industry": "Manufacturing",
        "revenue_annual": 120_000_000,
        "employees": 450,
        "has_inventory": True,
        "gross_margin": 0.32,
        "growth_rate": 0.08,
    },
    {
        "id": "gamma_retail",
        "name": "Gamma Retail Group",
        "industry": "Retail",
        "revenue_annual": 200_000_000,
        "employees": 1200,
        "has_inventory": True,
        "gross_margin": 0.38,
        "growth_rate": 0.12,
    },
    {
        "id": "delta_healthcare",
        "name": "Delta Healthcare Services",
        "industry": "Healthcare Services",
        "revenue_annual": 35_000_000,
        "employees": 220,
        "has_inventory": False,
        "gross_margin": 0.55,
        "growth_rate": 0.18,
    },
    {
        "id": "epsilon_logistics",
        "name": "Epsilon Logistics Corp",
        "industry": "Transportation",
        "revenue_annual": 80_000_000,
        "employees": 350,
        "has_inventory": False,
        "gross_margin": 0.22,
        "growth_rate": 0.15,
    },
    {
        "id": "zeta_distribution",
        "name": "Zeta Distribution Inc",
        "industry": "Distribution",
        "revenue_annual": 150_000_000,
        "employees": 280,
        "has_inventory": True,
        "gross_margin": 0.18,
        "growth_rate": 0.06,
    },
    {
        "id": "eta_analytics",
        "name": "Eta Analytics Partners",
        "industry": "Professional Services",
        "revenue_annual": 25_000_000,
        "employees": 95,
        "has_inventory": False,
        "gross_margin": 0.65,
        "growth_rate": 0.22,
    },
    {
        "id": "theta_packaging",
        "name": "Theta Sustainable Packaging",
        "industry": "Manufacturing",
        "revenue_annual": 60_000_000,
        "employees": 310,
        "has_inventory": True,
        "gross_margin": 0.28,
        "growth_rate": 0.25,
    },
]

# =============================================================================
# CHART OF ACCOUNTS
# =============================================================================

CHART_OF_ACCOUNTS = {
    "assets": [
        (1000, "Cash and Cash Equivalents", "Asset"),
        (1100, "Accounts Receivable", "Asset"),
        (1150, "Allowance for Doubtful Accounts", "Asset"),
        (1200, "Inventory - Raw Materials", "Asset"),
        (1210, "Inventory - Work in Process", "Asset"),
        (1220, "Inventory - Finished Goods", "Asset"),
        (1300, "Prepaid Expenses", "Asset"),
        (1400, "Property, Plant & Equipment", "Asset"),
        (1450, "Accumulated Depreciation", "Asset"),
        (1500, "Intangible Assets", "Asset"),
        (1550, "Accumulated Amortization", "Asset"),
    ],
    "liabilities": [
        (2000, "Accounts Payable", "Liability"),
        (2100, "Accrued Expenses", "Liability"),
        (2150, "Accrued Payroll", "Liability"),
        (2200, "Deferred Revenue", "Liability"),
        (2300, "Current Portion of Long-term Debt", "Liability"),
        (2400, "Long-term Debt", "Liability"),
    ],
    "equity": [
        (3000, "Common Stock", "Equity"),
        (3100, "Retained Earnings", "Equity"),
    ],
    "revenue": [
        (4000, "Product Revenue", "Revenue"),
        (4100, "Service Revenue", "Revenue"),
        (4200, "Subscription Revenue", "Revenue"),
    ],
    "cogs": [
        (5000, "Cost of Goods Sold - Materials", "COGS"),
        (5100, "Cost of Goods Sold - Labor", "COGS"),
        (5200, "Cost of Goods Sold - Overhead", "COGS"),
        (5300, "Cost of Services", "COGS"),
    ],
    "opex": [
        (6000, "Salaries and Wages", "Operating Expense"),
        (6100, "Employee Benefits", "Operating Expense"),
        (6200, "Rent", "Operating Expense"),
        (6300, "Utilities", "Operating Expense"),
        (6400, "Insurance", "Operating Expense"),
        (6500, "Professional Fees", "Operating Expense"),
        (7000, "Sales and Marketing", "Operating Expense"),
        (7100, "Advertising", "Operating Expense"),
        (7200, "Travel and Entertainment", "Operating Expense"),
        (8000, "Research and Development", "Operating Expense"),
        (9000, "Depreciation Expense", "Operating Expense"),
        (9100, "Amortization Expense", "Operating Expense"),
        (9500, "Interest Expense", "Operating Expense"),
    ],
}


def get_chart_of_accounts(industry: str, has_inventory: bool) -> list[tuple[int, str, str]]:
    coa: list[tuple[int, str, str]] = []
    coa.extend(CHART_OF_ACCOUNTS["assets"][:2])
    if has_inventory:
        coa.extend(CHART_OF_ACCOUNTS["assets"][2:6])
    else:
        coa.append(CHART_OF_ACCOUNTS["assets"][2])
    coa.extend(CHART_OF_ACCOUNTS["assets"][6:])
    coa.extend(CHART_OF_ACCOUNTS["liabilities"])
    coa.extend(CHART_OF_ACCOUNTS["equity"])

    if industry == "SaaS":
        coa.append((4200, "Subscription Revenue", "Revenue"))
        coa.append((4100, "Professional Services Revenue", "Revenue"))
    elif industry in ("Manufacturing", "Distribution"):
        coa.append((4000, "Product Sales", "Revenue"))
    elif industry == "Retail":
        coa.append((4000, "Store Revenue", "Revenue"))
    else:
        coa.append((4100, "Service Revenue", "Revenue"))

    if has_inventory:
        coa.extend(CHART_OF_ACCOUNTS["cogs"][:3])
    else:
        coa.append((5300, "Cost of Services", "COGS"))

    coa.extend(CHART_OF_ACCOUNTS["opex"])
    return coa


# =============================================================================
# TRIAL BALANCE
# =============================================================================

def generate_trial_balance(company: dict, period_date: datetime) -> pd.DataFrame:
    coa = get_chart_of_accounts(company["industry"], company["has_inventory"])
    monthly_revenue = company["revenue_annual"] / 12 * np.random.uniform(0.90, 1.10)
    gross_margin = company["gross_margin"]
    rows: list[dict] = []

    for code, name, atype in coa:
        if atype == "Revenue":
            amt = monthly_revenue if str(code).startswith("4") else 0
            rows.append({"account_code": code, "account_name": name,
                         "debit": 0, "credit": amt, "balance": -amt,
                         "account_type": atype})
        elif atype == "COGS":
            cogs = monthly_revenue * (1 - gross_margin)
            amt = cogs / 3 if company["has_inventory"] else cogs
            rows.append({"account_code": code, "account_name": name,
                         "debit": amt, "credit": 0, "balance": amt,
                         "account_type": atype})
        elif atype == "Operating Expense":
            gp = monthly_revenue * gross_margin
            if "Salaries" in name or "Wages" in name:
                amt = gp * 0.35
            elif "Benefits" in name:
                amt = gp * 0.08
            elif "Rent" in name:
                amt = gp * 0.05
            elif "Sales and Marketing" in name:
                amt = gp * (0.15 if company["industry"] == "SaaS" else 0.08)
            elif "Research and Development" in name:
                amt = gp * 0.12 if company["industry"] in ("SaaS", "Manufacturing") else 0
            elif "Depreciation" in name or "Amortization" in name:
                amt = gp * 0.03
            else:
                amt = gp * np.random.uniform(0.01, 0.05)
            rows.append({"account_code": code, "account_name": name,
                         "debit": amt, "credit": 0, "balance": amt,
                         "account_type": atype})
        elif atype == "Asset":
            if "Cash" in name:
                amt = monthly_revenue * np.random.uniform(1.5, 3.0)
            elif "Receivable" in name:
                amt = monthly_revenue * np.random.uniform(1.2, 2.5)
            elif "Allowance" in name:
                amt = monthly_revenue * 0.02
            elif "Inventory" in name:
                amt = monthly_revenue * np.random.uniform(0.5, 1.5) if company["has_inventory"] else 0
            elif "Property" in name or "PPE" in name:
                amt = company["revenue_annual"] * 0.4
            elif "Accumulated Depreciation" in name:
                amt = company["revenue_annual"] * 0.15
            elif "Intangible" in name:
                amt = company["revenue_annual"] * 0.3
            elif "Accumulated Amortization" in name:
                amt = company["revenue_annual"] * 0.08
            else:
                amt = monthly_revenue * np.random.uniform(0.1, 0.5)
            if "Accumulated" in name or "Allowance" in name:
                rows.append({"account_code": code, "account_name": name,
                             "debit": 0, "credit": amt, "balance": -amt,
                             "account_type": atype})
            else:
                rows.append({"account_code": code, "account_name": name,
                             "debit": amt, "credit": 0, "balance": amt,
                             "account_type": atype})
        elif atype == "Liability":
            if "Accounts Payable" in name:
                amt = monthly_revenue * np.random.uniform(0.4, 1.0)
            elif "Accrued" in name:
                amt = monthly_revenue * np.random.uniform(0.1, 0.3)
            elif "Deferred Revenue" in name:
                amt = monthly_revenue * np.random.uniform(0.5, 2.0) if company["industry"] == "SaaS" else 0
            elif "Debt" in name:
                amt = company["revenue_annual"] * np.random.uniform(0.5, 2.0)
            else:
                amt = monthly_revenue * np.random.uniform(0.1, 0.5)
            rows.append({"account_code": code, "account_name": name,
                         "debit": 0, "credit": amt, "balance": -amt,
                         "account_type": atype})
        elif atype == "Equity":
            amt = company["revenue_annual"] * 0.5 if "Common Stock" in name else 0
            rows.append({"account_code": code, "account_name": name,
                         "debit": 0, "credit": amt, "balance": -amt,
                         "account_type": atype})

    df = pd.DataFrame(rows)
    # Balance with retained earnings
    diff = df["debit"].sum() - df["credit"].sum()
    re_mask = df["account_name"] == "Retained Earnings"
    df.loc[re_mask, "credit"] += diff
    df.loc[re_mask, "balance"] = -df.loc[re_mask, "credit"]
    df["company_id"] = company["id"]
    df["company_name"] = company["name"]
    df["period"] = period_date.strftime("%Y-%m")
    return df


# =============================================================================
# BUDGETS
# =============================================================================

def generate_budgets(companies: list[dict], year: int = 2026) -> pd.DataFrame:
    rows: list[dict] = []
    for company in companies:
        coa = get_chart_of_accounts(company["industry"], company["has_inventory"])
        for month in range(1, 13):
            monthly = company["revenue_annual"] / 12
            gm = company["gross_margin"]
            for code, name, atype in coa:
                if atype == "Revenue":
                    amt = monthly
                elif atype == "COGS":
                    cogs = monthly * (1 - gm)
                    amt = cogs / 3 if company["has_inventory"] else cogs
                elif atype == "Operating Expense":
                    gp = monthly * gm
                    if "Salaries" in name:
                        amt = gp * 0.35
                    elif "Benefits" in name:
                        amt = gp * 0.08
                    elif "Sales and Marketing" in name:
                        amt = gp * (0.15 if company["industry"] == "SaaS" else 0.08)
                    else:
                        amt = gp * np.random.uniform(0.02, 0.06)
                else:
                    continue
                rows.append({
                    "company_id": company["id"], "company_name": company["name"],
                    "year": year, "month": month,
                    "account_code": code, "account_name": name,
                    "budget_amount": amt,
                })
    return pd.DataFrame(rows)


# =============================================================================
# INTERCOMPANY TRANSACTIONS
# =============================================================================

def generate_intercompany_transactions(
    companies: list[dict], period_date: datetime, num_transactions: int = 1000
) -> pd.DataFrame:
    rows: list[dict] = []
    for i in range(num_transactions):
        seller, buyer = np.random.choice(companies, size=2, replace=False)
        txn_types = [
            ("Management Fees", 7000, np.random.uniform(10000, 50000)),
            ("Shared Services", 6500, np.random.uniform(20000, 100000)),
            ("Inventory/Supplies", 5000, np.random.uniform(50000, 250000)),
            ("Software Licenses", 7000, np.random.uniform(5000, 25000)),
        ]
        desc, gl, amt = txn_types[i % len(txn_types)]
        txn_date = period_date + timedelta(days=int(np.random.randint(1, 28)))
        rows.append({
            "transaction_id": f"IC-{period_date.strftime('%Y%m')}-{i+1:03d}",
            "date": txn_date.strftime("%Y-%m-%d"),
            "selling_entity_id": seller["id"],
            "selling_entity_name": seller["name"],
            "buying_entity_id": buyer["id"],
            "buying_entity_name": buyer["name"],
            "description": desc,
            "amount": amt,
            "gl_account": gl,
        })
    return pd.DataFrame(rows)


# =============================================================================
# BANK STATEMENTS
# =============================================================================

def generate_bank_statements(company: dict, period_date: datetime) -> pd.DataFrame:
    days_in_month = 30
    balance = company["revenue_annual"] / 12 * np.random.uniform(1.5, 3.0)
    rows = [{
        "date": period_date.strftime("%Y-%m-%d"),
        "description": "Beginning Balance",
        "debit": "", "credit": "", "balance": balance,
    }]
    for day in range(1, days_in_month):
        d = period_date + timedelta(days=day)
        if np.random.random() < 0.25:
            amt = np.random.uniform(50_000, 500_000)
            balance += amt
            rows.append({"date": d.strftime("%Y-%m-%d"),
                         "description": f"Customer Payment - {fake.company()}",
                         "debit": "", "credit": amt, "balance": balance})
        if day in (15, 30):
            amt = company["revenue_annual"] * 0.35 / 24
            balance -= amt
            rows.append({"date": d.strftime("%Y-%m-%d"),
                         "description": "Payroll",
                         "debit": amt, "credit": "", "balance": balance})
        if np.random.random() < 0.30:
            amt = np.random.uniform(10_000, 200_000)
            balance -= amt
            rows.append({"date": d.strftime("%Y-%m-%d"),
                         "description": f"Vendor Payment - {fake.company()}",
                         "debit": amt, "credit": "", "balance": balance})
    df = pd.DataFrame(rows)
    df["company_id"] = company["id"]
    df["company_name"] = company["name"]
    df["period"] = period_date.strftime("%Y-%m")
    return df


# =============================================================================
# ACCRUAL SCHEDULES
# =============================================================================

def generate_accrual_schedules(companies: list[dict], num_records: int = 1000) -> pd.DataFrame:
    rows: list[dict] = []
    per_company = max(5, num_records // len(companies) + 1)
    extra_types = [
        ("Legal Fees Accrual", 6500, "monthly"),
        ("Audit Fees Accrual", 6500, "annual"),
        ("Property Tax Accrual", 6300, "annual"),
        ("Software License Accrual", 6300, "monthly"),
        ("Marketing Campaign Accrual", 7000, "quarterly"),
        ("Travel Expense Accrual", 7200, "monthly"),
        ("Commission Accrual", 6000, "monthly"),
        ("Contractor Fees Accrual", 6500, "monthly"),
        ("Interest Expense Accrual", 9500, "monthly"),
        ("Freight/Shipping Accrual", 5200, "monthly"),
        ("Warranty Provision", 5200, "annual"),
        ("Server Hosting Accrual", 6300, "monthly"),
        ("Equipment Maintenance", 6300, "quarterly"),
        ("Consulting Fees Accrual", 6500, "monthly"),
    ]
    for company in companies:
        monthly = company["revenue_annual"] / 12
        templates = [
            ("Rent Expense", 6200, "monthly", monthly * 0.05),
            ("Utilities", 6300, "monthly", monthly * 0.02),
            ("Insurance", 6400, "quarterly", monthly * 0.12),
            ("Professional Fees", 6500, "monthly", monthly * 0.03),
            ("Bonus Accrual", 6000, "annual", company["revenue_annual"] * 0.05),
        ]
        for name, gl, freq, amt in templates:
            rows.append({
                "company_id": company["id"], "company_name": company["name"],
                "accrual_type": name, "gl_account": gl,
                "frequency": freq, "amount": amt,
                "last_booked_date": "2025-12-31",
            })
        for _ in range(per_company - len(templates)):
            name, gl, freq = extra_types[np.random.randint(0, len(extra_types))]
            vendor = fake.company()
            amt = monthly * np.random.uniform(0.001, 0.02)
            m = np.random.randint(1, 13)
            d = np.random.randint(1, 28)
            rows.append({
                "company_id": company["id"], "company_name": company["name"],
                "accrual_type": f"{name} - {vendor}",
                "gl_account": gl, "frequency": freq, "amount": amt,
                "last_booked_date": f"2025-{m:02d}-{d:02d}",
            })
    return pd.DataFrame(rows)


# =============================================================================
# REVENUE CONTRACTS (ASC 606)
# =============================================================================

def generate_revenue_contracts(company: dict, num_contracts: int = 25) -> list[dict]:
    if company["industry"] == "SaaS":
        obligations = [("Software subscription", "ratable", 0.85),
                       ("Implementation services", "milestone", 0.15)]
    elif company["industry"] in ("Professional Services", "Healthcare Services"):
        obligations = [("Service retainer", "ratable", 0.60),
                       ("Project delivery", "milestone", 0.40)]
    else:
        obligations = [("Product supply agreement", "point_in_time", 0.80),
                       ("Installation & support", "ratable", 0.20)]

    avg_value = company["revenue_annual"] * 0.02
    contracts: list[dict] = []

    for i in range(num_contracts):
        start = datetime(2024, 1, 1) + timedelta(days=int(np.random.randint(0, 730)))
        term_years = int(np.random.choice([1, 2, 3]))
        end = start + timedelta(days=365 * term_years)
        total_value = round(avg_value * np.random.uniform(0.3, 2.5), 2)

        # Planted allocation drift on ~15% of contracts so the RevRec agent
        # has real allocation errors to surface
        skew = np.random.uniform(0.80, 0.92) if np.random.random() < 0.15 else 1.0

        pos: list[dict] = []
        for desc, method, share in obligations:
            po = {
                "description": desc,
                "revenue_recognition": method,
                "value": round(total_value * share * skew, 2),
            }
            if method == "milestone":
                if end < datetime(2026, 1, 31) and np.random.random() < 0.3:
                    po["completion_percentage"] = int(np.random.randint(60, 95))
                else:
                    po["completion_percentage"] = int(np.random.choice([25, 50, 75, 100]))
            pos.append(po)

        contracts.append({
            "contract_id": f"CNTR-{start.year}-{company['id'][:4].upper()}-{i+1:03d}",
            "customer": fake.company(),
            "company": company["id"],
            "start_date": start.strftime("%Y-%m-%d"),
            "end_date": end.strftime("%Y-%m-%d"),
            "total_contract_value": total_value,
            "billing_schedule": str(np.random.choice(["monthly", "quarterly", "annual"])),
            "performance_obligations": pos,
        })
    return contracts


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic portfolio data.")
    ap.add_argument("--output", default="./collected_data",
                    help="Output directory (default: ./collected_data)")
    args = ap.parse_args()

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[gen] writing to {out.resolve()}")

    periods = [
        datetime(2025, 11, 1),
        datetime(2025, 12, 1),
        datetime(2026, 1, 1),
    ]
    prior_periods = [
        datetime(2024, 11, 1),
        datetime(2024, 12, 1),
        datetime(2025, 1, 1),
    ]

    for sub in ("trial_balances", "budgets", "prior_year", "intercompany",
                "bank_statements", "accrual_schedules", "contracts"):
        (out / sub).mkdir(exist_ok=True)

    # ---- Trial balances ------------------------------------------------
    print("[gen] trial balances (current period)")
    for period in periods:
        for company in COMPANIES:
            tb = generate_trial_balance(company, period)
            tb.to_csv(out / "trial_balances" /
                      f"{company['id']}_{period.strftime('%Y_%m')}.csv",
                      index=False)

    # ---- Prior year ----------------------------------------------------
    print("[gen] trial balances (prior year)")
    for period in prior_periods:
        for company in COMPANIES:
            tb = generate_trial_balance(company, period)
            tb.to_csv(out / "prior_year" /
                      f"{company['id']}_{period.strftime('%Y_%m')}.csv",
                      index=False)

    # ---- Budgets -------------------------------------------------------
    print("[gen] budgets")
    budgets = generate_budgets(COMPANIES, year=2026)
    budgets.to_csv(out / "budgets" / "budgets_2026.csv", index=False)

    # ---- Intercompany --------------------------------------------------
    print("[gen] intercompany transactions")
    for period in periods:
        ic = generate_intercompany_transactions(COMPANIES, period)
        ic.to_csv(out / "intercompany" /
                  f"intercompany_{period.strftime('%Y_%m')}.csv", index=False)

    # ---- Bank statements ----------------------------------------------
    print("[gen] bank statements")
    for period in periods:
        for company in COMPANIES:
            bs = generate_bank_statements(company, period)
            bs.to_csv(out / "bank_statements" /
                      f"{company['id']}_{period.strftime('%Y_%m')}.csv", index=False)

    # ---- Accruals ------------------------------------------------------
    print("[gen] accrual schedules")
    accruals = generate_accrual_schedules(COMPANIES)
    accruals.to_csv(out / "accrual_schedules" / "accrual_schedules.csv", index=False)

    # ---- Contracts -----------------------------------------------------
    print("[gen] revenue contracts")
    for company in COMPANIES:
        contracts = generate_revenue_contracts(company)
        with open(out / "contracts" / f"{company['id']}_contracts.json", "w") as f:
            json.dump(contracts, f, indent=2)

    # ---- Company metadata ---------------------------------------------
    print("[gen] company metadata")
    metadata = pd.DataFrame(COMPANIES)
    metadata.to_json(out / "company_metadata.json", orient="records", indent=2)

    print("\n[done] synthetic portfolio generated.")
    print(f"[done] entities: {', '.join(c['name'] for c in COMPANIES)}")


if __name__ == "__main__":
    main()