#!/usr/bin/env python3
"""
Sample Data Generator for Assignment 1: Month-End Close
Generates realistic financial data for 8 portfolio companies

Usage:
    python generate_assignment1_data.py --output ./assignment1_data

Requirements:
    pip install pandas numpy faker openpyxl --break-system-packages
"""

import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
from pathlib import Path
import json
import argparse

# Set random seed for reproducibility
np.random.seed(42)
fake = Faker()
Faker.seed(42)

# Portfolio company configurations
COMPANIES = [
    {
        'id': 'techforge_saas',
        'name': 'TechForge SaaS',
        'revenue_annual': 45_000_000,
        'industry': 'SaaS',
        'employees': 180,
        'has_inventory': False,
        'gross_margin': 0.72,
        'growth_rate': 0.35
    },
    {
        'id': 'precisionmfg_inc',
        'name': 'PrecisionMfg Inc',
        'revenue_annual': 120_000_000,
        'industry': 'Manufacturing',
        'employees': 450,
        'has_inventory': True,
        'gross_margin': 0.32,
        'growth_rate': 0.08
    },
    {
        'id': 'retailco',
        'name': 'RetailCo',
        'revenue_annual': 200_000_000,
        'industry': 'Retail',
        'employees': 1200,
        'has_inventory': True,
        'gross_margin': 0.38,
        'growth_rate': 0.12
    },
    {
        'id': 'healthservices_plus',
        'name': 'HealthServices Plus',
        'revenue_annual': 35_000_000,
        'industry': 'Healthcare Services',
        'employees': 220,
        'has_inventory': False,
        'gross_margin': 0.55,
        'growth_rate': 0.18
    },
    {
        'id': 'logisticspro',
        'name': 'LogisticsPro',
        'revenue_annual': 80_000_000,
        'industry': 'Transportation',
        'employees': 350,
        'has_inventory': False,
        'gross_margin': 0.22,
        'growth_rate': 0.15
    },
    {
        'id': 'industrialsupply_co',
        'name': 'IndustrialSupply Co',
        'revenue_annual': 150_000_000,
        'industry': 'Distribution',
        'employees': 280,
        'has_inventory': True,
        'gross_margin': 0.18,
        'growth_rate': 0.06
    },
    {
        'id': 'dataanalytics_corp',
        'name': 'DataAnalytics Corp',
        'revenue_annual': 25_000_000,
        'industry': 'Professional Services',
        'employees': 95,
        'has_inventory': False,
        'gross_margin': 0.65,
        'growth_rate': 0.22
    },
    {
        'id': 'ecopackaging_ltd',
        'name': 'EcoPackaging Ltd',
        'revenue_annual': 60_000_000,
        'industry': 'Manufacturing',
        'employees': 310,
        'has_inventory': True,
        'gross_margin': 0.28,
        'growth_rate': 0.25
    }
]

# Standard Chart of Accounts
CHART_OF_ACCOUNTS = {
    # Assets (1000-1999)
    'assets': [
        (1000, 'Cash and Cash Equivalents', 'Asset'),
        (1100, 'Accounts Receivable', 'Asset'),
        (1150, 'Allowance for Doubtful Accounts', 'Asset'),
        (1200, 'Inventory - Raw Materials', 'Asset'),
        (1210, 'Inventory - Work in Process', 'Asset'),
        (1220, 'Inventory - Finished Goods', 'Asset'),
        (1300, 'Prepaid Expenses', 'Asset'),
        (1400, 'Property, Plant & Equipment', 'Asset'),
        (1450, 'Accumulated Depreciation', 'Asset'),
        (1500, 'Intangible Assets', 'Asset'),
        (1550, 'Accumulated Amortization', 'Asset'),
    ],
    # Liabilities (2000-2999)
    'liabilities': [
        (2000, 'Accounts Payable', 'Liability'),
        (2100, 'Accrued Expenses', 'Liability'),
        (2150, 'Accrued Payroll', 'Liability'),
        (2200, 'Deferred Revenue', 'Liability'),
        (2300, 'Current Portion of Long-term Debt', 'Liability'),
        (2400, 'Long-term Debt', 'Liability'),
    ],
    # Equity (3000-3999)
    'equity': [
        (3000, 'Common Stock', 'Equity'),
        (3100, 'Retained Earnings', 'Equity'),
    ],
    # Revenue (4000-4999)
    'revenue': [
        (4000, 'Product Revenue', 'Revenue'),
        (4100, 'Service Revenue', 'Revenue'),
        (4200, 'Subscription Revenue', 'Revenue'),
    ],
    # COGS (5000-5999)
    'cogs': [
        (5000, 'Cost of Goods Sold - Materials', 'COGS'),
        (5100, 'Cost of Goods Sold - Labor', 'COGS'),
        (5200, 'Cost of Goods Sold - Overhead', 'COGS'),
        (5300, 'Cost of Services', 'COGS'),
    ],
    # Operating Expenses (6000-9999)
    'opex': [
        (6000, 'Salaries and Wages', 'Operating Expense'),
        (6100, 'Employee Benefits', 'Operating Expense'),
        (6200, 'Rent', 'Operating Expense'),
        (6300, 'Utilities', 'Operating Expense'),
        (6400, 'Insurance', 'Operating Expense'),
        (6500, 'Professional Fees', 'Operating Expense'),
        (7000, 'Sales and Marketing', 'Operating Expense'),
        (7100, 'Advertising', 'Operating Expense'),
        (7200, 'Travel and Entertainment', 'Operating Expense'),
        (8000, 'Research and Development', 'Operating Expense'),
        (9000, 'Depreciation Expense', 'Operating Expense'),
        (9100, 'Amortization Expense', 'Operating Expense'),
        (9500, 'Interest Expense', 'Operating Expense'),
    ]
}


def get_chart_of_accounts(industry, has_inventory):
    """Get appropriate chart of accounts based on company profile"""
    coa = []
    
    # All companies get basic structure
    coa.extend(CHART_OF_ACCOUNTS['assets'][:2])  # Cash, AR
    
    # Inventory accounts only for companies with inventory
    if has_inventory:
        coa.extend(CHART_OF_ACCOUNTS['assets'][2:6])  # Allowance, Inventory accounts
    else:
        coa.append(CHART_OF_ACCOUNTS['assets'][2])  # Just Allowance
    
    # Continue with rest of assets
    coa.extend(CHART_OF_ACCOUNTS['assets'][6:])
    
    # All liabilities and equity
    coa.extend(CHART_OF_ACCOUNTS['liabilities'])
    coa.extend(CHART_OF_ACCOUNTS['equity'])
    
    # Revenue accounts vary by industry
    if industry == 'SaaS':
        coa.append((4200, 'Subscription Revenue', 'Revenue'))
        coa.append((4100, 'Professional Services Revenue', 'Revenue'))
    elif industry in ['Manufacturing', 'Distribution']:
        coa.append((4000, 'Product Sales', 'Revenue'))
    elif industry == 'Retail':
        coa.append((4000, 'Store Revenue', 'Revenue'))
    else:
        coa.append((4100, 'Service Revenue', 'Revenue'))
    
    # COGS varies by industry
    if has_inventory:
        coa.extend(CHART_OF_ACCOUNTS['cogs'][:3])  # Materials, Labor, Overhead
    else:
        coa.append((5300, 'Cost of Services', 'COGS'))
    
    # All companies get operating expenses
    coa.extend(CHART_OF_ACCOUNTS['opex'])
    
    return coa


def generate_trial_balance(company, period_date):
    """Generate trial balance for a company for a specific period"""
    coa = get_chart_of_accounts(company['industry'], company['has_inventory'])
    
    monthly_revenue = company['revenue_annual'] / 12
    gross_margin = company['gross_margin']
    
    # Add some monthly variance (±10%)
    monthly_revenue *= np.random.uniform(0.90, 1.10)
    
    trial_balance = []
    
    for account_code, account_name, account_type in coa:
        if account_type == 'Revenue':
            # Revenue (credit balance)
            amount = monthly_revenue if '4' in str(account_code)[:1] else 0
            trial_balance.append({
                'account_code': account_code,
                'account_name': account_name,
                'debit': 0,
                'credit': amount,
                'balance': -amount,  # Credit balance is negative
                'account_type': account_type
            })
        
        elif account_type == 'COGS':
            # COGS (debit balance)
            cogs = monthly_revenue * (1 - gross_margin)
            amount = cogs / 3 if company['has_inventory'] else cogs  # Split across multiple accounts if inventory
            trial_balance.append({
                'account_code': account_code,
                'account_name': account_name,
                'debit': amount,
                'credit': 0,
                'balance': amount,
                'account_type': account_type
            })
        
        elif account_type == 'Operating Expense':
            # Operating expenses (debit balance)
            gross_profit = monthly_revenue * gross_margin
            
            if 'Salaries' in account_name or 'Wages' in account_name:
                amount = gross_profit * 0.35
            elif 'Benefits' in account_name:
                amount = gross_profit * 0.08
            elif 'Rent' in account_name:
                amount = gross_profit * 0.05
            elif 'Sales and Marketing' in account_name:
                amount = gross_profit * 0.15 if company['industry'] == 'SaaS' else gross_profit * 0.08
            elif 'Research and Development' in account_name:
                amount = gross_profit * 0.12 if company['industry'] in ['SaaS', 'Manufacturing'] else 0
            elif 'Depreciation' in account_name or 'Amortization' in account_name:
                amount = gross_profit * 0.03
            else:
                amount = gross_profit * np.random.uniform(0.01, 0.05)
            
            trial_balance.append({
                'account_code': account_code,
                'account_name': account_name,
                'debit': amount,
                'credit': 0,
                'balance': amount,
                'account_type': account_type
            })
        
        elif account_type == 'Asset':
            # Assets (debit balance)
            if 'Cash' in account_name:
                amount = monthly_revenue * np.random.uniform(1.5, 3.0)
            elif 'Receivable' in account_name:
                amount = monthly_revenue * np.random.uniform(1.2, 2.5)
            elif 'Allowance' in account_name:
                amount = monthly_revenue * 0.02  # 2% of revenue
            elif 'Inventory' in account_name:
                amount = monthly_revenue * np.random.uniform(0.5, 1.5) if company['has_inventory'] else 0
            elif 'PPE' in account_name or 'Property' in account_name:
                amount = company['revenue_annual'] * 0.4
            elif 'Accumulated Depreciation' in account_name:
                amount = company['revenue_annual'] * 0.15
            elif 'Intangible' in account_name:
                amount = company['revenue_annual'] * 0.3
            elif 'Accumulated Amortization' in account_name:
                amount = company['revenue_annual'] * 0.08
            else:
                amount = monthly_revenue * np.random.uniform(0.1, 0.5)
            
            # Contra-assets are credits
            if 'Accumulated' in account_name or 'Allowance' in account_name:
                trial_balance.append({
                    'account_code': account_code,
                    'account_name': account_name,
                    'debit': 0,
                    'credit': amount,
                    'balance': -amount,
                    'account_type': account_type
                })
            else:
                trial_balance.append({
                    'account_code': account_code,
                    'account_name': account_name,
                    'debit': amount,
                    'credit': 0,
                    'balance': amount,
                    'account_type': account_type
                })
        
        elif account_type == 'Liability':
            # Liabilities (credit balance)
            if 'Accounts Payable' in account_name:
                amount = monthly_revenue * np.random.uniform(0.4, 1.0)
            elif 'Accrued' in account_name:
                amount = monthly_revenue * np.random.uniform(0.1, 0.3)
            elif 'Deferred Revenue' in account_name:
                amount = monthly_revenue * np.random.uniform(0.5, 2.0) if company['industry'] == 'SaaS' else 0
            elif 'Debt' in account_name:
                amount = company['revenue_annual'] * np.random.uniform(0.5, 2.0)
            else:
                amount = monthly_revenue * np.random.uniform(0.1, 0.5)
            
            trial_balance.append({
                'account_code': account_code,
                'account_name': account_name,
                'debit': 0,
                'credit': amount,
                'balance': -amount,
                'account_type': account_type
            })
        
        elif account_type == 'Equity':
            # Equity (credit balance)
            if 'Common Stock' in account_name:
                amount = company['revenue_annual'] * 0.5
            else:  # Retained Earnings - balancing figure
                amount = 0  # Will calculate later
            
            trial_balance.append({
                'account_code': account_code,
                'account_name': account_name,
                'debit': 0,
                'credit': amount,
                'balance': -amount,
                'account_type': account_type
            })
    
    # Convert to DataFrame
    df = pd.DataFrame(trial_balance)
    
    # Balance the trial balance by adjusting retained earnings
    total_debits = df['debit'].sum()
    total_credits = df['credit'].sum()
    difference = total_debits - total_credits
    
    # Add difference to retained earnings to balance
    df.loc[df['account_name'] == 'Retained Earnings', 'credit'] += difference
    df.loc[df['account_name'] == 'Retained Earnings', 'balance'] = -df.loc[df['account_name'] == 'Retained Earnings', 'credit']
    
    # Add company and period info
    df['company_id'] = company['id']
    df['company_name'] = company['name']
    df['period'] = period_date.strftime('%Y-%m')
    
    return df


def generate_budgets(companies, year=2026):
    """Generate annual budgets for all companies"""
    budget_data = []
    
    for company in companies:
        coa = get_chart_of_accounts(company['industry'], company['has_inventory'])
        
        for month in range(1, 13):
            monthly_revenue_budget = company['revenue_annual'] / 12
            gross_margin = company['gross_margin']
            
            for account_code, account_name, account_type in coa:
                if account_type == 'Revenue':
                    amount = monthly_revenue_budget
                elif account_type == 'COGS':
                    cogs = monthly_revenue_budget * (1 - gross_margin)
                    amount = cogs / 3 if company['has_inventory'] else cogs
                elif account_type == 'Operating Expense':
                    gross_profit = monthly_revenue_budget * gross_margin
                    
                    if 'Salaries' in account_name:
                        amount = gross_profit * 0.35
                    elif 'Benefits' in account_name:
                        amount = gross_profit * 0.08
                    elif 'Sales and Marketing' in account_name:
                        amount = gross_profit * 0.15 if company['industry'] == 'SaaS' else gross_profit * 0.08
                    else:
                        amount = gross_profit * np.random.uniform(0.02, 0.06)
                else:
                    continue  # Skip balance sheet accounts in budget
                
                budget_data.append({
                    'company_id': company['id'],
                    'company_name': company['name'],
                    'year': year,
                    'month': month,
                    'account_code': account_code,
                    'account_name': account_name,
                    'budget_amount': amount
                })
    
    return pd.DataFrame(budget_data)


def generate_intercompany_transactions(companies, period_date, num_transactions=1000):
    """Generate intercompany transactions between portfolio companies"""
    transactions = []
    
    for i in range(num_transactions):
        # Pick two different companies
        seller, buyer = np.random.choice(companies, size=2, replace=False)
        
        # Generate realistic transaction
        transaction_types = [
            ('Management Fees', 7000, np.random.uniform(10000, 50000)),
            ('Shared Services', 6500, np.random.uniform(20000, 100000)),
            ('Inventory/Supplies', 5000, np.random.uniform(50000, 250000)),
            ('Software Licenses', 7000, np.random.uniform(5000, 25000)),
        ]
        
        desc, gl_account, amount = transaction_types[i % len(transaction_types)]
        
        transaction_date = period_date + timedelta(days=np.random.randint(1, 28))
        
        transactions.append({
            'transaction_id': f'IC-{period_date.strftime("%Y%m")}-{i+1:03d}',
            'date': transaction_date.strftime('%Y-%m-%d'),
            'selling_entity_id': seller['id'],
            'selling_entity_name': seller['name'],
            'buying_entity_id': buyer['id'],
            'buying_entity_name': buyer['name'],
            'description': desc,
            'amount': amount,
            'gl_account': gl_account
        })
    
    return pd.DataFrame(transactions)


def generate_bank_statements(company, period_date):
    """Generate mock bank statement for a company"""
    start_date = period_date
    days_in_month = 30
    
    # Starting balance
    balance = company['revenue_annual'] / 12 * np.random.uniform(1.5, 3.0)
    
    transactions = [{
        'date': start_date.strftime('%Y-%m-%d'),
        'description': 'Beginning Balance',
        'debit': '',
        'credit': '',
        'balance': balance
    }]
    
    # Generate daily transactions
    for day in range(1, days_in_month):
        date = start_date + timedelta(days=day)
        
        # Customer payments (random, 5-8 per month)
        if np.random.random() < 0.25:
            amount = np.random.uniform(50000, 500000)
            balance += amount
            transactions.append({
                'date': date.strftime('%Y-%m-%d'),
                'description': f'Customer Payment - {fake.company()}',
                'debit': '',
                'credit': amount,
                'balance': balance
            })
        
        # Payroll (2x per month)
        if day in [15, 30]:
            amount = company['revenue_annual'] * 0.35 / 24  # Semi-monthly payroll
            balance -= amount
            transactions.append({
                'date': date.strftime('%Y-%m-%d'),
                'description': 'Payroll',
                'debit': amount,
                'credit': '',
                'balance': balance
            })
        
        # Vendor payments (random)
        if np.random.random() < 0.30:
            amount = np.random.uniform(10000, 200000)
            balance -= amount
            transactions.append({
                'date': date.strftime('%Y-%m-%d'),
                'description': f'Vendor Payment - {fake.company()}',
                'debit': amount,
                'credit': '',
                'balance': balance
            })
    
    df = pd.DataFrame(transactions)
    df['company_id'] = company['id']
    df['company_name'] = company['name']
    df['period'] = period_date.strftime('%Y-%m')
    
    return df


def generate_accrual_schedules(companies, num_records=1000):
    """Generate standard accrual schedules"""
    accruals = []
    
    records_per_company = max(5, num_records // len(companies) + 1)
    
    additional_types = [
        ('Legal Fees Accrual', 6500, 'monthly'),
        ('Audit Fees Accrual', 6500, 'annual'),
        ('Property Tax Accrual', 6300, 'annual'),
        ('Software License Accrual', 6300, 'monthly'),
        ('Marketing Campaign Accrual', 7000, 'quarterly'),
        ('Travel Expense Accrual', 7200, 'monthly'),
        ('Commission Accrual', 6000, 'monthly'),
        ('Contractor Fees Accrual', 6500, 'monthly'),
        ('Interest Expense Accrual', 9500, 'monthly'),
        ('Freight/Shipping Accrual', 5200, 'monthly'),
        ('Warranty Provision', 5200, 'annual'),
        ('Server Hosting Accrual', 6300, 'monthly'),
        ('Equipment Maintenance', 6300, 'quarterly'),
        ('Consulting Fees Accrual', 6500, 'monthly')
    ]
    
    for company in companies:
        monthly_revenue = company['revenue_annual'] / 12
        
        # Standard accruals
        accrual_templates = [
            ('Rent Expense', 6200, 'monthly', monthly_revenue * 0.05),
            ('Utilities', 6300, 'monthly', monthly_revenue * 0.02),
            ('Insurance', 6400, 'quarterly', monthly_revenue * 0.12),
            ('Professional Fees', 6500, 'monthly', monthly_revenue * 0.03),
            ('Bonus Accrual', 6000, 'annual', company['revenue_annual'] * 0.05),
        ]
        
        for accrual_type, gl_account, frequency, amount in accrual_templates:
            accruals.append({
                'company_id': company['id'],
                'company_name': company['name'],
                'accrual_type': accrual_type,
                'gl_account': gl_account,
                'frequency': frequency,
                'amount': amount,
                'last_booked_date': '2025-12-31'
            })
            
        for _ in range(records_per_company - len(accrual_templates)):
            type_name, gl_acc, freq = additional_types[np.random.randint(0, len(additional_types))]
            vendor = fake.company()
            accrual_name = f"{type_name} - {vendor}"
            amt = monthly_revenue * np.random.uniform(0.001, 0.02)
            month = np.random.randint(1, 13)
            day = np.random.randint(1, 28)
            last_booked = f"2025-{month:02d}-{day:02d}"
            
            accruals.append({
                'company_id': company['id'],
                'company_name': company['name'],
                'accrual_type': accrual_name,
                'gl_account': gl_acc,
                'frequency': freq,
                'amount': amt,
                'last_booked_date': last_booked
            })
    
    return pd.DataFrame(accruals)


def generate_revenue_contracts(company, num_contracts=25):
    """Generate ASC 606 revenue contracts with intentional recognition issues"""
    contracts = []

    if company['industry'] == 'SaaS':
        obligations = [('Software subscription', 'ratable', 0.85),
                       ('Implementation services', 'milestone', 0.15)]
    elif company['industry'] in ('Professional Services', 'Healthcare'):
        obligations = [('Service retainer', 'ratable', 0.60),
                       ('Project delivery', 'milestone', 0.40)]
    else:
        obligations = [('Product supply agreement', 'point_in_time', 0.80),
                       ('Installation & support', 'ratable', 0.20)]

    avg_value = company['revenue_annual'] * 0.02

    for i in range(num_contracts):
        start = datetime(2024, 1, 1) + timedelta(days=int(np.random.randint(0, 730)))
        term_years = int(np.random.choice([1, 2, 3]))
        end = start + timedelta(days=365 * term_years)
        total_value = round(avg_value * np.random.uniform(0.3, 2.5), 2)

        # Planted issue 1: ~15% of contracts have performance obligations that
        # don't sum to total_contract_value, so the Rev Rec agent has an
        # allocation error to catch.
        skew = np.random.uniform(0.80, 0.92) if np.random.random() < 0.15 else 1.0

        pos = []
        for desc, method, share in obligations:
            po = {
                'description': desc,
                'revenue_recognition': method,
                'value': round(total_value * share * skew, 2)
            }
            if method == 'milestone':
                # Planted issue 2: some finished contracts still show partial
                # completion, i.e. revenue recognised late.
                if end < datetime(2026, 1, 31) and np.random.random() < 0.3:
                    po['completion_percentage'] = int(np.random.randint(60, 95))
                else:
                    po['completion_percentage'] = int(np.random.choice([25, 50, 75, 100]))
            pos.append(po)

        contracts.append({
            'contract_id': f"CNTR-{start.year}-{company['id'][:4].upper()}-{i+1:03d}",
            'customer': fake.company(),
            'company': company['id'],
            'start_date': start.strftime('%Y-%m-%d'),
            'end_date': end.strftime('%Y-%m-%d'),
            'total_contract_value': total_value,
            'billing_schedule': str(np.random.choice(['monthly', 'quarterly', 'annual'])),
            'performance_obligations': pos
        })

    return contracts


def main():
    parser = argparse.ArgumentParser(description='Generate sample data for Assignment 1')
    parser.add_argument('--output', type=str, default='./assignment1_data',
                       help='Output directory for generated data')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("🚀 Generating Assignment 1 Sample Data...")
    print(f"📂 Output directory: {output_dir}")
    
    # Generate data for 3 months
    periods = [
        datetime(2025, 11, 1),
        datetime(2025, 12, 1),
        datetime(2026, 1, 1),
    ]
    
    # Create subdirectories
    (output_dir / 'trial_balances').mkdir(exist_ok=True)
    (output_dir / 'budgets').mkdir(exist_ok=True)
    (output_dir / 'prior_year').mkdir(exist_ok=True)
    (output_dir / 'intercompany').mkdir(exist_ok=True)
    (output_dir / 'bank_statements').mkdir(exist_ok=True)
    (output_dir / 'accrual_schedules').mkdir(exist_ok=True)
    (output_dir / 'contracts').mkdir(exist_ok=True)
    
    # Generate trial balances
    print("\n📊 Generating trial balances...")
    for period in periods:
        for company in COMPANIES:
            tb = generate_trial_balance(company, period)
            filename = f"{company['id']}_{period.strftime('%Y_%m')}.csv"
            tb.to_csv(output_dir / 'trial_balances' / filename, index=False)
            print(f"  ✓ {company['name']} - {period.strftime('%Y-%m')}")
    
    # Generate budgets
    print("\n💰 Generating budgets...")
    budgets = generate_budgets(COMPANIES, year=2026)
    budgets.to_csv(output_dir / 'budgets' / 'budgets_2026.csv', index=False)
    print(f"  ✓ Budget file created ({len(budgets)} records)")
    
    # Generate prior year data
    print("\n📅 Generating prior year comparatives...")
    prior_year_periods = [
        datetime(2024, 11, 1),
        datetime(2024, 12, 1),
        datetime(2025, 1, 1),
    ]
    for period in prior_year_periods:
        for company in COMPANIES:
            tb = generate_trial_balance(company, period)
            filename = f"{company['id']}_{period.strftime('%Y_%m')}.csv"
            tb.to_csv(output_dir / 'prior_year' / filename, index=False)
    print(f"  ✓ Prior year data generated")
    
    # Generate intercompany transactions
    print("\n🔄 Generating intercompany transactions...")
    for period in periods:
        ic_trans = generate_intercompany_transactions(COMPANIES, period)
        filename = f"intercompany_{period.strftime('%Y_%m')}.csv"
        ic_trans.to_csv(output_dir / 'intercompany' / filename, index=False)
        print(f"  ✓ {period.strftime('%Y-%m')} - {len(ic_trans)} transactions")
    
    # Generate bank statements
    print("\n🏦 Generating bank statements...")
    for period in periods:
        for company in COMPANIES:
            bank_stmt = generate_bank_statements(company, period)
            filename = f"{company['id']}_{period.strftime('%Y_%m')}.csv"
            bank_stmt.to_csv(output_dir / 'bank_statements' / filename, index=False)
    print(f"  ✓ Bank statements generated")
    
    # Generate accrual schedules
    print("\n📋 Generating accrual schedules...")
    accruals = generate_accrual_schedules(COMPANIES)
    accruals.to_csv(output_dir / 'accrual_schedules' / 'accrual_schedules.csv', index=False)
    print(f"  ✓ Accrual schedules created ({len(accruals)} records)")
    
    # Generate revenue contracts
    print("\n📜 Generating revenue contracts...")
    total_contracts = 0
    for company in COMPANIES:
        contracts = generate_revenue_contracts(company)
        with open(output_dir / 'contracts' / f"{company['id']}_contracts.json", 'w') as f:
            json.dump(contracts, f, indent=2)
        total_contracts += len(contracts)
    print(f"  ✓ Revenue contracts created ({total_contracts} contracts)")

    # Generate company metadata
    print("\n🏢 Generating company metadata...")
    metadata = pd.DataFrame(COMPANIES)
    metadata.to_json(output_dir / 'company_metadata.json', orient='records', indent=2)
    print(f"  ✓ Metadata file created")
    
    # Create README
    print("\n📝 Creating README...")
    readme_content = f"""# Assignment 1: Month-End Close Sample Data

## Dataset Overview
This dataset contains realistic financial data for 8 portfolio companies spanning 3 months (Nov 2025 - Jan 2026).

## Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Directory Structure
```
{args.output}/
├── trial_balances/       # Monthly trial balances for each company
├── budgets/             # 2026 annual budgets
├── prior_year/          # Prior year comparatives (2024-2025)
├── intercompany/        # Intercompany transactions requiring elimination
├── bank_statements/     # Mock bank statements
├── accrual_schedules/   # Standard accrual templates
├── contracts/           # ASC 606 revenue contracts (for the Rev Rec agent)
└── company_metadata.json # Company profiles
```

## Portfolio Companies
{chr(10).join([f"- {c['name']}: ${c['revenue_annual']:,.0f} revenue, {c['industry']}" for c in COMPANIES])}

## Data Validation
Total trial balances generated: {len(COMPANIES) * len(periods)}
Total budget records: {len(budgets)}
Total intercompany transactions: {len(periods) * 1000}
Total accrual schedules: {len(accruals)}

## Known Issues (Intentional for Testing)
1. Some expenses may be miscategorized (e.g., marketing in COGS)
2. Missing accruals in December (bonus accrual not booked)
3. Some intercompany transactions missing elimination entries
4. Timing differences in prepaid expenses

## Loading Instructions
```python
import pandas as pd

# Load trial balance
tb = pd.read_csv('trial_balances/techforge_saas_2026_01.csv')

# Validate it balances
assert abs(tb['debit'].sum() - tb['credit'].sum()) < 0.01

# Load all budgets
budgets = pd.read_csv('budgets/budgets_2026.csv')
```

## Questions?
See the main assignment document, or tag @darsh-vaishnani / @sreeragee on GitHub
"""
    
    with open(output_dir / 'README.md', 'w') as f:
        f.write(readme_content)
    
    print("\n✅ Data generation complete!")
    print(f"\n📦 Total files created: {sum([len(list(p.rglob('*.csv'))) for p in output_dir.iterdir() if p.is_dir()])}")
    print(f"💾 Total size: ~{sum([f.stat().st_size for f in output_dir.rglob('*') if f.is_file()]) / 1024 / 1024:.1f} MB")
    print(f"\n🎯 Ready to use! Start your agents with: {output_dir}")


if __name__ == '__main__':
    main()
