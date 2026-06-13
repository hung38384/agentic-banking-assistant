from __future__ import annotations

from datetime import date
from typing import Any

from app.services.data_quality import (
    annotate_accounts,
    annotate_branches,
    annotate_loans,
    annotate_transactions,
    collect_flags,
)
from app.services.privacy import mask_identifier


CUSTOMERS: dict[str, dict[str, Any]] = {
    "CUST1001": {
        "customer_id": "CUST1001",
        "segment": "Mass Affluent",
        "kyc_level": "full_kyc",
        "risk_level": "medium",
        "preferred_language": "Vietnamese",
        "relationship_since": "2021-04-18",
        "accounts": [
            {
                "account_id": "SA-1001-001",
                "account_no": "123-4-56789-0",
                "product_name": "MB Everyday Savings",
                "currency": "VND",
                "ledger_balance": 128450750.0,
                "available_balance": 126950750.0,
                "hold_amount": 1500000.0,
                "status": "active",
                "branch_code": "001",
            },
            {
                "account_id": "CA-1001-002",
                "account_no": "456-1-22222-9",
                "product_name": "MB Payroll Current Account",
                "currency": "VND",
                "ledger_balance": 32000000.0,
                "available_balance": 32000000.0,
                "hold_amount": 0.0,
                "status": "active",
                "branch_code": "014",
            },
        ],
        "credit_cards": [
            {
                "card_id": "CARD-9001",
                "masked_card_no": "**** **** **** 4242",
                "product_name": "MB Visa Classic Credit Card",
                "credit_limit": 85000000.0,
                "outstanding_balance": 18420250.0,
                "minimum_due": 1850000.0,
                "payment_due_date": "2026-06-25",
                "status": "active",
                "reward_points": 12450,
            }
        ],
        "loans": [
            {
                "loan_id": "HL-7001",
                "product_name": "MB Home Loan",
                "original_amount": 3200000000.0,
                "outstanding_principal": 2890500000.0,
                "annual_rate": 5.75,
                "remaining_months": 276,
                "next_due_date": "2026-07-02",
                "next_due_amount": 20540000.0,
                "dpd": 0,
                "status": "normal",
            }
        ],
    }
    ,
    "CUST2002": {
        "customer_id": "CUST2002",
        "segment": "Student",
        "kyc_level": "partial_kyc",
        "risk_level": "high",
        "preferred_language": "Vietnamese",
        "relationship_since": "2026-01-12",
        "profile_flags": ["missing_income_document", "phone_unverified"],
        "accounts": [
            {
                "account_id": "SA-2002-001",
                "account_no": "789-0-11111-2",
                "product_name": "MB Student Savings",
                "currency": "VND",
                "ledger_balance": 2400000.0,
                "available_balance": -150000.0,
                "hold_amount": None,
                "status": "active",
                "branch_code": "053",
            },
            {
                "account_id": "SA-2002-OLD",
                "account_no": "789-0-99999-8",
                "product_name": None,
                "currency": "VND",
                "ledger_balance": 0.0,
                "available_balance": 0.0,
                "hold_amount": 0.0,
                "status": "dormant",
                "branch_code": "999",
            },
        ],
        "credit_cards": [
            {
                "card_id": "CARD-2002",
                "masked_card_no": "**** **** **** 9988",
                "product_name": "MB Starter Credit Card",
                "credit_limit": 20000000.0,
                "outstanding_balance": 20500500.0,
                "minimum_due": None,
                "payment_due_date": None,
                "status": "blocked",
                "reward_points": None,
                "data_quality_flags": ["overlimit", "missing_minimum_due", "blocked_card"],
            }
        ],
        "loans": [
            {
                "loan_id": "PL-2002",
                "product_name": "MB Personal Loan",
                "original_amount": 80000000.0,
                "outstanding_principal": 63500000.0,
                "annual_rate": 14.5,
                "rate_type": "floating",
                "rate_reset_date": None,
                "remaining_months": 18,
                "next_due_date": "2026-06-15",
                "next_due_amount": None,
                "dpd": 12,
                "status": "restructured",
            }
        ],
    }
}

TRANSACTIONS: dict[str, list[dict[str, Any]]] = {
    "CUST1001": [
        {
            "transaction_id": "TXN-20260607-0001",
            "posted_date": "2026-06-07",
            "account_id": "SA-1001-001",
            "description": "NAPAS transfer received",
            "amount": 12500000.0,
            "currency": "VND",
            "direction": "credit",
            "channel": "mobile_banking",
            "merchant_category": "transfer",
            "balance_after": 128450750.0,
        },
        {
            "transaction_id": "TXN-20260606-0008",
            "posted_date": "2026-06-06",
            "account_id": "SA-1001-001",
            "description": "Card purchase - supermarket",
            "amount": 1840250.0,
            "currency": "VND",
            "direction": "debit",
            "channel": "debit_card",
            "merchant_category": "groceries",
            "balance_after": 115950750.0,
        },
        {
            "transaction_id": "TXN-20260605-0012",
            "posted_date": "2026-06-05",
            "account_id": "SA-1001-001",
            "description": "ATM withdrawal NAPAS bank",
            "amount": 3000000.0,
            "currency": "VND",
            "direction": "debit",
            "channel": "atm",
            "merchant_category": "cash",
            "fee": 3300.0,
            "balance_after": 117791000.0,
        },
    ]
    ,
    "CUST2002": [
        {
            "transaction_id": "TXN-20260604-0002",
            "posted_date": "2026-06-04",
            "account_id": "SA-2002-001",
            "description": "POS AUTH HOLD convenience store HCM",
            "amount": 320000.0,
            "currency": "VND",
            "direction": "debit",
            "channel": "debit_card",
            "merchant_category": None,
            "status": "pending",
            "balance_after": None,
        },
        {
            "transaction_id": "TXN-20260604-0002",
            "posted_date": "2026-06-04",
            "account_id": "SA-2002-001",
            "description": "POS AUTH HOLD convenience store HCM DUPLICATE",
            "amount": 320000.0,
            "currency": "VND",
            "direction": "debit",
            "channel": "debit_card",
            "merchant_category": None,
            "status": "pending",
            "balance_after": None,
        },
        {
            "transaction_id": "TXN-20260603-0044",
            "posted_date": "2026-06-03",
            "account_id": "SA-2002-001",
            "description": "REVERSAL ATM FEE",
            "amount": 3300.0,
            "currency": "VND",
            "direction": "credit",
            "channel": "core_adjustment",
            "merchant_category": "",
            "status": "posted",
            "reversal_of": "TXN-20260603-0043",
            "balance_after": 2400000.0,
        },
        {
            "transaction_id": "TXN-20260602-FEE",
            "posted_date": "2026-06-02",
            "account_id": "SA-2002-001",
            "description": "ATM SERVICE FEE",
            "amount": 3300.0,
            "currency": "VND",
            "direction": "debit",
            "channel": "fee_engine",
            "merchant_category": None,
            "status": "posted",
            "fee": 3300.0,
            "balance_after": 2385000.0,
        },
    ]
}

BRANCHES: list[dict[str, Any]] = [
    {
        "branch_code": "001",
        "name": "MB Hội Sở Hà Nội",
        "province": "Hanoi",
        "district": "Cau Giay",
        "services": ["account_opening", "loan_consultation", "wealth_advisory", "cash_service"],
        "hours": "Mon-Fri 08:30-16:30",
        "queue_supported": True,
    },
    {
        "branch_code": "014",
        "name": "MB Nguyễn Huệ Branch",
        "province": "Ho Chi Minh City",
        "district": "District 1",
        "services": ["account_opening", "credit_card", "cash_service"],
        "hours": "Mon-Sun 10:30-19:00",
        "queue_supported": True,
    },
    {
        "branch_code": "053",
        "name": "MB Đà Nẵng Central Branch",
        "province": "Da Nang",
        "district": "Hai Chau",
        "services": ["account_opening", "loan_consultation", "cash_service"],
        "hours": "Mon-Sun 10:00-18:00",
        "queue_supported": False,
    },
    {
        "branch_code": "999",
        "name": "MB Old Quarter Temporary Service Point",
        "province": "Hanoi",
        "district": "Hoan Kiem",
        "services": [],
        "hours": None,
        "queue_supported": False,
        "temporarily_closed": True,
        "appointment_required": True,
        "data_as_of": "2025-01-15",
    },
]


class MockCoreBankingAdapter:
    """MBBank-style adapter contract with detailed synthetic data.

    Replace this class with real authenticated core banking/API gateway calls
    when integrating with a bank environment.
    """

    def get_customer_profile(self, customer_id: str) -> dict[str, Any] | None:
        customer = CUSTOMERS.get(customer_id)
        if not customer:
            return None
        return {
            **customer,
            "accounts": [
                {**account, "account_no": mask_identifier(account["account_no"])}
                for account in customer["accounts"]
            ],
        }

    def account_summary(self, customer_id: str) -> dict[str, Any]:
        customer = self.get_customer_profile(customer_id)
        if not customer:
            return {"found": False, "message": "Customer not found in mock core banking."}
        accounts = annotate_accounts(customer["accounts"])
        return {
            "found": True,
            "as_of": date.today().isoformat(),
            "customer_id": customer_id,
            "segment": customer["segment"],
            "kyc_level": customer["kyc_level"],
            "profile_flags": customer.get("profile_flags", []),
            "accounts": accounts,
            "credit_cards": customer["credit_cards"],
            "data_quality_flags": sorted(set(customer.get("profile_flags", []) + collect_flags(accounts))),
        }

    def search_transactions(self, customer_id: str, keyword: str | None = None, limit: int = 10) -> dict[str, Any]:
        transactions = TRANSACTIONS.get(customer_id, [])
        if keyword:
            lowered = keyword.lower()
            transactions = [
                txn for txn in transactions if lowered in txn["description"].lower()
                or lowered in txn["channel"].lower()
                or lowered in txn["merchant_category"].lower()
            ]
        annotated = annotate_transactions(transactions)
        return {
            "found": bool(annotated),
            "transactions": annotated[:limit],
            "count": len(annotated),
            "data_quality_flags": collect_flags(annotated),
        }

    def loan_status(self, customer_id: str) -> dict[str, Any]:
        customer = CUSTOMERS.get(customer_id)
        if not customer:
            return {"found": False, "message": "Customer not found in mock core banking."}
        loans = annotate_loans(customer["loans"])
        return {"found": True, "loans": loans, "count": len(loans), "data_quality_flags": collect_flags(loans)}

    def find_branches(self, province: str | None = None, service: str | None = None) -> dict[str, Any]:
        branches = BRANCHES
        if province:
            branches = [branch for branch in branches if branch["province"].lower() == province.lower()]
        if service:
            normalized = service.lower().replace(" ", "_")
            branches = [branch for branch in branches if normalized in branch["services"]]
        annotated = annotate_branches(branches)
        return {"found": bool(annotated), "branches": annotated, "count": len(annotated), "data_quality_flags": collect_flags(annotated)}


mock_core = MockCoreBankingAdapter()
