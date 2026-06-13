from __future__ import annotations

import math
from typing import Any

from app.core.config import get_settings
from app.services.access_control import get_role, has_consent, is_allowed
from app.services.audit import audit_logger
from app.services.mock_core import mock_core
from app.services.privacy import redact_text
from app.services.vector_store import BankingVectorStore


FEES: dict[str, dict[str, dict[str, Any]]] = {
    "savings_account": {
        "atm_withdrawal_foreign_bank": {"amount": 3300, "currency": "VND", "waiver": "First 3 NAPAS ATM withdrawals per month"},
        "debit_card_replacement": {"amount": 55000, "currency": "VND", "waiver": "None"},
        "monthly_maintenance": {"amount": 0, "currency": "VND", "waiver": "E-statement enabled and active account"},
    },
    "credit_card": {
        "late_payment": {"amount": "4% of overdue minimum payment", "currency": "VND", "waiver": "Minimum 100,000 VND"},
        "cash_advance": {"amount": "4%", "currency": "VND", "waiver": "Minimum 100,000 VND"},
        "annual_fee": {"amount": 299000, "currency": "VND", "waiver": "Spend 60,000,000 VND/year"},
    },
    "home_loan": {
        "appraisal": {"amount": 1500000, "currency": "VND", "waiver": "Promotional campaigns only"},
        "early_repayment": {"amount": "1%-3% of prepaid principal", "currency": "VND", "waiver": "May be waived after year 3"},
    },
}

ELIGIBILITY_RULES: dict[str, dict[str, Any]] = {
    "credit_card": {"min_age": 20, "min_income": 8000000, "required_employment_months": 6},
    "personal_loan": {"min_age": 20, "min_income": 7000000, "required_employment_months": 6},
    "home_loan": {"min_age": 20, "min_income": 15000000, "required_employment_months": 12},
}


def retrieve_banking_docs(query: str, store: BankingVectorStore | None = None, top_k: int | None = None) -> dict[str, Any]:
    settings = get_settings()
    vector_store = store or BankingVectorStore(settings)
    results = vector_store.search(query, top_k=top_k or settings.top_k)
    return {"results": results, "count": len(results)}


def loan_calculator(amount: float, rate: float, months: int) -> dict[str, Any]:
    if amount <= 0 or rate < 0 or months <= 0:
        raise ValueError("amount and months must be positive; rate cannot be negative")
    monthly_rate = rate / 100 / 12
    if monthly_rate == 0:
        payment = amount / months
    else:
        payment = amount * monthly_rate * math.pow(1 + monthly_rate, months)
        payment /= math.pow(1 + monthly_rate, months) - 1
    total_payment = payment * months
    return {
        "monthly_payment": round(payment, 2),
        "total_payment": round(total_payment, 2),
        "total_interest": round(total_payment - amount, 2),
        "currency": "VND",
        "assumption": "Fixed-rate amortization estimate before bank approval and fees",
    }


def fee_lookup(product_type: str, service_name: str) -> dict[str, Any]:
    product = product_type.lower().strip().replace(" ", "_")
    service = service_name.lower().strip().replace(" ", "_")
    fee = FEES.get(product, {}).get(service)
    if not fee:
        return {
            "found": False,
            "message": "Fee not found in structured fee table. Use document retrieval or ask support.",
        }
    return {"found": True, "product_type": product, "service_name": service, **fee}


def eligibility_checker(user_profile: dict[str, Any], product: str) -> dict[str, Any]:
    normalized_product = product.lower().strip().replace(" ", "_")
    rules = ELIGIBILITY_RULES.get(normalized_product)
    if not rules:
        return {"eligible": None, "reason": "No structured eligibility rules found for this product."}

    missing = [field for field in ["age", "monthly_income", "employment_months"] if field not in user_profile]
    if missing:
        return {
            "eligible": None,
            "missing_fields": missing,
            "rules": rules,
            "reason": "More user information is required before eligibility can be checked.",
        }

    checks = {
        "age": int(user_profile["age"]) >= rules["min_age"],
        "monthly_income": float(user_profile["monthly_income"]) >= rules["min_income"],
        "employment_months": int(user_profile["employment_months"]) >= rules["required_employment_months"],
    }
    return {
        "eligible": all(checks.values()),
        "checks": checks,
        "rules": rules,
        "human_review_required": True,
        "disclaimer": "This is a preliminary rule check, not an approval decision.",
    }


def create_support_ticket(issue_type: str, summary: str, human_review_confirmed: bool = False) -> dict[str, Any]:
    if not human_review_confirmed:
        return {
            "created": False,
            "human_review_required": True,
            "message": "Please confirm consent before creating a support ticket.",
        }
    ticket_seed = abs(hash((issue_type, summary))) % 100000
    return {"created": True, "ticket_id": f"SUP-{ticket_seed:05d}", "status": "pending_human_review"}


def customer_account_summary(user_profile: dict[str, Any], purpose: str = "customer_service") -> dict[str, Any]:
    role = get_role(user_profile)
    customer_id = str(user_profile.get("customer_id", ""))
    consent = has_consent(user_profile)
    if not is_allowed(user_profile, "account_summary"):
        audit_logger.record(
            actor_id=customer_id or "anonymous",
            role=role,
            action="customer_account_summary",
            purpose=purpose,
            consent_confirmed=consent,
            status="denied",
        )
        return {"allowed": False, "reason": "Authenticated customer role is required."}
    if not consent:
        audit_logger.record(
            actor_id=customer_id,
            role=role,
            action="customer_account_summary",
            purpose=purpose,
            consent_confirmed=False,
            status="consent_required",
        )
        return {"allowed": False, "human_review_required": True, "reason": "Customer consent is required."}
    result = mock_core.account_summary(customer_id)
    audit_logger.record(
        actor_id=customer_id,
        role=role,
        action="customer_account_summary",
        purpose=purpose,
        consent_confirmed=True,
        status="success" if result.get("found") else "not_found",
        metadata={"account_count": len(result.get("accounts", []))},
    )
    return {"allowed": True, **result}


def transaction_search(
    user_profile: dict[str, Any],
    keyword: str | None = None,
    limit: int = 10,
    purpose: str = "transaction_inquiry",
) -> dict[str, Any]:
    role = get_role(user_profile)
    customer_id = str(user_profile.get("customer_id", ""))
    consent = has_consent(user_profile)
    if not is_allowed(user_profile, "transaction_search"):
        return {"allowed": False, "reason": "Authenticated customer role is required."}
    if not consent:
        return {"allowed": False, "human_review_required": True, "reason": "Customer consent is required."}
    redacted_keyword = redact_text(keyword or "")
    result = mock_core.search_transactions(customer_id, redacted_keyword.redacted or None, limit=limit)
    audit_logger.record(
        actor_id=customer_id,
        role=role,
        action="transaction_search",
        purpose=purpose,
        consent_confirmed=True,
        pii_types=redacted_keyword.pii_types,
        status="success" if result.get("found") else "not_found",
        metadata={"count": result.get("count", 0)},
    )
    return {"allowed": True, **result}


def loan_status_lookup(user_profile: dict[str, Any], purpose: str = "loan_status_inquiry") -> dict[str, Any]:
    role = get_role(user_profile)
    customer_id = str(user_profile.get("customer_id", ""))
    consent = has_consent(user_profile)
    if not is_allowed(user_profile, "loan_status"):
        return {"allowed": False, "reason": "Authenticated customer role is required."}
    if not consent:
        return {"allowed": False, "human_review_required": True, "reason": "Customer consent is required."}
    result = mock_core.loan_status(customer_id)
    audit_logger.record(
        actor_id=customer_id,
        role=role,
        action="loan_status_lookup",
        purpose=purpose,
        consent_confirmed=True,
        status="success" if result.get("found") else "not_found",
        metadata={"loan_count": result.get("count", 0)},
    )
    return {"allowed": True, **result}


def branch_finder(province: str | None = None, service: str | None = None) -> dict[str, Any]:
    return mock_core.find_branches(province=province, service=service)
