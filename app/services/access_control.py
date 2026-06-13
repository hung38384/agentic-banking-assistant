from __future__ import annotations

from typing import Any


ROLE_PERMISSIONS: dict[str, set[str]] = {
    "anonymous": {"product_qa", "fee_lookup", "loan_calculation", "branch_finder"},
    "customer": {
        "product_qa",
        "fee_lookup",
        "loan_calculation",
        "branch_finder",
        "eligibility",
        "account_summary",
        "transaction_search",
        "loan_status",
        "support",
    },
    "staff": {
        "product_qa",
        "fee_lookup",
        "loan_calculation",
        "branch_finder",
        "eligibility",
        "account_summary",
        "transaction_search",
        "loan_status",
        "support",
    },
}

SENSITIVE_INTENTS = {"account_summary", "transaction_search", "loan_status", "support"}


def get_role(user_profile: dict[str, Any]) -> str:
    role = str(user_profile.get("role", "anonymous")).lower()
    return role if role in ROLE_PERMISSIONS else "anonymous"


def is_allowed(user_profile: dict[str, Any], intent: str) -> bool:
    return intent in ROLE_PERMISSIONS[get_role(user_profile)]


def has_consent(user_profile: dict[str, Any]) -> bool:
    return bool(user_profile.get("consent_confirmed") or user_profile.get("consent") is True)

