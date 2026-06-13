from __future__ import annotations

from datetime import date, datetime
from typing import Any


def assess_account(account: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for field in ["account_id", "product_name", "currency", "status"]:
        if account.get(field) in {None, ""}:
            flags.append(f"missing_{field}")
    if account.get("available_balance") is None:
        flags.append("missing_available_balance")
    if account.get("ledger_balance") is None:
        flags.append("missing_ledger_balance")
    if account.get("available_balance") is not None and float(account["available_balance"]) < 0:
        flags.append("negative_available_balance")
    if account.get("status") in {"frozen", "dormant", "closed"}:
        flags.append(f"account_{account['status']}")
    if account.get("hold_amount") is None:
        flags.append("missing_hold_amount")
    return flags


def assess_transaction(txn: dict[str, Any], seen_ids: set[str]) -> list[str]:
    flags: list[str] = []
    txn_id = txn.get("transaction_id")
    if not txn_id:
        flags.append("missing_transaction_id")
    elif txn_id in seen_ids:
        flags.append("duplicate_transaction_id")
    else:
        seen_ids.add(txn_id)
    for field in ["posted_date", "amount", "direction", "channel"]:
        if txn.get(field) in {None, ""}:
            flags.append(f"missing_{field}")
    if not txn.get("merchant_category"):
        flags.append("missing_merchant_category")
    if txn.get("status") == "pending":
        flags.append("pending_authorization")
    if txn.get("reversal_of"):
        flags.append("reversal_transaction")
    if txn.get("fee") and not txn.get("linked_transaction_id"):
        flags.append("standalone_fee_posting")
    return flags


def assess_loan(loan: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    for field in ["loan_id", "product_name", "outstanding_principal", "status"]:
        if loan.get(field) in {None, ""}:
            flags.append(f"missing_{field}")
    if loan.get("next_due_amount") is None:
        flags.append("missing_next_due_amount")
    if loan.get("dpd", 0) and int(loan["dpd"]) > 0:
        flags.append("loan_delinquent")
    if loan.get("status") in {"restructured", "hardship", "collections"}:
        flags.append(f"loan_{loan['status']}")
    if loan.get("rate_type") == "floating" and loan.get("rate_reset_date") is None:
        flags.append("floating_rate_missing_reset_date")
    return flags


def assess_branch(branch: dict[str, Any]) -> list[str]:
    flags: list[str] = []
    if branch.get("temporarily_closed"):
        flags.append("branch_temporarily_closed")
    if branch.get("data_as_of") and _days_old(branch["data_as_of"]) > 90:
        flags.append("stale_branch_metadata")
    if not branch.get("services"):
        flags.append("missing_services")
    if branch.get("appointment_required"):
        flags.append("appointment_required")
    return flags


def annotate_accounts(accounts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**account, "data_quality_flags": assess_account(account)} for account in accounts]


def annotate_transactions(transactions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen_ids: set[str] = set()
    return [{**txn, "data_quality_flags": assess_transaction(txn, seen_ids)} for txn in transactions]


def annotate_loans(loans: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**loan, "data_quality_flags": assess_loan(loan)} for loan in loans]


def annotate_branches(branches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [{**branch, "data_quality_flags": assess_branch(branch)} for branch in branches]


def collect_flags(items: list[dict[str, Any]]) -> list[str]:
    flags: list[str] = []
    for item in items:
        flags.extend(item.get("data_quality_flags", []))
    return sorted(set(flags))


def _days_old(value: str) -> int:
    try:
        dt = datetime.fromisoformat(value).date()
    except ValueError:
        return 9999
    return (date.today() - dt).days

