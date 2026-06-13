from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

from app.services.mock_core import mock_core


@dataclass(frozen=True)
class BankingInsight:
    insight_id: str
    category: str
    severity: str
    title: str
    message: str
    recommended_action: str | None = None
    evidence: dict[str, Any] | None = None


class ProactiveInsightEngine:
    def generate(self, user_profile: dict[str, Any]) -> list[dict[str, Any]]:
        customer_id = str(user_profile.get("customer_id", ""))
        if not customer_id:
            return []
        summary = mock_core.account_summary(customer_id)
        transactions = mock_core.search_transactions(customer_id)
        loan_status = mock_core.loan_status(customer_id)
        insights: list[BankingInsight] = []

        for account in summary.get("accounts", []):
            available = float(account.get("available_balance", 0))
            hold = float(account.get("hold_amount", 0))
            if hold > 0:
                insights.append(
                    BankingInsight(
                        "ins-hold-001",
                        "balance",
                        "medium",
                        "Balance hold detected",
                        f"{hold:,.2f} {account.get('currency')} is currently on hold in {account.get('product_name')}.",
                        "VIEW_ACCOUNT_HOLD_DETAILS",
                        {"account_status": account.get("status"), "hold_amount": hold},
                    )
                )
            if available < 5000:
                insights.append(
                    BankingInsight(
                        "ins-low-balance-001",
                        "balance",
                        "high",
                        "Low available balance",
                        f"Available balance is {available:,.2f} {account.get('currency')}. Consider reviewing upcoming payments.",
                        "VIEW_TRANSACTIONS",
                        {"available_balance": available},
                    )
                )

        atm_fees = [txn for txn in transactions.get("transactions", []) if txn.get("fee")]
        if atm_fees:
            insights.append(
                BankingInsight(
                    "ins-atm-fee-001",
                    "fee_avoidance",
                    "medium",
                    "ATM fee detected",
                    "A recent foreign-bank ATM withdrawal included a fee. Use in-network ATMs to avoid repeated fees.",
                    "FIND_NEARBY_BRANCH_OR_ATM",
                    {"fee_count": len(atm_fees), "latest_fee": atm_fees[0].get("fee")},
                )
            )

        for loan in loan_status.get("loans", []):
            insights.append(
                BankingInsight(
                    "ins-loan-due-001",
                    "loan",
                    "medium",
                    "Upcoming loan payment",
                    f"Your next {loan.get('product_name')} payment is due on {loan.get('next_due_date')} for {loan.get('next_due_amount'):,.2f} {loan.get('currency', 'VND')}.",
                    "VIEW_LOAN_STATUS",
                    {"next_due_date": loan.get("next_due_date"), "dpd": loan.get("dpd")},
                )
            )
        return [asdict(insight) for insight in insights]


insight_engine = ProactiveInsightEngine()
