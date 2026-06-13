from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class TokenBudget:
    intent: str
    top_k: int
    max_context_chars: int
    model_tier: str
    reason: str


class TokenBudgetManager:
    def __init__(self, default_top_k: int = 4, default_context_chars: int = 4500):
        self.default_top_k = default_top_k
        self.default_context_chars = default_context_chars

    def plan(self, question: str, intent: str, has_tools: bool) -> TokenBudget:
        complexity = self._complexity(question)
        if intent in {"fee_lookup", "loan_calculation", "branch_finder"} and has_tools:
            return TokenBudget(intent, top_k=2, max_context_chars=1800, model_tier="fast", reason="Tool-first simple workflow.")
        if intent in {"account_summary", "transaction_search", "loan_status"}:
            return TokenBudget(intent, top_k=3, max_context_chars=2600, model_tier="fast", reason="Personal-data tool carries primary facts.")
        if complexity >= 2:
            return TokenBudget(intent, top_k=6, max_context_chars=5200, model_tier="strong", reason="Complex or multi-part query.")
        return TokenBudget(intent, top_k=self.default_top_k, max_context_chars=self.default_context_chars, model_tier="balanced", reason="Standard grounded QA.")

    def compact_tool_outputs(self, tool_outputs: list[dict[str, Any]]) -> list[dict[str, Any]]:
        compacted: list[dict[str, Any]] = []
        for call in tool_outputs:
            output = call.get("output", {})
            if call.get("name") == "customer_account_summary":
                output = {
                    "found": output.get("found"),
                    "account_count": len(output.get("accounts", [])),
                    "accounts": [
                        {
                            "product_name": account.get("product_name"),
                            "available_balance": account.get("available_balance"),
                            "currency": account.get("currency"),
                            "status": account.get("status"),
                        }
                        for account in output.get("accounts", [])[:3]
                    ],
                }
            elif call.get("name") == "transaction_search":
                output = {
                    "found": output.get("found"),
                    "count": output.get("count"),
                    "transactions": [
                        {
                            "posted_date": txn.get("posted_date"),
                            "description": txn.get("description"),
                            "amount": txn.get("amount"),
                            "direction": txn.get("direction"),
                            "channel": txn.get("channel"),
                            "fee": txn.get("fee"),
                        }
                        for txn in output.get("transactions", [])[:5]
                    ],
                }
            compacted.append({**call, "output": output})
        return compacted

    def estimate_tokens(self, text: str | dict[str, Any] | list[Any]) -> int:
        if not isinstance(text, str):
            text = json.dumps(text, ensure_ascii=False, separators=(",", ":"))
        return max(1, len(text) // 4)

    def _complexity(self, question: str) -> int:
        lowered = question.lower()
        multi_part = len(re.findall(r"\b(and|also|compare|then|both|between)\b", lowered))
        financial_terms = sum(term in lowered for term in ["eligibility", "loan", "fee", "transaction", "account", "policy"])
        return multi_part + (1 if financial_terms >= 2 else 0)


token_budget_manager = TokenBudgetManager()

