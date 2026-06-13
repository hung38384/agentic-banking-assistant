from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PolicyDecision:
    allowed: bool
    flags: list[str] = field(default_factory=list)
    severity: str = "low"
    reason: str = ""
    requires_human_approval: bool = False


PROMPT_INJECTION_PATTERNS = [
    r"ignore (all )?(previous|prior|system|developer|policies|policy) instructions?",
    r"ignore (all )?(policies|policy)",
    r"reveal (the )?(system|developer) prompt",
    r"you are now",
    r"act as (an )?admin",
    r"bypass (policy|guardrails|safety)",
    r"call .*tool .*without",
    r"exfiltrate|leak|dump",
    r"reveal full (account|card|customer)",
    r"full account number",
    r"bo qua (tat ca )?(huong dan|chi dan|chinh sach|guardrail)",
    r"tiet lo (system prompt|prompt he thong|chi dan he thong|developer prompt)",
    r"vuot qua (chinh sach|guardrail|bao mat)",
    r"goi .*tool .*khong can",
]

SQL_INJECTION_PATTERNS = [
    r"('|--|;)\s*(or|and)\s*'?1'?\s*=\s*'?1'?",
    r"union\s+select",
    r"drop\s+table",
    r"insert\s+into",
    r"delete\s+from",
    r"xp_cmdshell",
]

HIGH_RISK_INTENTS = {"account_summary", "transaction_search", "loan_status", "support"}
MUTATING_INTENTS = {"support"}


class PolicyEngine:
    def __init__(self, policy_path: Path | None = None):
        self.policy_path = policy_path

    def evaluate_input(self, text: str) -> PolicyDecision:
        lowered = text.lower()
        normalized = _normalize_vi_text(text)
        flags: list[str] = []
        if any(re.search(pattern, lowered) or re.search(pattern, normalized) for pattern in PROMPT_INJECTION_PATTERNS):
            flags.append("prompt_injection_detected")
        if any(re.search(pattern, lowered) for pattern in SQL_INJECTION_PATTERNS):
            flags.append("sql_injection_detected")
        if flags:
            return PolicyDecision(
                allowed=False,
                flags=flags,
                severity="high",
                reason="Security policy blocked adversarial instructions or injection payloads.",
                requires_human_approval=True,
            )
        return PolicyDecision(allowed=True)

    def evaluate_tool_call(self, intent: str, tool_name: str, user_profile: dict[str, Any]) -> PolicyDecision:
        risk = score_tool_risk(intent, tool_name)
        flags = [f"tool_risk_{risk['level']}"]
        if intent in HIGH_RISK_INTENTS and not user_profile.get("consent_confirmed"):
            return PolicyDecision(
                allowed=False,
                flags=[*flags, "consent_required"],
                severity="high",
                reason="Customer consent is required before invoking personal-data tools.",
                requires_human_approval=True,
            )
        if intent in MUTATING_INTENTS:
            return PolicyDecision(
                allowed=False,
                flags=[*flags, "human_approval_required"],
                severity="high",
                reason="Sensitive side-effect action requires human approval.",
                requires_human_approval=True,
            )
        return PolicyDecision(
            allowed=True,
            flags=flags,
            severity=risk["level"],
            requires_human_approval=risk["level"] == "high",
        )


def score_tool_risk(intent: str, tool_name: str) -> dict[str, Any]:
    if intent in {"support"} or tool_name in {"create_support_ticket"}:
        return {"score": 0.9, "level": "high", "reason": "Potential side-effect or escalation."}
    if intent in {"account_summary", "transaction_search", "loan_status"}:
        return {"score": 0.75, "level": "high", "reason": "Personal banking data access."}
    if intent in {"eligibility"}:
        return {"score": 0.55, "level": "medium", "reason": "Decision-support workflow requires caveats."}
    return {"score": 0.2, "level": "low", "reason": "Read-only public/product workflow."}


policy_engine = PolicyEngine()


def _normalize_vi_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")
