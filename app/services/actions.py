from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


ACTION_RISK = {
    "VIEW_TRANSACTIONS": "medium",
    "VIEW_LOAN_STATUS": "medium",
    "SCHEDULE_BRANCH_APPOINTMENT": "medium",
    "FIND_NEARBY_BRANCH_OR_ATM": "low",
    "CREATE_FRAUD_TICKET": "high",
    "LOCK_CARD": "high",
    "REPLACE_CARD": "high",
}


@dataclass
class BankingAction:
    action_id: str
    created_at: str
    action_type: str
    status: str
    risk_level: str
    requires_human_approval: bool
    customer_id: str
    payload: dict[str, Any]


class ActionEngine:
    def __init__(self) -> None:
        self._actions: dict[str, BankingAction] = {}

    def recommend(self, insights: list[dict[str, Any]]) -> list[dict[str, Any]]:
        actions = []
        for insight in insights:
            action_type = insight.get("recommended_action")
            if not action_type:
                continue
            risk = ACTION_RISK.get(action_type, "low")
            actions.append(
                {
                    "action_type": action_type,
                    "risk_level": risk,
                    "requires_human_approval": risk == "high",
                    "reason": insight.get("title"),
                }
            )
        return actions

    def create(self, action_type: str, customer_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        risk = ACTION_RISK.get(action_type, "low")
        action = BankingAction(
            action_id=f"act-{uuid4().hex[:12]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            action_type=action_type,
            status="pending_approval" if risk == "high" else "ready",
            risk_level=risk,
            requires_human_approval=risk == "high",
            customer_id=customer_id,
            payload=payload,
        )
        self._actions[action.action_id] = action
        return asdict(action)

    def list(self, customer_id: str | None = None) -> list[dict[str, Any]]:
        actions = self._actions.values()
        if customer_id:
            actions = [action for action in actions if action.customer_id == customer_id]
        return [asdict(action) for action in actions]

    def transition(self, action_id: str, status: str) -> dict[str, Any]:
        action = self._actions.get(action_id)
        if not action:
            return {"found": False, "message": "Action not found."}
        action.status = status
        return {"found": True, **asdict(action)}


action_engine = ActionEngine()

