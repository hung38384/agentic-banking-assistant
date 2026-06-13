from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class ApprovalRequest:
    approval_id: str
    created_at: str
    status: str
    intent: str
    tool_name: str
    reason: str
    requested_by: str
    payload: dict[str, Any]


class ApprovalQueue:
    def __init__(self) -> None:
        self._requests: dict[str, ApprovalRequest] = {}

    def create(
        self,
        *,
        intent: str,
        tool_name: str,
        reason: str,
        requested_by: str,
        payload: dict[str, Any],
    ) -> ApprovalRequest:
        request = ApprovalRequest(
            approval_id=f"appr-{uuid4().hex[:12]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            status="pending",
            intent=intent,
            tool_name=tool_name,
            reason=reason,
            requested_by=requested_by,
            payload=payload,
        )
        self._requests[request.approval_id] = request
        return request

    def list(self, status: str | None = None) -> list[dict[str, Any]]:
        requests = self._requests.values()
        if status:
            requests = [request for request in requests if request.status == status]
        return [asdict(request) for request in requests]

    def decide(self, approval_id: str, approved: bool) -> dict[str, Any]:
        request = self._requests.get(approval_id)
        if not request:
            return {"found": False, "message": "Approval request not found."}
        request.status = "approved" if approved else "rejected"
        return {"found": True, **asdict(request)}


approval_queue = ApprovalQueue()

