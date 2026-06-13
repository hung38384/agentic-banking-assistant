from __future__ import annotations

from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class AuditEvent:
    event_id: str
    timestamp: str
    actor_id: str
    role: str
    action: str
    purpose: str
    consent_confirmed: bool
    pii_types: list[str]
    status: str
    metadata: dict[str, Any]


class AuditLogger:
    def __init__(self) -> None:
        self._events: list[AuditEvent] = []

    def record(
        self,
        *,
        actor_id: str,
        role: str,
        action: str,
        purpose: str,
        consent_confirmed: bool,
        pii_types: list[str] | None = None,
        status: str = "success",
        metadata: dict[str, Any] | None = None,
    ) -> AuditEvent:
        event = AuditEvent(
            event_id=f"audit-{uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc).isoformat(),
            actor_id=actor_id,
            role=role,
            action=action,
            purpose=purpose,
            consent_confirmed=consent_confirmed,
            pii_types=pii_types or [],
            status=status,
            metadata=metadata or {},
        )
        self._events.append(event)
        return event

    def recent(self, limit: int = 50) -> list[dict[str, Any]]:
        return [asdict(event) for event in self._events[-limit:]]


audit_logger = AuditLogger()

