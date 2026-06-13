from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4


@dataclass
class FeedbackEvent:
    feedback_id: str
    created_at: str
    conversation_id: str
    rating: int
    category: str
    comment: str | None
    metadata: dict[str, Any]


class FeedbackStore:
    def __init__(self) -> None:
        self._events: list[FeedbackEvent] = []

    def add(
        self,
        *,
        conversation_id: str,
        rating: int,
        category: str,
        comment: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        event = FeedbackEvent(
            feedback_id=f"fb-{uuid4().hex[:12]}",
            created_at=datetime.now(timezone.utc).isoformat(),
            conversation_id=conversation_id,
            rating=rating,
            category=category,
            comment=comment,
            metadata=metadata or {},
        )
        self._events.append(event)
        return asdict(event)

    def recent(self, limit: int = 100) -> list[dict[str, Any]]:
        return [asdict(event) for event in self._events[-limit:]]

    def content_gaps(self) -> list[dict[str, Any]]:
        gaps: dict[str, dict[str, Any]] = {}
        for event in self._events:
            if event.rating >= 3 and event.category not in {"missing_content", "wrong_answer", "escalation_needed"}:
                continue
            key = event.category
            gaps.setdefault(key, {"category": key, "count": 0, "examples": []})
            gaps[key]["count"] += 1
            if event.comment and len(gaps[key]["examples"]) < 5:
                gaps[key]["examples"].append(event.comment)
        return sorted(gaps.values(), key=lambda item: item["count"], reverse=True)


feedback_store = FeedbackStore()

