import json
import unicodedata
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

from app.schemas import ChatMessage


class ConversationMemory:
    def __init__(
        self,
        max_turns: int = 8,
        persist_path: Path | None = None,
        persist_requires_consent: bool = True,
    ):
        self.max_turns = max_turns
        self.persist_path = persist_path
        self.persist_requires_consent = persist_requires_consent
        self._messages: dict[str, deque[ChatMessage]] = defaultdict(lambda: deque(maxlen=max_turns))
        self._preferences: dict[str, dict[str, Any]] = defaultdict(dict)
        self._slots: dict[str, dict[str, Any]] = defaultdict(dict)
        self._pending_tasks: dict[str, dict[str, Any] | None] = defaultdict(lambda: None)
        self._consent: dict[str, bool] = defaultdict(bool)
        self._load()

    def add(self, conversation_id: str, role: str, content: str) -> None:
        self._messages[conversation_id].append(ChatMessage(role=role, content=content))
        self._persist()

    def history(self, conversation_id: str) -> list[ChatMessage]:
        return list(self._messages[conversation_id])

    def set_consent(self, conversation_id: str, user_profile: dict[str, Any] | None = None) -> None:
        self._consent[conversation_id] = bool((user_profile or {}).get("consent_confirmed"))
        self._persist()

    def update_preferences(self, conversation_id: str, message: str) -> None:
        lowered = message.lower()
        normalized = _normalize_text(message)
        if "vietnamese" in lowered or "tiếng việt" in lowered or "tieng viet" in lowered:
            self._preferences[conversation_id]["preferred_language"] = "Vietnamese"
        if "english" in lowered:
            self._preferences[conversation_id]["preferred_language"] = "English"
        if "credit card" in lowered:
            self._preferences[conversation_id]["product_interest"] = "credit_card"
        if "home loan" in lowered or "mortgage" in lowered:
            self._preferences[conversation_id]["product_interest"] = "home_loan"
        if "tieng viet" in normalized:
            self._preferences[conversation_id]["preferred_language"] = "Vietnamese"
        if "the tin dung" in normalized:
            self._preferences[conversation_id]["product_interest"] = "credit_card"
        if "vay mua nha" in normalized or "vay nha" in normalized:
            self._preferences[conversation_id]["product_interest"] = "home_loan"
        if "thẻ tín dụng" in lowered or "the tin dung" in lowered:
            self._preferences[conversation_id]["product_interest"] = "credit_card"
        if "vay mua nhà" in lowered or "vay nha" in lowered:
            self._preferences[conversation_id]["product_interest"] = "home_loan"
        self._persist()

    def preferences(self, conversation_id: str) -> dict[str, Any]:
        return dict(self._preferences[conversation_id])

    def slots(self, conversation_id: str) -> dict[str, Any]:
        return dict(self._slots[conversation_id])

    def update_slots(self, conversation_id: str, slots: dict[str, Any]) -> dict[str, Any]:
        clean_slots = {key: value for key, value in slots.items() if value is not None}
        self._slots[conversation_id].update(clean_slots)
        self._persist()
        return self.slots(conversation_id)

    def clear_slots(self, conversation_id: str, keys: list[str] | None = None) -> None:
        if keys is None:
            self._slots[conversation_id].clear()
            self._persist()
            return
        for key in keys:
            self._slots[conversation_id].pop(key, None)
        self._persist()

    def set_pending_task(
        self,
        conversation_id: str,
        *,
        intent: str,
        missing_fields: list[str],
        collected_slots: dict[str, Any] | None = None,
        question: str | None = None,
    ) -> dict[str, Any]:
        task = {
            "intent": intent,
            "missing_fields": missing_fields,
            "collected_slots": collected_slots or {},
            "question": question,
        }
        self._pending_tasks[conversation_id] = task
        if collected_slots:
            self.update_slots(conversation_id, collected_slots)
        self._persist()
        return task

    def pending_task(self, conversation_id: str) -> dict[str, Any] | None:
        task = self._pending_tasks[conversation_id]
        return dict(task) if task else None

    def clear_pending_task(self, conversation_id: str) -> None:
        self._pending_tasks[conversation_id] = None
        self._persist()

    def context(self, conversation_id: str) -> dict[str, Any]:
        return {
            "history": [message.model_dump() for message in self.history(conversation_id)],
            "preferences": self.preferences(conversation_id),
            "slots": self.slots(conversation_id),
            "pending_task": self.pending_task(conversation_id),
            "persistent_memory_allowed": self._persistent_allowed(conversation_id),
            "consent_confirmed": self._consent[conversation_id],
        }

    def _persistent_allowed(self, conversation_id: str) -> bool:
        return bool(self.persist_path) and (not self.persist_requires_consent or self._consent[conversation_id])

    def _persist(self) -> None:
        if not self.persist_path:
            return
        payload: dict[str, Any] = {}
        for conversation_id in set(self._messages) | set(self._preferences) | set(self._slots) | set(self._pending_tasks):
            if not self._persistent_allowed(conversation_id):
                continue
            payload[conversation_id] = {
                "history": [message.model_dump() for message in self._messages[conversation_id]],
                "preferences": dict(self._preferences[conversation_id]),
                "slots": dict(self._slots[conversation_id]),
                "pending_task": self.pending_task(conversation_id),
                "consent_confirmed": self._consent[conversation_id],
            }
        self.persist_path.parent.mkdir(parents=True, exist_ok=True)
        self.persist_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _load(self) -> None:
        if not self.persist_path or not self.persist_path.exists():
            return
        raw = json.loads(self.persist_path.read_text(encoding="utf-8") or "{}")
        for conversation_id, item in raw.items():
            self._consent[conversation_id] = bool(item.get("consent_confirmed"))
            self._preferences[conversation_id].update(item.get("preferences", {}))
            self._slots[conversation_id].update(item.get("slots", {}))
            self._pending_tasks[conversation_id] = item.get("pending_task")
            for message in item.get("history", [])[-self.max_turns :]:
                self._messages[conversation_id].append(ChatMessage(**message))


def _normalize_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    without_marks = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    return without_marks.replace("đ", "d").replace("Ä‘", "d")
