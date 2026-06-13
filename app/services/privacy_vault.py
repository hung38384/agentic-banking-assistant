from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from app.services.privacy import PII_PATTERNS


@dataclass(frozen=True)
class VaultToken:
    token: str
    pii_type: str
    original_value: str
    created_at: str
    scope: str


class PrivacyVault:
    """In-memory privacy vault for portfolio-grade PII tokenization.

    Production systems should replace this with an encrypted vault/KMS-backed
    service with TTL, access policies, and audit hooks. The agent receives only
    scoped tokens, never raw personal identifiers.
    """

    def __init__(self) -> None:
        self._tokens: dict[str, VaultToken] = {}

    def tokenize(self, text: str, scope: str) -> tuple[str, list[dict[str, Any]]]:
        tokenized = text
        created: list[VaultToken] = []
        for pii_type, pattern, _ in PII_PATTERNS:
            matches = list(pattern.finditer(tokenized))
            for match in reversed(matches):
                original = match.group(0)
                token = f"[PII:{pii_type}:{uuid4().hex[:10]}]"
                record = VaultToken(
                    token=token,
                    pii_type=pii_type,
                    original_value=original,
                    created_at=datetime.now(timezone.utc).isoformat(),
                    scope=scope,
                )
                self._tokens[token] = record
                created.append(record)
                tokenized = tokenized[: match.start()] + token + tokenized[match.end() :]
        return tokenized, [asdict(item) | {"original_value": "***"} for item in reversed(created)]

    def resolve(self, token: str, scope: str) -> dict[str, Any]:
        record = self._tokens.get(token)
        if not record or record.scope != scope:
            return {"found": False, "reason": "Token not found or scope mismatch."}
        return {
            "found": True,
            "token": token,
            "pii_type": record.pii_type,
            "original_value": record.original_value,
        }

    def stats(self) -> dict[str, Any]:
        counts: dict[str, int] = {}
        for record in self._tokens.values():
            counts[record.pii_type] = counts.get(record.pii_type, 0) + 1
        return {"token_count": len(self._tokens), "by_type": counts}


privacy_vault = PrivacyVault()

