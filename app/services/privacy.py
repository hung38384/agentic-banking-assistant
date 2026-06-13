from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class RedactionResult:
    original: str
    redacted: str
    pii_types: list[str]


PII_PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    ("thai_national_id", re.compile(r"\b\d{1}[- ]?\d{4}[- ]?\d{5}[- ]?\d{2}[- ]?\d{1}\b"), "[REDACTED_NATIONAL_ID]"),
    ("card_number", re.compile(r"\b(?:\d[ -]?){13,19}\b"), "[REDACTED_CARD_NUMBER]"),
    ("phone_number", re.compile(r"\b(?:\+66|0)\d{8,9}\b"), "[REDACTED_PHONE]"),
    ("email", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "[REDACTED_EMAIL]"),
    ("bank_account", re.compile(r"\b\d{3}[- ]?\d{1}[- ]?\d{5}[- ]?\d{1}\b"), "[REDACTED_ACCOUNT]"),
]


def redact_text(text: str) -> RedactionResult:
    redacted = text
    detected: list[str] = []
    for pii_type, pattern, replacement in PII_PATTERNS:
        if pattern.search(redacted):
            detected.append(pii_type)
            redacted = pattern.sub(replacement, redacted)
    return RedactionResult(original=text, redacted=redacted, pii_types=sorted(set(detected)))


def mask_identifier(value: str, visible_last: int = 4) -> str:
    cleaned = re.sub(r"\s+", "", str(value))
    if len(cleaned) <= visible_last:
        return "*" * len(cleaned)
    return f"{'*' * (len(cleaned) - visible_last)}{cleaned[-visible_last:]}"

