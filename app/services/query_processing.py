from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class QueryPlan:
    original_query: str
    rewritten_query: str
    subqueries: list[str]
    reasoning: list[str]


DOMAIN_SYNONYMS = {
    "repayment": ["monthly payment", "installment", "emi"],
    "credit card": ["card", "classic rewards", "annual fee", "late payment"],
    "savings": ["deposit account", "everyday savings", "atm withdrawal"],
    "home loan": ["mortgage", "housing loan", "outstanding principal"],
    "support": ["ticket", "complaint", "human review", "escalation"],
}


def rewrite_and_decompose_query(query: str) -> QueryPlan:
    normalized = re.sub(r"\s+", " ", query.strip())
    lowered = normalized.lower()
    expansions: list[str] = []
    reasoning: list[str] = []
    for key, synonyms in DOMAIN_SYNONYMS.items():
        if key in lowered or any(term in lowered for term in synonyms):
            expansions.extend([key, *synonyms])
            reasoning.append(f"Expanded banking vocabulary for {key}.")
    rewritten = normalized
    if expansions:
        rewritten = f"{normalized} {' '.join(sorted(set(expansions)))}"

    separators = r"\b(?:and|also|then|,|;)\b"
    candidates = [part.strip(" .?") for part in re.split(separators, normalized, flags=re.IGNORECASE)]
    subqueries = [part for part in candidates if len(part.split()) >= 3]
    if not subqueries:
        subqueries = [normalized]
    if len(subqueries) > 1:
        reasoning.append("Decomposed multi-intent query into smaller retrieval queries.")
    return QueryPlan(
        original_query=normalized,
        rewritten_query=rewritten,
        subqueries=subqueries[:4],
        reasoning=reasoning or ["Query was already specific enough."],
    )

