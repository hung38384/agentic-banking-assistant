from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any


STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "based",
    "be",
    "by",
    "can",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "with",
    "available",
    "bank",
    "banking",
    "documents",
    "tool",
    "tools",
    "result",
    "results",
}


@dataclass(frozen=True)
class HallucinationScore:
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    citation_precision: float
    numeric_consistency: float
    unsupported_refusal: float
    hallucination_risk: float
    unsupported_claims: list[str]
    unsupported_numbers: list[str]


def evaluate_groundedness(
    *,
    question: str,
    answer: str,
    retrieved_chunks: list[dict[str, Any]],
    citations: list[Any],
    tool_calls: list[Any],
    guardrail_flags: list[str],
    expect_refusal: bool = False,
) -> HallucinationScore:
    evidence = _evidence_text(retrieved_chunks, tool_calls)
    claims = _extract_claims(answer)
    refusal_ok = 1.0
    if expect_refusal:
        refusal_ok = 1.0 if _looks_like_refusal(answer, guardrail_flags) else 0.0
    unsupported_claims = (
        []
        if expect_refusal and refusal_ok == 1.0
        else [claim for claim in claims if not _is_claim_supported(claim, evidence, guardrail_flags)]
    )
    faithfulness = 1.0 if not claims else 1 - (len(unsupported_claims) / len(claims))

    answer_relevancy = _overlap_score(question, answer)
    context_precision = _context_precision(question, retrieved_chunks)
    citation_precision = _citation_precision(citations, retrieved_chunks)

    unsupported_numbers = _unsupported_numbers(answer, evidence, guardrail_flags)
    numeric_consistency = 1.0 if not unsupported_numbers else 0.0

    risk = 1 - (
        0.35 * faithfulness
        + 0.2 * numeric_consistency
        + 0.15 * citation_precision
        + 0.15 * context_precision
        + 0.15 * refusal_ok
    )
    return HallucinationScore(
        faithfulness=round(max(0.0, faithfulness), 4),
        answer_relevancy=round(answer_relevancy, 4),
        context_precision=round(context_precision, 4),
        citation_precision=round(citation_precision, 4),
        numeric_consistency=round(numeric_consistency, 4),
        unsupported_refusal=round(refusal_ok, 4),
        hallucination_risk=round(max(0.0, risk), 4),
        unsupported_claims=unsupported_claims,
        unsupported_numbers=unsupported_numbers,
    )


def _evidence_text(retrieved_chunks: list[dict[str, Any]], tool_calls: list[Any]) -> str:
    chunk_text = "\n".join(chunk.get("text", "") for chunk in retrieved_chunks)
    tool_text = "\n".join(json.dumps(_tool_output(call), sort_keys=True, default=str) for call in tool_calls)
    return f"{chunk_text}\n{tool_text}".lower().replace(",", "")


def _tool_output(call: Any) -> dict[str, Any]:
    if isinstance(call, dict):
        return call.get("output", {})
    return getattr(call, "output", {})


def _extract_claims(answer: str) -> list[str]:
    sentences = re.split(r"(?<=[.!?])\s+|\n+", answer.strip())
    return [
        sentence.strip(" -")
        for sentence in sentences
        if len(_content_tokens(sentence)) >= 3 and not _looks_like_generic_prefix(sentence)
    ]


def _is_claim_supported(claim: str, evidence: str, guardrail_flags: list[str]) -> bool:
    if _looks_like_refusal(claim, guardrail_flags):
        return True
    tokens = _content_tokens(claim)
    if not tokens:
        return True
    overlap = sum(1 for token in tokens if token in evidence)
    numeric_claims = _numbers(claim)
    numeric_supported = all(number in evidence for number in numeric_claims)
    return numeric_supported and overlap / len(tokens) >= 0.45


def _unsupported_numbers(answer: str, evidence: str, guardrail_flags: list[str]) -> list[str]:
    if _looks_like_refusal(answer, guardrail_flags):
        return []
    return sorted({number for number in _numbers(answer) if number not in evidence})


def _context_precision(question: str, retrieved_chunks: list[dict[str, Any]]) -> float:
    if not retrieved_chunks:
        return 1.0 if "could not find" in question.lower() else 0.0
    relevant = sum(1 for chunk in retrieved_chunks if _overlap_score(question, chunk.get("text", "")) >= 0.12)
    return relevant / len(retrieved_chunks)


def _citation_precision(citations: list[Any], retrieved_chunks: list[dict[str, Any]]) -> float:
    if not citations:
        return 1.0 if not retrieved_chunks else 0.0
    retrieved_ids = {chunk.get("metadata", {}).get("chunk_id") for chunk in retrieved_chunks}
    cited_ids = {
        getattr(citation, "chunk_id", citation.get("chunk_id", None))
        if isinstance(citation, dict)
        else getattr(citation, "chunk_id", None)
        for citation in citations
    }
    return len(cited_ids & retrieved_ids) / max(len(cited_ids), 1)


def _overlap_score(left: str, right: str) -> float:
    left_tokens = set(_content_tokens(left))
    right_tokens = set(_content_tokens(right))
    if not left_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens)


def _content_tokens(text: str) -> list[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_%-]*|\d+(?:\.\d+)?%?", text.lower())
        if token not in STOPWORDS and len(token) > 1
    ]


def _numbers(text: str) -> list[str]:
    return re.findall(r"\d+(?:\.\d+)?%?", text.lower().replace(",", ""))


def _looks_like_refusal(text: str, guardrail_flags: list[str]) -> bool:
    lowered = text.lower()
    return (
        "no_retrieval_context" in guardrail_flags
        or "unsupported_financial_advice" in guardrail_flags
        or "could not find" in lowered
        or "cannot provide" in lowered
        or "must not invent" in lowered
        or "must not guarantee" in lowered
        or "not guarantee" in lowered
        or "human review" in lowered
    )


def _looks_like_generic_prefix(sentence: str) -> bool:
    return sentence.lower().strip().startswith("based on the available bank documents")
