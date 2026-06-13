import asyncio
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import BankingAgent
from app.core.config import get_settings
from app.services.hallucination import evaluate_groundedness
from app.services.memory import ConversationMemory
from app.services.vector_store import BankingVectorStore


async def main() -> None:
    settings = get_settings()
    settings.semantic_cache_enabled = False
    settings.llm_provider = "mock"
    store = BankingVectorStore(settings)
    store.ingest_path(settings.docs_dir, reset=True)
    agent = BankingAgent(settings, store, ConversationMemory())

    rows = []
    eval_path = ROOT / "data" / "eval" / "hallucination_cases.jsonl"
    suite_start = time.perf_counter()
    for line in eval_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        start = time.perf_counter()
        state = await agent.run(f"hallucination-{item['id']}", item["question"], {})
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        score = evaluate_groundedness(
            question=item["question"],
            answer=state.get("answer", ""),
            retrieved_chunks=state.get("retrieved_chunks", []),
            citations=state.get("citations", []),
            tool_calls=state.get("tool_calls", []),
            guardrail_flags=state.get("guardrail_flags", []),
            expect_refusal=item.get("expect_refusal", False),
        )
        passed = (
            score.faithfulness >= 0.75
            and score.numeric_consistency == 1.0
            and score.citation_precision >= 0.9
            and score.hallucination_risk <= 0.25
            and (not item.get("expect_refusal") or score.unsupported_refusal == 1.0)
        )
        rows.append(
            {
                "id": item["id"],
                "passed": passed,
                "intent_ok": state.get("intent") == item.get("expected_intent"),
                "faithfulness": score.faithfulness,
                "answer_relevancy": score.answer_relevancy,
                "context_precision": score.context_precision,
                "citation_precision": score.citation_precision,
                "numeric_consistency": score.numeric_consistency,
                "unsupported_refusal": score.unsupported_refusal,
                "hallucination_risk": score.hallucination_risk,
                "unsupported_claim_count": len(score.unsupported_claims),
                "unsupported_numbers": score.unsupported_numbers,
                "latency_ms": latency_ms,
            }
        )

    total = len(rows)
    summary = {
        "hallucination_cases": total,
        "pass_rate": round(sum(row["passed"] for row in rows) / total, 4),
        "intent_accuracy": round(sum(row["intent_ok"] for row in rows) / total, 4),
        "faithfulness": round(sum(row["faithfulness"] for row in rows) / total, 4),
        "answer_relevancy": round(sum(row["answer_relevancy"] for row in rows) / total, 4),
        "context_precision": round(sum(row["context_precision"] for row in rows) / total, 4),
        "citation_precision": round(sum(row["citation_precision"] for row in rows) / total, 4),
        "numeric_consistency": round(sum(row["numeric_consistency"] for row in rows) / total, 4),
        "unsupported_refusal_rate": round(sum(row["unsupported_refusal"] for row in rows) / total, 4),
        "avg_hallucination_risk": round(sum(row["hallucination_risk"] for row in rows) / total, 4),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / total, 2),
        "suite_latency_ms": round((time.perf_counter() - suite_start) * 1000, 2),
        "cases": rows,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    asyncio.run(main())
