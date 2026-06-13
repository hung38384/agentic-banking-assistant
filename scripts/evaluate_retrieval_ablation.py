import argparse
import asyncio
import json
import sys
import time
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.agent.graph import BankingAgent
from app.core.config import get_settings
from app.services.memory import ConversationMemory
from app.services.vector_store import BankingVectorStore


VARIANTS = {
    "hybrid_default": {"hybrid_vector_weight": 0.65, "hybrid_bm25_weight": 0.35, "top_k": 4, "max_context_chars": 4500, "rerank_top_n": 8},
    "vector_heavy": {"hybrid_vector_weight": 0.9, "hybrid_bm25_weight": 0.1, "top_k": 4, "max_context_chars": 4500, "rerank_top_n": 8},
    "bm25_heavy": {"hybrid_vector_weight": 0.2, "hybrid_bm25_weight": 0.8, "top_k": 4, "max_context_chars": 4500, "rerank_top_n": 8},
    "high_recall_top6": {"hybrid_vector_weight": 0.65, "hybrid_bm25_weight": 0.35, "top_k": 6, "max_context_chars": 6500, "rerank_top_n": 12},
    "compressed_top3": {"hybrid_vector_weight": 0.65, "hybrid_bm25_weight": 0.35, "top_k": 3, "max_context_chars": 1800, "rerank_top_n": 8},
}


async def main() -> None:
    parser = argparse.ArgumentParser(description="Compare retrieval/RAG configurations.")
    parser.add_argument("--limit", type=int, default=50)
    args = parser.parse_args()

    base_settings = get_settings()
    base_settings.llm_provider = "mock"
    base_settings.semantic_cache_enabled = False
    eval_items = [
        json.loads(line)
        for line in (ROOT / "data" / "eval_questions.jsonl").read_text(encoding="utf-8-sig").splitlines()
        if line.strip()
    ][: args.limit]

    rows = []
    for name, overrides in VARIANTS.items():
        settings = deepcopy(base_settings)
        for key, value in overrides.items():
            setattr(settings, key, value)
        store = BankingVectorStore(settings)
        store.ingest_path(settings.docs_dir, reset=True)
        agent = BankingAgent(settings, store, ConversationMemory())

        start = time.perf_counter()
        per_case = []
        for item in eval_items:
            profile = (
                {"role": "customer", "customer_id": "CUST1001", "consent_confirmed": True}
                if item["expected_intent"] in {"account_summary", "transaction_search", "loan_status", "support"}
                else {}
            )
            state = await agent.run(f"ablation-{name}-{item['id']}", item["question"], profile)
            citations = [citation.source for citation in state.get("citations", [])]
            per_case.append(
                {
                    "id": item["id"],
                    "intent_ok": state.get("intent") == item["expected_intent"],
                    "source_hit": any(item["expected_source"] in source for source in citations),
                    "tool_success": all(call.success for call in state.get("tool_calls", []) if call.name not in {"human_approval_interrupt", "access_control", "consent_check"}),
                    "tokens": state.get("token_usage", {}).get("total_tokens", 0),
                }
            )
        total = len(per_case)
        rows.append(
            {
                "variant": name,
                "cases": total,
                "retrieval_top_k_accuracy": round(sum(row["source_hit"] for row in per_case) / total, 4),
                "intent_accuracy": round(sum(row["intent_ok"] for row in per_case) / total, 4),
                "tool_success_rate": round(sum(row["tool_success"] for row in per_case) / total, 4),
                "avg_token_usage": round(sum(row["tokens"] for row in per_case) / total, 2),
                "latency_ms": round((time.perf_counter() - start) * 1000, 2),
            }
        )

    best = sorted(rows, key=lambda row: (row["retrieval_top_k_accuracy"], -row["avg_token_usage"]), reverse=True)[0]
    print(json.dumps({"cases_per_variant": len(eval_items), "best_variant": best["variant"], "variants": rows}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
