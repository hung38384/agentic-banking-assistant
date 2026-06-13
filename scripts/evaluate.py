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
    eval_path = Path("data/eval_questions.jsonl")
    for line in eval_path.read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        start = time.perf_counter()
        user_profile = (
            {"role": "customer", "customer_id": "CUST1001", "consent_confirmed": True}
            if item["expected_intent"] in {"account_summary", "transaction_search", "loan_status", "support"}
            else {}
        )
        state = await agent.run(f"eval-{item['id']}", item["question"], user_profile)
        latency_ms = round((time.perf_counter() - start) * 1000, 2)
        citation_sources = [citation.source for citation in state.get("citations", [])]
        source_hit = any(item["expected_source"] in source for source in citation_sources)
        tool_success = all(
            call.success or call.name in {"human_approval_interrupt", "access_control", "consent_check"}
            for call in state.get("tool_calls", [])
        )
        grounded = bool(citation_sources or state.get("tool_calls") or state.get("guardrail_flags"))
        rows.append(
            {
                "id": item["id"],
                "intent_ok": state.get("intent") == item["expected_intent"],
                "source_hit": source_hit,
                "grounded": grounded,
                "latency_ms": latency_ms,
                "tokens": state.get("token_usage", {}).get("total_tokens", 0),
                "tool_success": tool_success,
            }
        )

    total = len(rows)
    summary = {
        "questions": total,
        "retrieval_top_k_accuracy": round(sum(row["source_hit"] for row in rows) / total, 4),
        "intent_accuracy": round(sum(row["intent_ok"] for row in rows) / total, 4),
        "faithfulness_proxy": round(sum(row["grounded"] for row in rows) / total, 4),
        "avg_latency_ms": round(sum(row["latency_ms"] for row in rows) / total, 2),
        "avg_token_usage": round(sum(row["tokens"] for row in rows) / total, 2),
        "tool_call_success_rate": round(sum(row["tool_success"] for row in rows) / total, 4),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
