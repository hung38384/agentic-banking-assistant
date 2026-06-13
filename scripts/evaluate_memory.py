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
    settings.llm_provider = "mock"
    settings.semantic_cache_enabled = False
    store = BankingVectorStore(settings)
    store.ingest_path(settings.docs_dir, reset=True)
    memory = ConversationMemory()
    agent = BankingAgent(settings, store, memory)
    start = time.perf_counter()

    loan_first = await agent.run("eval-memory-loan", "Calculate a 100000 VND loan at 6 percent", {})
    loan_second = await agent.run("eval-memory-loan", "24 months", {})
    eligibility_first = await agent.run(
        "eval-memory-eligibility",
        "Am I eligible for a home loan? My age 25 and income 30000000",
        {},
    )
    eligibility_second = await agent.run("eval-memory-eligibility", "12 months employed", {})

    summary = {
        "memory_cases": 2,
        "loan_slot_fill_success": bool(
            loan_second.get("tool_calls")
            and loan_second["tool_calls"][0].output.get("monthly_payment")
        ),
        "eligibility_slot_fill_success": bool(
            eligibility_second.get("tool_calls")
            and eligibility_second["tool_calls"][0].output.get("eligible") is True
        ),
        "clarification_detected": "clarification_required" in loan_first.get("guardrail_flags", [])
        and "clarification_required" in eligibility_first.get("guardrail_flags", []),
        "loan_memory": memory.context("eval-memory-loan"),
        "eligibility_memory": memory.context("eval-memory-eligibility"),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

