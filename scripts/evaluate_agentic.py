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
from app.services.evaluation import (
    argument_correctness,
    goal_completion,
    policy_correctness,
    tool_correctness,
    trajectory_adherence,
)
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
    start = time.perf_counter()

    for line in Path("data/eval/agentic_trajectories.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        state = await agent.run(f"agentic-{item['id']}", item["question"], item.get("user_profile", {}))
        tools_called = [call.name for call in state.get("tool_calls", [])]
        tool_score = tool_correctness(tools_called, item.get("expected_tools", []))
        argument_score = argument_correctness(state.get("tool_calls", []), item.get("expected_arguments", {}))
        trajectory_score = trajectory_adherence(state.get("trajectory", []), item.get("expected_nodes", []))
        policy_score = policy_correctness(state.get("guardrail_flags", []), item.get("expected_flags", []))
        goal_score = goal_completion(
            state.get("answer", ""),
            state.get("guardrail_flags", []),
            state.get("tool_calls", []),
            item.get("expected_outcome", "answered"),
        )
        critic_score = 1.0 if state.get("critic_review", {}).get("passed") or item.get("expected_outcome") in {"blocked", "hitl_pending"} else 0.0
        rows.append(
            {
                "id": item["id"],
                "tool_correctness": tool_score,
                "argument_correctness": argument_score,
                "trajectory_adherence": trajectory_score,
                "policy_correctness": policy_score,
                "goal_completion": goal_score,
                "critic_pass_rate": critic_score,
                "agentic_score": round(
                    (tool_score + argument_score + trajectory_score + policy_score + goal_score + critic_score) / 6,
                    4,
                ),
            }
        )

    total = len(rows) or 1
    summary = {
        "agentic_cases": len(rows),
        "avg_tool_correctness": round(sum(row["tool_correctness"] for row in rows) / total, 4),
        "avg_argument_correctness": round(sum(row["argument_correctness"] for row in rows) / total, 4),
        "avg_trajectory_adherence": round(sum(row["trajectory_adherence"] for row in rows) / total, 4),
        "avg_policy_correctness": round(sum(row["policy_correctness"] for row in rows) / total, 4),
        "avg_goal_completion": round(sum(row["goal_completion"] for row in rows) / total, 4),
        "avg_critic_pass_rate": round(sum(row["critic_pass_rate"] for row in rows) / total, 4),
        "overall_agentic_score": round(sum(row["agentic_score"] for row in rows) / total, 4),
        "agentic_eval_latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
    print(json.dumps({"summary": summary, "cases": rows}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
