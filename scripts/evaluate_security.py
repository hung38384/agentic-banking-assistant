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
from app.services.evaluation import evaluate_attack_blocked, local_llm_as_judge
from app.services.memory import ConversationMemory
from app.services.vector_store import BankingVectorStore


AUTH_PROFILE = {"role": "customer", "customer_id": "CUST1001", "consent_confirmed": True}


async def run_red_team(agent: BankingAgent) -> dict[str, float | int]:
    rows = []
    for line in Path("data/eval/red_team_cases.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        profile = AUTH_PROFILE if item["category"] not in {"tool_abuse", "data_leakage"} else {}
        state = await agent.run(f"redteam-{item['id']}", item["question"], profile)
        blocked_ok = evaluate_attack_blocked(item["question"], state.get("guardrail_flags", []), state.get("tool_calls", []))
        expected_flag = item.get("expected_flag")
        flag_ok = expected_flag is None or expected_flag in state.get("guardrail_flags", [])
        rows.append(blocked_ok and flag_ok)
    return {"red_team_cases": len(rows), "red_team_pass_rate": round(sum(rows) / max(len(rows), 1), 4)}


async def run_golden(agent: BankingAgent) -> dict[str, float | int]:
    rows = []
    for line in Path("data/eval/golden_answers.jsonl").read_text(encoding="utf-8-sig").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        profile = AUTH_PROFILE if item.get("auth") else {}
        state = await agent.run(f"golden-{item['id']}", item["question"], profile)
        answer_blob = (state.get("answer", "") + " " + str([call.output for call in state.get("tool_calls", [])])).lower()
        contains_ok = all(term.lower() in answer_blob for term in item.get("must_contain", []))
        intent_ok = item.get("expected_intent") is None or state.get("intent") == item.get("expected_intent")
        flag_ok = item.get("expected_flag") is None or item.get("expected_flag") in state.get("guardrail_flags", [])
        judge = local_llm_as_judge(state.get("answer", ""), state.get("citations", []), state.get("tool_calls", []), state.get("guardrail_flags", []))
        rows.append(contains_ok and intent_ok and flag_ok and judge.score >= 0.66)
    return {"golden_cases": len(rows), "golden_pass_rate": round(sum(rows) / max(len(rows), 1), 4)}


async def run_journeys(agent: BankingAgent) -> dict[str, float | int]:
    journeys = json.loads(Path("data/eval/customer_journeys.json").read_text(encoding="utf-8-sig"))
    passes = 0
    for journey in journeys:
        states = []
        for turn in journey["turns"]:
            states.append(await agent.run(journey["id"], turn, journey["user_profile"]))
        first_flags = states[0].get("guardrail_flags", [])
        final_flags = states[-1].get("guardrail_flags", [])
        ok = True
        if "expected_final_intent" in journey:
            ok = ok and states[-1].get("intent") == journey["expected_final_intent"]
        if "expected_final_flag" in journey:
            ok = ok and journey["expected_final_flag"] in final_flags
        if "expected_first_flag" in journey:
            ok = ok and journey["expected_first_flag"] in first_flags
        passes += int(ok)
    return {"journey_cases": len(journeys), "journey_pass_rate": round(passes / max(len(journeys), 1), 4)}


async def main() -> None:
    settings = get_settings()
    settings.semantic_cache_enabled = False
    settings.llm_provider = "mock"
    store = BankingVectorStore(settings)
    store.ingest_path(settings.docs_dir, reset=True)
    agent = BankingAgent(settings, store, ConversationMemory())
    start = time.perf_counter()
    summary = {}
    summary.update(await run_red_team(agent))
    summary.update(await run_golden(agent))
    summary.update(await run_journeys(agent))
    summary["security_eval_latency_ms"] = round((time.perf_counter() - start) * 1000, 2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
