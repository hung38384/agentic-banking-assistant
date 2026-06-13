import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_DIR = ROOT / "reports"
REPORT_PATH = REPORT_DIR / "EVALUATION_REPORT.md"


def run_json(command: list[str]) -> dict:
    completed = subprocess.run(
        command,
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    stdout = completed.stdout.strip()
    start = stdout.find("{")
    if start == -1:
        raise RuntimeError(f"No JSON output from {' '.join(command)}")
    return json.loads(stdout[start:])


def main() -> None:
    REPORT_DIR.mkdir(exist_ok=True)
    rag = run_json([sys.executable, "scripts/evaluate.py"])
    security = run_json([sys.executable, "scripts/evaluate_security.py"])
    agentic = run_json([sys.executable, "scripts/evaluate_agentic.py"])["summary"]
    product = run_json([sys.executable, "scripts/evaluate_product_intelligence.py"])
    memory = run_json([sys.executable, "scripts/evaluate_memory.py"])
    messy = run_json([sys.executable, "scripts/evaluate_messy_data.py"])
    vietnamese = run_json([sys.executable, "scripts/evaluate_vietnamese.py"])
    hallucination = run_json([sys.executable, "scripts/evaluate_hallucination.py"])
    ragas = run_json([
        sys.executable,
        "scripts/evaluate_ragas.py",
        "--cases",
        "4",
        "--app-provider",
        "mock",
        "--mode",
        "metric",
        "--timeout",
        "30",
        "--max-retries",
        "0",
    ])
    ablation = run_json([sys.executable, "scripts/evaluate_retrieval_ablation.py", "--limit", "50"])
    generated_at = datetime.now(timezone.utc).isoformat()

    report = f"""# Evaluation Report

Generated at: `{generated_at}`

This report is generated from deterministic local evaluation scripts. Live Groq/Gemini runs are intentionally excluded from CI-style metrics to avoid quota, latency, and nondeterminism.

## RAG And Tool Evaluation

| Metric | Value |
|---|---:|
| Questions | {rag["questions"]} |
| Retrieval top-k accuracy | {rag["retrieval_top_k_accuracy"]:.2%} |
| Intent accuracy | {rag["intent_accuracy"]:.2%} |
| Groundedness proxy | {rag["faithfulness_proxy"]:.2%} |
| Avg latency | {rag["avg_latency_ms"]} ms |
| Avg token usage | {rag["avg_token_usage"]} |
| Tool-call success rate | {rag["tool_call_success_rate"]:.2%} |

## Security Harness

| Metric | Value |
|---|---:|
| Red-team cases | {security["red_team_cases"]} |
| Red-team pass rate | {security["red_team_pass_rate"]:.2%} |
| Golden cases | {security["golden_cases"]} |
| Golden pass rate | {security["golden_pass_rate"]:.2%} |
| Customer journeys | {security["journey_cases"]} |
| Journey pass rate | {security["journey_pass_rate"]:.2%} |

## Hallucination And RAGAS-Lite Groundedness

| Metric | Value |
|---|---:|
| Hallucination cases | {hallucination["hallucination_cases"]} |
| Pass rate | {hallucination["pass_rate"]:.2%} |
| Faithfulness | {hallucination["faithfulness"]:.2%} |
| Context precision | {hallucination["context_precision"]:.2%} |
| Citation precision | {hallucination["citation_precision"]:.2%} |
| Numeric consistency | {hallucination["numeric_consistency"]:.2%} |
| Unsupported refusal rate | {hallucination["unsupported_refusal_rate"]:.2%} |
| Avg hallucination risk | {hallucination["avg_hallucination_risk"]:.2%} |

## Real RAGAS Evaluation

| Metric | Value |
|---|---:|
| Status | {ragas["status"]} |
| Cases prepared | {ragas.get("cases", ragas.get("prepared_cases", 0))} |
| Metrics | `{json.dumps(ragas.get("metrics", {}), ensure_ascii=False)}` |
| Note | `{ragas.get("reason", "Run with optional [ragas] dependencies and a judge API key.")}` |

## Retrieval Ablation Study

| Variant | Retrieval | Intent | Tool Success | Avg Tokens |
|---|---:|---:|---:|---:|
{_ablation_rows(ablation)}

## Agentic Trajectory Evaluation

| Metric | Value |
|---|---:|
| Agentic cases | {agentic["agentic_cases"]} |
| Tool correctness | {agentic["avg_tool_correctness"]:.2%} |
| Argument correctness | {agentic["avg_argument_correctness"]:.2%} |
| Trajectory adherence | {agentic["avg_trajectory_adherence"]:.2%} |
| Policy correctness | {agentic["avg_policy_correctness"]:.2%} |
| Goal completion | {agentic["avg_goal_completion"]:.2%} |
| Critic pass rate | {agentic["avg_critic_pass_rate"]:.2%} |
| Overall agentic score | {agentic["overall_agentic_score"]:.2%} |

## Product Intelligence And Token Optimization

| Metric | Value |
|---|---:|
| Proactive insights | {product["insight_count"]} |
| Recommended actions | {product["recommended_action_count"]} |
| Tool payload compaction ratio | {product["token_compaction_ratio"]:.2%} |
| Estimated tokens saved | {product["estimated_tokens_saved"]} |
| Content gap categories | {product["content_gap_categories"]} |

## Multi-Turn Memory

| Metric | Value |
|---|---:|
| Memory cases | {memory["memory_cases"]} |
| Loan slot-fill success | {str(memory["loan_slot_fill_success"])} |
| Eligibility slot-fill success | {str(memory["eligibility_slot_fill_success"])} |
| Clarification detected | {str(memory["clarification_detected"])} |

## Vietnamese Banking Experience

| Metric | Value |
|---|---:|
| Vietnamese cases | {vietnamese["vietnamese_cases"]} |
| Intent accuracy | {vietnamese["intent_accuracy"]:.2%} |
| Tool accuracy | {vietnamese["tool_accuracy"]:.2%} |
| Guardrail accuracy | {vietnamese["guardrail_accuracy"]:.2%} |
| Memory slot-fill accuracy | {vietnamese["memory_slot_fill_accuracy"]:.2%} |
| Avg suite latency | {vietnamese["latency_ms"]} ms |

## Messy Banking Data Realism

| Metric | Value |
|---|---:|
| Messy customer id | {messy["messy_customer_id"]} |
| Unique data-quality flags | {messy["unique_quality_flag_count"]} |
| Account flags | `{", ".join(messy["account_quality_flags"])}` |
| Transaction flags | `{", ".join(messy["transaction_quality_flags"])}` |
| Loan flags | `{", ".join(messy["loan_quality_flags"])}` |
| Branch flags | `{", ".join(messy["branch_quality_flags"])}` |

## Interpretation

- The system is a controlled agentic banking assistant, not an unconstrained autonomous agent.
- Metrics are synthetic regression indicators, not claims of production performance.
- The strongest evidence is trajectory-level evaluation: the project verifies planning, policy, tool routing, tool arguments, HITL outcomes, critic review, and final goal completion.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(json.dumps({"report_path": str(REPORT_PATH), "generated_at": generated_at}, indent=2))


def _ablation_rows(ablation: dict) -> str:
    rows = []
    for item in ablation.get("variants", []):
        rows.append(
            "| {variant} | {retrieval:.2%} | {intent:.2%} | {tool:.2%} | {tokens} |".format(
                variant=item["variant"],
                retrieval=item["retrieval_top_k_accuracy"],
                intent=item["intent_accuracy"],
                tool=item["tool_success_rate"],
                tokens=item["avg_token_usage"],
            )
        )
    return "\n".join(rows)


if __name__ == "__main__":
    main()
