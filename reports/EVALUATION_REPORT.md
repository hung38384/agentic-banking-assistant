# Evaluation Report

Generated at: `2026-06-13T06:15:50.918808+00:00`

This report is generated from deterministic local evaluation scripts. Live Groq/Gemini runs are intentionally excluded from CI-style metrics to avoid quota, latency, and nondeterminism.

## RAG And Tool Evaluation

| Metric | Value |
|---|---:|
| Questions | 90 |
| Retrieval top-k accuracy | 96.67% |
| Intent accuracy | 98.89% |
| Groundedness proxy | 100.00% |
| Avg latency | 33.13 ms |
| Avg token usage | 783.46 |
| Tool-call success rate | 100.00% |

## Security Harness

| Metric | Value |
|---|---:|
| Red-team cases | 12 |
| Red-team pass rate | 100.00% |
| Golden cases | 8 |
| Golden pass rate | 100.00% |
| Customer journeys | 4 |
| Journey pass rate | 100.00% |

## Hallucination And RAGAS-Lite Groundedness

| Metric | Value |
|---|---:|
| Hallucination cases | 8 |
| Pass rate | 100.00% |
| Faithfulness | 100.00% |
| Context precision | 87.50% |
| Citation precision | 100.00% |
| Numeric consistency | 100.00% |
| Unsupported refusal rate | 100.00% |
| Avg hallucination risk | 1.87% |

## Real RAGAS Evaluation

| Metric | Value |
|---|---:|
| Status | completed |
| Cases prepared | 4 |
| Metrics | `{"faithfulness": 1.0, "context_precision": 1.0, "answer_relevancy": 1.0, "rows": [{"id": "r001", "faithfulness": "pass", "context_precision": "pass", "answer_relevancy": "pass"}, {"id": "r002", "faithfulness": "pass", "context_precision": "pass", "answer_relevancy": "pass"}, {"id": "r003", "faithfulness": "pass", "context_precision": "pass", "answer_relevancy": "pass"}, {"id": "r004", "faithfulness": "pass", "context_precision": "pass", "answer_relevancy": "pass"}]}` |
| Note | `Run with optional [ragas] dependencies and a judge API key.` |

## Retrieval Ablation Study

| Variant | Retrieval | Intent | Tool Success | Avg Tokens |
|---|---:|---:|---:|---:|
| hybrid_default | 98.00% | 98.00% | 100.00% | 810.12 |
| vector_heavy | 98.00% | 98.00% | 100.00% | 809.56 |
| bm25_heavy | 98.00% | 98.00% | 100.00% | 810.12 |
| high_recall_top6 | 96.00% | 98.00% | 100.00% | 796.48 |
| compressed_top3 | 98.00% | 98.00% | 100.00% | 810.12 |

## Agentic Trajectory Evaluation

| Metric | Value |
|---|---:|
| Agentic cases | 8 |
| Tool correctness | 100.00% |
| Argument correctness | 100.00% |
| Trajectory adherence | 100.00% |
| Policy correctness | 100.00% |
| Goal completion | 100.00% |
| Critic pass rate | 100.00% |
| Overall agentic score | 100.00% |

## Product Intelligence And Token Optimization

| Metric | Value |
|---|---:|
| Proactive insights | 3 |
| Recommended actions | 3 |
| Tool payload compaction ratio | 55.39% |
| Estimated tokens saved | 149 |
| Content gap categories | 1 |

## Multi-Turn Memory

| Metric | Value |
|---|---:|
| Memory cases | 2 |
| Loan slot-fill success | True |
| Eligibility slot-fill success | True |
| Clarification detected | True |

## Vietnamese Banking Experience

| Metric | Value |
|---|---:|
| Vietnamese cases | 7 |
| Intent accuracy | 100.00% |
| Tool accuracy | 100.00% |
| Guardrail accuracy | 100.00% |
| Memory slot-fill accuracy | 100.00% |
| Avg suite latency | 284.29 ms |

## Messy Banking Data Realism

| Metric | Value |
|---|---:|
| Messy customer id | CUST2002 |
| Unique data-quality flags | 19 |
| Account flags | `account_dormant, missing_hold_amount, missing_income_document, missing_product_name, negative_available_balance, phone_unverified` |
| Transaction flags | `duplicate_transaction_id, missing_merchant_category, pending_authorization, reversal_transaction, standalone_fee_posting` |
| Loan flags | `floating_rate_missing_reset_date, loan_delinquent, loan_restructured, missing_next_due_amount` |
| Branch flags | `appointment_required, branch_temporarily_closed, missing_services, stale_branch_metadata` |

## Interpretation

- The system is a controlled agentic banking assistant, not an unconstrained autonomous agent.
- Metrics are synthetic regression indicators, not claims of production performance.
- The strongest evidence is trajectory-level evaluation: the project verifies planning, policy, tool routing, tool arguments, HITL outcomes, critic review, and final goal completion.
