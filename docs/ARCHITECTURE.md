# Architecture Notes

## Positioning

This project is a controlled agentic banking assistant, not an unconstrained autonomous agent. Banking workflows require predictable execution, least-privilege tool access, auditable decisions, and human approval for sensitive actions.

## Key Decisions

### Controlled Agentic Graph

The agent uses LangGraph-style nodes for query understanding, policy checks, planning, retrieval, tool routing, execution, generation, validation, critic review, and retry. This gives more control than a free-form agent loop while still demonstrating planning, tool use, memory, verification, and trajectory capture.

### Tool-First Banking Workflows

Structured banking tools are preferred when the answer depends on calculation, account data, transaction data, loan status, branch data, or fees. RAG provides policy context and caveats; tools provide structured facts.

### Privacy Vault Boundary

PII is tokenized before LLM processing. The current vault is in-memory for portfolio purposes. A production bank implementation should replace it with an encrypted KMS-backed vault with TTL, scope isolation, and audit controls.

### Hybrid Retrieval

The retriever combines deterministic BM25 with vector search, reranking, provenance metadata, and MMR diversification. This reduces dependency on a single embedding similarity signal and provides more explainable retrieval behavior.

### Evaluation Strategy

The project uses deterministic evaluation for CI-style stability and live Groq/Gemini only for demo. Evaluation covers final answers, security, customer journeys, and trajectory-level agent behavior.

## Production Gaps

- Replace mock core banking with API gateway integration.
- Replace local traces with OpenTelemetry or LangSmith.
- Replace local heuristic reranker with cross-encoder or learned reranker.
- Replace synthetic eval data with anonymized, human-labeled logs.
- Add JWT/OAuth session auth, step-up auth, device trust, and consent-token validation.

