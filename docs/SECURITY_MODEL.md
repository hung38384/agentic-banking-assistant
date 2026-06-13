# Security Model

## Threats Covered

- Prompt injection from users.
- SQL injection simulation.
- Sensitive information disclosure.
- Excessive agency through high-risk tools.
- Unauthorized personal-data access.
- Missing consent for customer-specific workflows.
- PII leakage into model prompts.

## Controls

- PII tokenization through Privacy Vault.
- PII redaction fallback.
- Policy-as-code guardrails.
- Role-based access control.
- Consent checks.
- Tool risk scoring.
- HITL approval queue.
- Audit logging for personal-data tools.
- Prompt and SQL injection benchmark.
- Agent trajectory evaluation.

## Intentional Limits

This is a portfolio simulation. It does not include real customer authentication, KMS, hardware security modules, data retention enforcement, SIEM integration, network controls, or regulatory approval workflows.

## Production Upgrade Path

1. Use OAuth/JWT and bank IAM scopes.
2. Store vault mappings in encrypted KMS-backed storage with TTL.
3. Enforce consent tokens issued by a consent management service.
4. Stream audit events to a SIEM.
5. Add OpenTelemetry traces and LangSmith evaluation datasets.
6. Run red-team evaluation against anonymized historical chat logs.

