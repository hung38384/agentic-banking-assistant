# Digital Banking and Core Integration Operating Guide

The digital banking assistant can support account inquiries, transaction questions, loan status checks, support ticket creation, and branch guidance only after customer authentication and consent. Personal banking data must be retrieved from approved core banking APIs or an API gateway. The assistant must not store raw personal data in prompts, vector stores, or analytics logs.

For production integration, the assistant uses an adapter layer between the AI workflow and core banking services. The adapter is responsible for customer identity verification, role-based access control, consent validation, request signing, timeout handling, response masking, and audit logging. The current project uses synthetic MBBank-style mock data to demonstrate this contract.

Account summary responses may include masked account numbers, product names, ledger balance, available balance, hold amount, currency, account status, and branch code. Transaction responses may include transaction id, posted date, description, amount, direction, channel, merchant category, fee, and balance after posting. Loan status responses may include loan id, product name, original amount, outstanding principal, annual rate, remaining months, next due date, next due amount, days past due, and account status.

Sensitive actions require human review or explicit consent, including account data lookup, transaction search, loan status inquiry, support ticket creation, suspected fraud reports, loan recommendation, data correction, and complaint escalation. If consent is missing, the assistant must ask for consent instead of calling personal-data tools.

All personal-data tool calls must create audit events with actor id, role, action, purpose, consent status, PII categories detected, result status, and non-sensitive metadata. Audit logs must not include raw account numbers, card numbers, phone numbers, national id, or full transaction descriptions containing personal identifiers.

