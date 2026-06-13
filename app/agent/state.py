from typing import Any, TypedDict

from app.schemas import Citation, ToolCallRecord


class AgentState(TypedDict, total=False):
    trace_id: str
    conversation_id: str
    question: str
    original_question: str
    user_profile: dict[str, Any]
    preferences: dict[str, Any]
    memory_context: dict[str, Any]
    slots: dict[str, Any]
    intent: str
    plan: dict[str, Any]
    tool_routing: dict[str, Any]
    critic_review: dict[str, Any]
    retry_count: int
    trajectory: list[dict[str, Any]]
    token_budget: dict[str, Any]
    optimization_metrics: dict[str, Any]
    vault_tokens: list[dict[str, Any]]
    rewritten_query: str
    subqueries: list[str]
    needs_retrieval: bool
    needs_tools: bool
    retrieved_chunks: list[dict[str, Any]]
    context: str
    tool_calls: list[ToolCallRecord]
    guardrail_flags: list[str]
    answer: str
    citations: list[Citation]
    token_usage: dict[str, int]
    cache_hit: bool
    hitl_requests: list[dict[str, Any]]
