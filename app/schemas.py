from typing import Any, Literal

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=3000)
    conversation_id: str = "default"
    user_profile: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    source: str
    title: str
    chunk_id: str


class ToolCallRecord(BaseModel):
    name: str
    input: dict[str, Any]
    output: dict[str, Any]
    success: bool


class AgentPlanSchema(BaseModel):
    rewritten_query: str
    subqueries: list[str] = Field(default_factory=list)
    retrieval_strategy: Literal["none", "vector", "hybrid"] = "hybrid"
    required_tools: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    risk_level: Literal["low", "medium", "high"] = "low"
    requires_human_approval: bool = False


class ToolRoutingDecisionSchema(BaseModel):
    selected_tool: str | None = None
    reason: str
    expected_arguments: dict[str, Any] = Field(default_factory=dict)
    risk_level: Literal["low", "medium", "high"] = "low"
    allowed_by_scope: bool = True
    requires_human_approval: bool = False


class AnswerValidationSchema(BaseModel):
    valid: bool
    grounded: bool
    flags: list[str] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)


class CriticReviewSchema(BaseModel):
    passed: bool
    grounded: bool
    tool_use_correct: bool
    policy_compliant: bool
    needs_retry: bool = False
    retry_reason: str | None = None
    confidence: float = Field(ge=0, le=1)


class ChatResponse(BaseModel):
    conversation_id: str
    answer: str
    intent: str
    trace_id: str | None = None
    cache_hit: bool = False
    citations: list[Citation] = Field(default_factory=list)
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    guardrail_flags: list[str] = Field(default_factory=list)
    trajectory: list[dict[str, Any]] = Field(default_factory=list)
    critic_review: dict[str, Any] = Field(default_factory=dict)
    latency_ms: float
    token_usage: dict[str, int] = Field(default_factory=dict)


class IngestRequest(BaseModel):
    docs_path: str | None = None
    reset: bool = True


class IngestResponse(BaseModel):
    ingested_documents: int
    chunks: int
    vector_backend: str


class HealthResponse(BaseModel):
    status: str
    vector_backend: str


class FeedbackRequest(BaseModel):
    conversation_id: str
    rating: int = Field(ge=1, le=5)
    category: str = "general"
    comment: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ActionRequest(BaseModel):
    action_type: str
    customer_id: str
    payload: dict[str, Any] = Field(default_factory=dict)
