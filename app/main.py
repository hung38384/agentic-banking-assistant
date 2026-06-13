import time
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import Depends, FastAPI, HTTPException

from app.agent.graph import BankingAgent
from app.api.auth import require_api_key
from app.core.config import get_settings
from app.core.logging import configure_logging
from app.schemas import ActionRequest, ChatRequest, ChatResponse, FeedbackRequest, HealthResponse, IngestRequest, IngestResponse
from app.services.audit import audit_logger
from app.services.actions import action_engine
from app.services.cost_tracker import cost_tracker
from app.services.feedback import feedback_store
from app.services.hitl import approval_queue
from app.services.insights import insight_engine
from app.services.memory import ConversationMemory
from app.services.metrics import metrics
from app.services.observability import trace_recorder
from app.services.privacy_vault import privacy_vault
from app.services.vector_store import BankingVectorStore

configure_logging()
settings = get_settings()
logger = structlog.get_logger()

store = BankingVectorStore(settings)
memory = ConversationMemory(
    persist_path=settings.memory_path if settings.memory_persistence_enabled else None,
    persist_requires_consent=settings.memory_persist_requires_consent,
)
agent = BankingAgent(settings, store, memory)


@asynccontextmanager
async def lifespan(_: FastAPI):
    if settings.docs_dir.exists():
        try:
            docs, chunks = store.ingest_path(settings.docs_dir, reset=True)
            logger.info("startup_ingest_complete", docs=docs, chunks=chunks, backend=store.backend)
        except Exception as exc:
            logger.warning("startup_ingest_failed", error=str(exc))
    yield


app = FastAPI(title=settings.app_name, version="0.1.0", lifespan=lifespan)


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok", vector_backend=store.backend)


@app.post("/ingest", response_model=IngestResponse, dependencies=[Depends(require_api_key)])
async def ingest(request: IngestRequest) -> IngestResponse:
    docs_path = Path(request.docs_path) if request.docs_path else settings.docs_dir
    if not docs_path.exists():
        raise HTTPException(status_code=404, detail=f"Docs path not found: {docs_path}")
    docs, chunks = store.ingest_path(docs_path, reset=request.reset)
    return IngestResponse(ingested_documents=docs, chunks=chunks, vector_backend=store.backend)


@app.post("/chat", response_model=ChatResponse, dependencies=[Depends(require_api_key)])
async def chat(request: ChatRequest) -> ChatResponse:
    start = time.perf_counter()
    try:
        state = await agent.run(request.conversation_id, request.message, request.user_profile)
    except Exception as exc:
        logger.exception("chat_failed", error=str(exc))
        raise HTTPException(status_code=500, detail="Assistant failed to process the request") from exc

    latency_ms = round((time.perf_counter() - start) * 1000, 2)
    tool_successes = sum(1 for call in state.get("tool_calls", []) if call.success)
    token_usage = state.get("token_usage", {})
    metrics.record_chat(latency_ms, token_usage, tool_successes)
    return ChatResponse(
        conversation_id=request.conversation_id,
        answer=state.get("answer", ""),
        intent=state.get("intent", "unknown"),
        trace_id=state.get("trace_id"),
        cache_hit=state.get("cache_hit", False),
        citations=state.get("citations", []),
        tool_calls=state.get("tool_calls", []),
        guardrail_flags=state.get("guardrail_flags", []),
        trajectory=state.get("trajectory", []),
        critic_review=state.get("critic_review", {}),
        latency_ms=latency_ms,
        token_usage=token_usage,
    )


@app.get("/metrics")
async def get_metrics() -> dict[str, object]:
    return {**metrics.as_dict(), "optimization": cost_tracker.as_dict()}


@app.get("/audit", dependencies=[Depends(require_api_key)])
async def get_audit_events(limit: int = 50) -> dict[str, object]:
    return {"events": audit_logger.recent(limit=limit), "count": len(audit_logger.recent(limit=limit))}


@app.get("/approvals", dependencies=[Depends(require_api_key)])
async def list_approvals(status: str | None = None) -> dict[str, object]:
    approvals = approval_queue.list(status=status)
    return {"approvals": approvals, "count": len(approvals)}


@app.post("/approvals/{approval_id}/decision", dependencies=[Depends(require_api_key)])
async def decide_approval(approval_id: str, approved: bool) -> dict[str, object]:
    decision = approval_queue.decide(approval_id=approval_id, approved=approved)
    if decision.get("found") and approved:
        payload = decision.get("payload", {})
        action_type = "CREATE_FRAUD_TICKET" if decision.get("intent") == "support" else "SCHEDULE_BRANCH_APPOINTMENT"
        action = action_engine.create(
            action_type=action_type,
            customer_id=str(decision.get("requested_by", "anonymous")),
            payload=payload if isinstance(payload, dict) else {},
        )
        return {**decision, "resumed_action": action}
    return decision


@app.get("/traces", dependencies=[Depends(require_api_key)])
async def get_traces(limit: int = 100) -> dict[str, object]:
    traces = trace_recorder.recent(limit=limit)
    return {"traces": traces, "count": len(traces)}


@app.get("/privacy-vault/stats", dependencies=[Depends(require_api_key)])
async def get_privacy_vault_stats() -> dict[str, object]:
    return privacy_vault.stats()


@app.get("/insights", dependencies=[Depends(require_api_key)])
async def get_insights(customer_id: str) -> dict[str, object]:
    user_profile = {"customer_id": customer_id}
    insights = insight_engine.generate(user_profile)
    actions = action_engine.recommend(insights)
    return {"customer_id": customer_id, "insights": insights, "recommended_actions": actions, "count": len(insights)}


@app.post("/actions", dependencies=[Depends(require_api_key)])
async def create_action(request: ActionRequest) -> dict[str, object]:
    return action_engine.create(request.action_type, request.customer_id, request.payload)


@app.get("/actions", dependencies=[Depends(require_api_key)])
async def list_actions(customer_id: str | None = None) -> dict[str, object]:
    actions = action_engine.list(customer_id=customer_id)
    return {"actions": actions, "count": len(actions)}


@app.post("/actions/{action_id}/transition", dependencies=[Depends(require_api_key)])
async def transition_action(action_id: str, status: str) -> dict[str, object]:
    return action_engine.transition(action_id=action_id, status=status)


@app.post("/feedback", dependencies=[Depends(require_api_key)])
async def submit_feedback(request: FeedbackRequest) -> dict[str, object]:
    return feedback_store.add(
        conversation_id=request.conversation_id,
        rating=request.rating,
        category=request.category,
        comment=request.comment,
        metadata=request.metadata,
    )


@app.get("/feedback", dependencies=[Depends(require_api_key)])
async def get_feedback(limit: int = 100) -> dict[str, object]:
    events = feedback_store.recent(limit=limit)
    return {"feedback": events, "count": len(events), "content_gaps": feedback_store.content_gaps()}


@app.get("/memory/{conversation_id}", dependencies=[Depends(require_api_key)])
async def get_memory(conversation_id: str) -> dict[str, object]:
    return memory.context(conversation_id)
