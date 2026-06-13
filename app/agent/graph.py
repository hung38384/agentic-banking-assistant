from __future__ import annotations

import re
import unicodedata
from typing import Any
from uuid import uuid4

from langgraph.graph import END, StateGraph

from app.agent.state import AgentState
from app.core.config import Settings
from app.schemas import (
    AgentPlanSchema,
    AnswerValidationSchema,
    Citation,
    CriticReviewSchema,
    ToolCallRecord,
    ToolRoutingDecisionSchema,
)
from app.services.access_control import SENSITIVE_INTENTS, get_role, has_consent, is_allowed
from app.services.hitl import approval_queue
from app.services.llm import LLMClient
from app.services.memory import ConversationMemory
from app.services.model_gateway import ModelGateway
from app.services.observability import trace_recorder
from app.services.policy import policy_engine, score_tool_risk
from app.services.privacy import redact_text
from app.services.privacy_vault import privacy_vault
from app.services.query_processing import rewrite_and_decompose_query
from app.services.retrieval import compress_context
from app.services.semantic_cache import semantic_cache
from app.services.cost_tracker import cost_tracker
from app.services.token_optimization import token_budget_manager
from app.services.vector_store import BankingVectorStore
from app.tools.banking_tools import (
    branch_finder,
    create_support_ticket,
    customer_account_summary,
    eligibility_checker,
    fee_lookup,
    loan_status_lookup,
    loan_calculator,
    transaction_search,
)


class BankingAgent:
    def __init__(self, settings: Settings, store: BankingVectorStore, memory: ConversationMemory):
        self.settings = settings
        self.store = store
        self.memory = memory
        self.llm = LLMClient(settings)
        self.model_gateway = ModelGateway(self.llm, settings.llm_provider)
        self.graph = self._build_graph()

    async def run(self, conversation_id: str, question: str, user_profile: dict[str, Any]) -> AgentState:
        self.memory.set_consent(conversation_id, user_profile)
        self.memory.update_preferences(conversation_id, question)
        initial: AgentState = {
            "trace_id": f"trace-{uuid4().hex[:12]}",
            "conversation_id": conversation_id,
            "original_question": question,
            "question": question,
            "user_profile": user_profile,
            "preferences": self.memory.preferences(conversation_id),
            "memory_context": self.memory.context(conversation_id),
            "slots": self.memory.slots(conversation_id),
            "tool_calls": [],
            "guardrail_flags": [],
            "hitl_requests": [],
            "trajectory": [],
            "retry_count": 0,
        }
        state = await self.graph.ainvoke(initial)
        self.memory.add(conversation_id, "user", question)
        self.memory.add(conversation_id, "assistant", state.get("answer", ""))
        return state

    def _build_graph(self):
        graph = StateGraph(AgentState)
        graph.add_node("understand_query", self._understand_query)
        graph.add_node("apply_input_policy", self._apply_input_policy)
        graph.add_node("plan_and_rewrite", self._plan_and_rewrite)
        graph.add_node("classify_intent", self._classify_intent)
        graph.add_node("decide_retrieval", self._decide_retrieval)
        graph.add_node("retrieve_context", self._retrieve_context)
        graph.add_node("decide_tools", self._decide_tools)
        graph.add_node("route_tool", self._route_tool)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_node("generate_answer", self._generate_answer)
        graph.add_node("validate_response", self._validate_response)
        graph.add_node("critic_verification", self._critic_verification)
        graph.add_node("retry_retrieval", self._retry_retrieval)

        graph.set_entry_point("understand_query")
        graph.add_edge("understand_query", "apply_input_policy")
        graph.add_conditional_edges(
            "apply_input_policy",
            lambda state: "validate_response" if "input_policy_blocked" in state.get("guardrail_flags", []) else "plan_and_rewrite",
            {"validate_response": "validate_response", "plan_and_rewrite": "plan_and_rewrite"},
        )
        graph.add_conditional_edges(
            "plan_and_rewrite",
            lambda state: "validate_response" if state.get("cache_hit") else "classify_intent",
            {"validate_response": "validate_response", "classify_intent": "classify_intent"},
        )
        graph.add_edge("classify_intent", "decide_retrieval")
        graph.add_conditional_edges(
            "decide_retrieval",
            lambda state: "retrieve_context" if state.get("needs_retrieval") else "decide_tools",
            {"retrieve_context": "retrieve_context", "decide_tools": "decide_tools"},
        )
        graph.add_edge("retrieve_context", "decide_tools")
        graph.add_conditional_edges(
            "decide_tools",
            lambda state: "route_tool" if state.get("needs_tools") else "generate_answer",
            {"route_tool": "route_tool", "generate_answer": "generate_answer"},
        )
        graph.add_edge("route_tool", "execute_tools")
        graph.add_edge("execute_tools", "generate_answer")
        graph.add_edge("generate_answer", "validate_response")
        graph.add_edge("validate_response", "critic_verification")
        graph.add_conditional_edges(
            "critic_verification",
            lambda state: "retry_retrieval" if state.get("critic_review", {}).get("needs_retry") else END,
            {"retry_retrieval": "retry_retrieval", END: END},
        )
        graph.add_edge("retry_retrieval", "retrieve_context")
        return graph.compile()

    def _record_step(
        self,
        state: AgentState,
        node: str,
        decision: str,
        details: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        return [
            *state.get("trajectory", []),
            {
                "node": node,
                "decision": decision,
                "details": details or {},
            },
        ]

    async def _understand_query(self, state: AgentState) -> AgentState:
        with trace_recorder.span(state["trace_id"], "understand_query"):
            tokenized_question, vault_tokens = privacy_vault.tokenize(
                state["question"].strip(),
                scope=state["trace_id"],
            )
            redaction = redact_text(tokenized_question)
        question = redaction.redacted
        flags = state.get("guardrail_flags", [])
        pii_types = sorted(set(redaction.pii_types + [token["pii_type"] for token in vault_tokens]))
        if pii_types:
            flags.append("pii_redacted")
        advice_terms = [
            "should i invest",
            "best card for me",
            "recommend a loan",
            "recommend the best loan",
            "guaranteed approval",
            "guarantee approval",
            "should i take",
        ]
        lowered_question = question.lower()
        if any(term in lowered_question for term in advice_terms) or (
            "guarantee" in lowered_question and any(term in lowered_question for term in ["approval", "approved"])
        ):
            flags.append("unsupported_financial_advice")
        if _looks_like_unsupported_rate_request(lowered_question):
            flags.append("unsupported_rate_request")
        return {
            **state,
            "question": question,
            "guardrail_flags": flags,
            "pii_types": pii_types,
            "vault_tokens": vault_tokens,
            "trajectory": self._record_step(
                state,
                "understand_query",
                "tokenized_and_redacted",
                {"pii_types": pii_types, "vault_token_count": len(vault_tokens)},
            ),
        }

    async def _apply_input_policy(self, state: AgentState) -> AgentState:
        with trace_recorder.span(state["trace_id"], "apply_input_policy"):
            decision = policy_engine.evaluate_input(state["question"])
        flags = sorted(set(state.get("guardrail_flags", []) + decision.flags))
        if not decision.allowed:
            flags.append("input_policy_blocked")
            return {
                **state,
                "guardrail_flags": sorted(set(flags)),
                "answer": decision.reason,
                "token_usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "trajectory": self._record_step(
                    state,
                    "apply_input_policy",
                    "blocked",
                    {"flags": decision.flags, "severity": decision.severity},
                ),
            }
        return {
            **state,
            "guardrail_flags": flags,
            "trajectory": self._record_step(state, "apply_input_policy", "allowed", {"flags": decision.flags}),
        }

    async def _plan_and_rewrite(self, state: AgentState) -> AgentState:
        with trace_recorder.span(state["trace_id"], "plan_and_rewrite"):
            query_plan = rewrite_and_decompose_query(state["question"])
            inferred_intent = _classify_intent_text(state["question"], state.get("memory_context", {}))
            cache_entry = (
                semantic_cache.get(query_plan.rewritten_query)
                if self.settings.semantic_cache_enabled and _semantic_cache_allowed(inferred_intent, state.get("guardrail_flags", []))
                else None
            )
        if cache_entry:
            return {
                **state,
                "cache_hit": True,
                "answer": cache_entry["answer"],
                "intent": cache_entry.get("intent", "cached"),
                "citations": cache_entry.get("citations", []),
                "tool_calls": cache_entry.get("tool_calls", []),
                "guardrail_flags": sorted(set(state.get("guardrail_flags", []) + ["semantic_cache_hit"])),
                "token_usage": cache_entry.get("token_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}),
                "trajectory": self._record_step(state, "plan_and_rewrite", "cache_hit"),
            }
        required_tools = _tools_for_intent(inferred_intent)
        risk_level = "high" if inferred_intent in SENSITIVE_INTENTS | {"support"} else "medium" if inferred_intent == "eligibility" else "low"
        plan = AgentPlanSchema(
            rewritten_query=query_plan.rewritten_query,
            subqueries=query_plan.subqueries,
            retrieval_strategy="hybrid",
            required_tools=required_tools,
            clarification_questions=_clarifications_for_query(state["question"]),
            risk_level=risk_level,
            requires_human_approval=risk_level == "high" and inferred_intent == "support",
        )
        return {
            **state,
            "rewritten_query": query_plan.rewritten_query,
            "subqueries": query_plan.subqueries,
            "plan": plan.model_dump(),
            "trajectory": self._record_step(
                state,
                "plan_and_rewrite",
                "planned",
                {"plan": plan.model_dump(), "reasoning": query_plan.reasoning},
            ),
        }

    async def _classify_intent(self, state: AgentState) -> AgentState:
        intent = _classify_intent_text(state["question"], state.get("memory_context", {}))
        return {
            **state,
            "intent": intent,
            "trajectory": self._record_step(state, "classify_intent", intent),
        }

    async def _decide_retrieval(self, state: AgentState) -> AgentState:
        needs_retrieval = state.get("intent") in {"product_qa", "fee_lookup", "eligibility", "loan_calculation", "support", "branch_finder", "account_summary", "transaction_search", "loan_status"}
        token_budget = token_budget_manager.plan(
            state["question"],
            state.get("intent", "product_qa"),
            has_tools=state.get("intent") in {"loan_calculation", "fee_lookup", "eligibility", "support", "branch_finder", "account_summary", "transaction_search", "loan_status"},
        )
        return {
            **state,
            "needs_retrieval": needs_retrieval,
            "token_budget": token_budget.__dict__,
            "trajectory": self._record_step(
                state,
                "decide_retrieval",
                "retrieve" if needs_retrieval else "skip",
                {"token_budget": token_budget.__dict__},
            ),
        }

    async def _retrieve_context(self, state: AgentState) -> AgentState:
        query = state.get("rewritten_query", state["question"])
        subqueries = state.get("subqueries") or [query]
        if state.get("intent") in {"account_summary", "transaction_search", "loan_status"}:
            subqueries = [*subqueries, "digital banking core integration account transaction loan status consent audit"]
        chunks_by_id: dict[str, dict[str, Any]] = {}
        with trace_recorder.span(state["trace_id"], "hybrid_retrieval", subquery_count=len(subqueries)):
            for subquery in subqueries:
                for chunk in self.store.hybrid_search(subquery, top_k=state.get("token_budget", {}).get("top_k", self.settings.top_k)):
                    chunk_id = chunk["metadata"]["chunk_id"]
                    if chunk_id not in chunks_by_id or chunk.get("rerank_score", 0) > chunks_by_id[chunk_id].get("rerank_score", 0):
                        chunks_by_id[chunk_id] = chunk
        chunks = sorted(
            chunks_by_id.values(),
            key=lambda item: item.get("rerank_score", item.get("score", 0)),
            reverse=True,
        )[: state.get("token_budget", {}).get("top_k", self.settings.top_k)]
        if "unsupported_rate_request" in state.get("guardrail_flags", []):
            rate_chunks = [
                chunk
                for chunk in chunks
                if any(term in chunk.get("text", "").lower() for term in ["deposit rate", "interest", "must not invent deposit rates", "rate table"])
            ]
            chunks = (rate_chunks or chunks)[:1]
        context_parts: list[str] = []
        citations: list[Citation] = []
        char_budget = state.get("token_budget", {}).get("max_context_chars", self.settings.max_context_chars)
        raw_context_chars = sum(len(chunk["text"]) for chunk in chunks)
        for chunk in chunks:
            metadata = chunk["metadata"]
            text = compress_context(query, chunk["text"], max(400, char_budget // max(state.get("token_budget", {}).get("top_k", self.settings.top_k), 1)))
            if sum(len(part) for part in context_parts) + len(text) > char_budget:
                text = text[: max(0, char_budget - sum(len(part) for part in context_parts))]
            context_parts.append(f"[{metadata['title']}] {text}")
            citations.append(
                Citation(
                    source=metadata["source"],
                    title=metadata["title"],
                    chunk_id=metadata["chunk_id"],
                )
            )
            if sum(len(part) for part in context_parts) >= char_budget:
                break
        return {
            **state,
            "retrieved_chunks": chunks,
            "context": "\n\n".join(context_parts),
            "citations": citations,
            "trajectory": self._record_step(
                state,
                "retrieve_context",
                "retrieved",
                {
                    "chunk_count": len(chunks),
                    "sources": [citation.title for citation in citations],
                    "raw_context_chars": raw_context_chars,
                    "packed_context_chars": len("\n\n".join(context_parts)),
                },
            ),
            "optimization_metrics": {
                **state.get("optimization_metrics", {}),
                "raw_context_chars": raw_context_chars,
                "packed_context_chars": len("\n\n".join(context_parts)),
                "adaptive_top_k": state.get("token_budget", {}).get("top_k", self.settings.top_k),
            },
        }

    async def _decide_tools(self, state: AgentState) -> AgentState:
        needs_tools = state.get("intent") in {"loan_calculation", "fee_lookup", "eligibility", "support", "branch_finder", "account_summary", "transaction_search", "loan_status"}
        return {
            **state,
            "needs_tools": needs_tools,
            "trajectory": self._record_step(state, "decide_tools", "use_tool" if needs_tools else "skip"),
        }

    async def _route_tool(self, state: AgentState) -> AgentState:
        intent = state.get("intent", "unknown")
        selected_tool = (_tools_for_intent(intent) or [None])[0]
        risk = score_tool_risk(intent, selected_tool or "none")
        decision = ToolRoutingDecisionSchema(
            selected_tool=selected_tool,
            reason=f"Selected by controlled router for intent={intent}.",
            expected_arguments=_expected_arguments_for_intent(intent, state["question"], state.get("user_profile", {})),
            risk_level=risk["level"],
            allowed_by_scope=is_allowed(state.get("user_profile", {}), intent) if intent in SENSITIVE_INTENTS else True,
            requires_human_approval=risk["level"] == "high" and intent == "support",
        )
        return {
            **state,
            "tool_routing": decision.model_dump(),
            "trajectory": self._record_step(state, "route_tool", selected_tool or "none", decision.model_dump()),
        }

    async def _execute_tools(self, state: AgentState) -> AgentState:
        calls = state.get("tool_calls", [])
        q = state["question"]
        intent = state.get("intent")

        def record(name: str, args: dict[str, Any], output: dict[str, Any], success: bool = True) -> None:
            calls.append(ToolCallRecord(name=name, input=args, output=output, success=success))

        try:
            routed_tool = state.get("tool_routing", {}).get("selected_tool") or intent or "unknown"
            risk = score_tool_risk(intent or "unknown", routed_tool)
            state["guardrail_flags"] = sorted(set(state.get("guardrail_flags", []) + [f"risk_{risk['level']}"]))
            if intent in SENSITIVE_INTENTS and not is_allowed(state.get("user_profile", {}), intent):
                output = {
                    "allowed": False,
                    "role": get_role(state.get("user_profile", {})),
                    "reason": "This request requires an authenticated customer session.",
                }
                record("access_control", {"intent": intent}, output, success=False)
                return {
                    **state,
                    "tool_calls": calls,
                    "guardrail_flags": sorted(set(state.get("guardrail_flags", []) + ["authentication_required"])),
                    "trajectory": self._record_step(
                        state,
                        "execute_tools",
                        "blocked_or_failed",
                        {"tools": [call.name for call in calls], "reason": "authentication_required"},
                    ),
                }
            if intent in SENSITIVE_INTENTS and not has_consent(state.get("user_profile", {})):
                output = {
                    "allowed": False,
                    "human_review_required": True,
                    "reason": "Customer consent is required before accessing personal banking data.",
                }
                record("consent_check", {"intent": intent}, output, success=False)
                return {
                    **state,
                    "tool_calls": calls,
                    "guardrail_flags": sorted(set(state.get("guardrail_flags", []) + ["consent_required"])),
                    "trajectory": self._record_step(
                        state,
                        "execute_tools",
                        "blocked_or_failed",
                        {"tools": [call.name for call in calls], "reason": "consent_required"},
                    ),
                }
            tool_policy = policy_engine.evaluate_tool_call(intent or "unknown", routed_tool, state.get("user_profile", {}))
            if tool_policy.requires_human_approval and self.settings.hitl_enabled:
                approval = approval_queue.create(
                    intent=intent or "unknown",
                    tool_name=intent or "unknown",
                    reason=tool_policy.reason or "Human approval required by policy.",
                    requested_by=str(state.get("user_profile", {}).get("customer_id", "anonymous")),
                    payload={"question": state["question"], "intent": intent},
                )
                state["hitl_requests"] = [*state.get("hitl_requests", []), approval.__dict__]
                if not tool_policy.allowed:
                    record(
                        "human_approval_interrupt",
                        {"intent": intent},
                        {"approval_id": approval.approval_id, "status": approval.status, "reason": approval.reason},
                        success=False,
                    )
                    return {
                        **state,
                        "tool_calls": calls,
                        "guardrail_flags": sorted(set(state.get("guardrail_flags", []) + tool_policy.flags + ["human_approval_pending"])),
                        "trajectory": self._record_step(
                            state,
                            "execute_tools",
                            "blocked_or_failed",
                            {"tools": [call.name for call in calls], "reason": "human_approval_pending"},
                        ),
                    }
            if intent == "loan_calculation":
                loan_slots = {
                    **state.get("slots", {}),
                    **state.get("memory_context", {}).get("slots", {}),
                    **_drop_none(_extract_loan_slots(q)),
                }
                amount = loan_slots.get("loan_amount")
                rate = loan_slots.get("annual_rate")
                months = loan_slots.get("repayment_months")
                if amount is None or rate is None or months is None:
                    missing = [
                        field
                        for field, value in {
                            "loan_amount": amount,
                            "annual_rate": rate,
                            "repayment_months": months,
                        }.items()
                        if value is None
                    ]
                    self.memory.set_pending_task(
                        state["conversation_id"],
                        intent="loan_calculation",
                        missing_fields=missing,
                        collected_slots=loan_slots,
                        question=state["question"],
                    )
                    output = {
                        "missing_fields": missing,
                        "collected_slots": {key: value for key, value in loan_slots.items() if value is not None},
                    }
                    state = {
                        **state,
                        "guardrail_flags": sorted(set(state.get("guardrail_flags", []) + ["missing_loan_terms"])),
                    }
                else:
                    output = loan_calculator(float(amount), float(rate), int(months))
                    self.memory.update_slots(state["conversation_id"], loan_slots)
                    self.memory.clear_pending_task(state["conversation_id"])
                record("loan_calculator", {"amount": amount, "rate": rate, "months": months}, output)
            elif intent == "fee_lookup":
                product = _detect_product(q)
                service = _detect_service(q)
                output = fee_lookup(product, service)
                record("fee_lookup", {"product_type": product, "service_name": service}, output)
            elif intent == "eligibility":
                remembered_product = (
                    state.get("memory_context", {})
                    .get("slots", {})
                    .get("eligibility_product")
                )
                pending_product = (
                    state.get("memory_context", {})
                    .get("pending_task", {})
                    .get("collected_slots", {})
                    .get("eligibility_product")
                    if state.get("memory_context", {}).get("pending_task")
                    else None
                )
                preference_product = state.get("memory_context", {}).get("preferences", {}).get("product_interest")
                product = _detect_product(q, default=remembered_product or pending_product or preference_product or "savings_account")
                eligibility_slots = {
                    **state.get("memory_context", {}).get("slots", {}),
                    **_drop_none(_extract_eligibility_slots(q)),
                    **state.get("user_profile", {}),
                    "eligibility_product": product,
                }
                output = eligibility_checker(eligibility_slots, product)
                if output.get("missing_fields"):
                    self.memory.set_pending_task(
                        state["conversation_id"],
                        intent="eligibility",
                        missing_fields=output["missing_fields"],
                        collected_slots=eligibility_slots,
                        question=state["question"],
                    )
                else:
                    self.memory.update_slots(state["conversation_id"], eligibility_slots)
                    self.memory.clear_pending_task(state["conversation_id"])
                record("eligibility_checker", {"user_profile": eligibility_slots, "product": product}, output)
            elif intent == "support":
                output = create_support_ticket("general", q[:240], human_review_confirmed=False)
                record("create_support_ticket", {"issue_type": "general", "summary": q[:240]}, output)
            elif intent == "account_summary":
                output = customer_account_summary(state.get("user_profile", {}))
                record("customer_account_summary", {"customer_id": state.get("user_profile", {}).get("customer_id")}, output)
            elif intent == "transaction_search":
                keyword = _detect_transaction_keyword(q)
                output = transaction_search(state.get("user_profile", {}), keyword=keyword)
                record("transaction_search", {"keyword": keyword}, output)
            elif intent == "loan_status":
                output = loan_status_lookup(state.get("user_profile", {}))
                record("loan_status_lookup", {"customer_id": state.get("user_profile", {}).get("customer_id")}, output)
            elif intent == "branch_finder":
                province = _detect_province(q)
                service = _detect_branch_service(q)
                output = branch_finder(province=province, service=service)
                record("branch_finder", {"province": province, "service": service}, output)
        except Exception as exc:
            record(intent or "unknown", {}, {"error": str(exc)}, success=False)
        return {
            **state,
            "tool_calls": calls,
            "trajectory": self._record_step(
                state,
                "execute_tools",
                "executed" if calls and all(call.success for call in calls) else "blocked_or_failed" if calls else "no_tool",
                {"tools": [call.name for call in calls]},
            ),
        }

    async def _generate_answer(self, state: AgentState) -> AgentState:
        if state.get("cache_hit") or state.get("answer") and "input_policy_blocked" in state.get("guardrail_flags", []):
            return {
                **state,
                "trajectory": self._record_step(state, "generate_answer", "skipped"),
            }
        tool_outputs = token_budget_manager.compact_tool_outputs(
            [call.model_dump() for call in state.get("tool_calls", [])]
        )
        original_tool_tokens = token_budget_manager.estimate_tokens([call.model_dump() for call in state.get("tool_calls", [])])
        compact_tool_tokens = token_budget_manager.estimate_tokens(tool_outputs)
        with trace_recorder.span(state["trace_id"], "model_gateway_generate", provider=self.settings.llm_provider):
            result = await self.model_gateway.generate(
                state["question"],
                state.get("context", ""),
                tool_outputs,
                state.get("guardrail_flags", []),
            )
        flags = state.get("guardrail_flags", [])
        if result.fallback_used:
            flags.append("model_gateway_fallback")
        cost_tracker.record(
            provider=self.settings.llm_provider,
            prompt_tokens=result.token_usage.get("prompt_tokens", 0),
            completion_tokens=result.token_usage.get("completion_tokens", 0),
            cache_hit=state.get("cache_hit", False),
            context_chars_before=state.get("optimization_metrics", {}).get("raw_context_chars", 0),
            context_chars_after=state.get("optimization_metrics", {}).get("packed_context_chars", 0),
        )
        return {
            **state,
            "answer": result.answer,
            "token_usage": result.token_usage,
            "guardrail_flags": sorted(set(flags)),
            "optimization_metrics": {
                **state.get("optimization_metrics", {}),
                "original_tool_tokens": original_tool_tokens,
                "compact_tool_tokens": compact_tool_tokens,
                "estimated_prompt_tokens": result.token_usage.get("prompt_tokens", 0),
            },
            "trajectory": self._record_step(
                state,
                "generate_answer",
                "generated",
                {
                    "provider": self.settings.llm_provider,
                    "fallback": result.fallback_used,
                    "tool_tokens_before": original_tool_tokens,
                    "tool_tokens_after": compact_tool_tokens,
                },
            ),
        }

    async def _validate_response(self, state: AgentState) -> AgentState:
        flags = state.get("guardrail_flags", [])
        if state.get("needs_retrieval") and not state.get("retrieved_chunks"):
            flags.append("no_retrieval_context")
            state["answer"] = (
                "I could not find this information in the available banking documents. "
                "Please provide more detail or ask for human support."
            )
        if "unsupported_rate_request" in flags:
            state["answer"] = (
                "I could not find a current published deposit interest rate table in the available "
                "banking documents, so I must not invent a rate. Please check the bank's current "
                "rate table or ask a human banker."
            )
        if "unsupported_financial_advice" in flags and "cannot provide personalized financial advice" not in state.get("answer", "").lower():
            state["answer"] = (
                "I cannot provide personalized financial advice, guarantee approval, or recommend a "
                "product as the best choice. I can explain documented product features, fees, "
                "eligibility rules, and calculation assumptions for human review."
            )
        if state.get("tool_calls"):
            for call in state["tool_calls"]:
                if call.output.get("data_quality_flags"):
                    flags.append("data_quality_caveat")
                if call.name == "loan_calculator" and call.output.get("missing_fields"):
                    flags.append("clarification_required")
                    state["answer"] = _clarification_message(
                        "loan_calculation",
                        call.output.get("missing_fields", []),
                        call.output.get("collected_slots", {}),
                    )
                if call.name == "eligibility_checker" and call.output.get("missing_fields"):
                    flags.append("clarification_required")
                    state["answer"] = _clarification_message(
                        "eligibility",
                        call.output.get("missing_fields", []),
                        call.output.get("rules", {}),
                    )
                if call.output.get("allowed") is False:
                    state["answer"] = call.output.get("reason", "This action is not allowed for the current session.")
                if call.name == "human_approval_interrupt":
                    state["answer"] = (
                        "This request is waiting for human approval before the assistant can continue. "
                        f"Approval id: {call.output.get('approval_id')}"
                    )
        if "data_quality_caveat" in flags and state.get("answer"):
            state["answer"] += (
                " Note: some source data contains quality flags such as missing, pending, duplicate, "
                "stale, dormant, frozen, or delinquent records; please verify with a human banker before acting."
            )
        validation = AnswerValidationSchema(
            valid=bool(state.get("answer")),
            grounded=bool(state.get("citations") or state.get("tool_calls") or flags),
            flags=sorted(set(flags)),
            confidence=0.85 if state.get("citations") or state.get("tool_calls") else 0.45,
        )
        flags = sorted(set(flags + validation.flags + ([] if validation.grounded else ["ungrounded_answer"])))
        updated = {
            **state,
            "guardrail_flags": flags,
            "trajectory": self._record_step(
                state,
                "validate_response",
                "valid" if validation.valid and validation.grounded else "invalid",
                validation.model_dump(),
            ),
        }
        if (
            self.settings.semantic_cache_enabled
            and not state.get("cache_hit")
            and validation.valid
            and _semantic_cache_allowed(updated.get("intent", "unknown"), flags)
        ):
            semantic_cache.set(
                state.get("rewritten_query", state["question"]),
                {
                    "answer": updated.get("answer", ""),
                    "intent": updated.get("intent", "unknown"),
                    "citations": updated.get("citations", []),
                    "tool_calls": updated.get("tool_calls", []),
                    "token_usage": updated.get("token_usage", {}),
                },
            )
        return updated

    async def _critic_verification(self, state: AgentState) -> AgentState:
        expected_tool = (_tools_for_intent(state.get("intent", "unknown")) or [None])[0]
        actual_tools = [call.name for call in state.get("tool_calls", [])]
        policy_compliant = not any(flag in state.get("guardrail_flags", []) for flag in ["ungrounded_answer"])
        grounded = bool(state.get("citations") or state.get("tool_calls") or "input_policy_blocked" in state.get("guardrail_flags", []))
        tool_use_correct = expected_tool is None or expected_tool in actual_tools or any(
            tool in actual_tools for tool in ["access_control", "consent_check", "human_approval_interrupt"]
        )
        needs_retry = (
            not grounded
            and state.get("needs_retrieval", False)
            and state.get("retry_count", 0) < 1
            and "input_policy_blocked" not in state.get("guardrail_flags", [])
        )
        review = CriticReviewSchema(
            passed=grounded and policy_compliant and tool_use_correct,
            grounded=grounded,
            tool_use_correct=tool_use_correct,
            policy_compliant=policy_compliant,
            needs_retry=needs_retry,
            retry_reason="No grounded evidence found; retrying with broadened query." if needs_retry else None,
            confidence=0.9 if grounded and tool_use_correct else 0.55,
        )
        return {
            **state,
            "critic_review": review.model_dump(),
            "trajectory": self._record_step(
                state,
                "critic_verification",
                "retry" if review.needs_retry else "passed" if review.passed else "accepted_with_flags",
                review.model_dump(),
            ),
        }

    async def _retry_retrieval(self, state: AgentState) -> AgentState:
        retry_count = state.get("retry_count", 0) + 1
        broadened = f"{state.get('rewritten_query', state['question'])} banking product policy fee eligibility service support"
        return {
            **state,
            "retry_count": retry_count,
            "rewritten_query": broadened,
            "subqueries": [broadened],
            "trajectory": self._record_step(
                state,
                "retry_retrieval",
                "broaden_query",
                {"retry_count": retry_count, "query": broadened},
            ),
        }


def _semantic_cache_allowed(intent: str, flags: list[str]) -> bool:
    if intent in {"account_summary", "transaction_search", "loan_status", "support", "loan_calculation"}:
        return False
    blocked_flags = {
        "authentication_required",
        "consent_required",
        "human_approval_pending",
        "human_approval_required",
        "data_quality_caveat",
        "input_policy_blocked",
        "prompt_injection_detected",
        "sql_injection_detected",
    }
    return not (set(flags) & blocked_flags)


def _extract_loan_terms(text: str) -> tuple[float | None, float | None, int | None]:
    slots = _extract_loan_slots(text)
    return slots.get("loan_amount"), slots.get("annual_rate"), slots.get("repayment_months")


def _extract_loan_slots(text: str) -> dict[str, float | int | None]:
    cleaned = text.replace(",", "").lower()
    numbers = [float(value) for value in re.findall(r"\d+(?:\.\d+)?", cleaned)]
    amount = None
    rate = None
    months = None
    amount_match = re.search(
        r"(?:loan|amount|khoan vay|vay|so tien)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)|"
        r"(\d+(?:\.\d+)?)\s*(?:vnd|đ|dong|thb|baht)",
        _normalize_vi_text(cleaned),
    )
    if amount_match:
        amount = float(next(group for group in amount_match.groups() if group is not None))
    rate_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:%|percent|annual rate|rate|lai suat)",
        _normalize_vi_text(cleaned),
    )
    if rate_match:
        rate = float(rate_match.group(1))
    month_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:months?|mo|tenor|period|thang|ky han)",
        _normalize_vi_text(cleaned),
    )
    if month_match:
        months = int(float(month_match.group(1)))
    if len(numbers) >= 3:
        amount = amount or max(numbers)
        months_candidates = [value for value in numbers if value <= 360 and value != amount]
        rate_candidates = [value for value in numbers if value <= 100 and value != amount]
        months = months or (int(months_candidates[-1]) if months_candidates else None)
        rate = rate or (rate_candidates[0] if rate_candidates else None)
    elif len(numbers) == 1:
        value = numbers[0]
        normalized = _normalize_vi_text(cleaned)
        if any(term in normalized for term in ["month", "tenor", "period", "thang", "ky han"]):
            months = int(value)
        elif any(term in normalized for term in ["%", "percent", "rate", "lai suat"]):
            rate = value
        elif any(term in normalized for term in ["vnd", "dong", "thb", "baht", "loan", "amount", "khoan vay", "vay", "so tien"]):
            amount = value
    return {"loan_amount": amount, "annual_rate": rate, "repayment_months": months}


def _drop_none(values: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in values.items() if value is not None}


def _extract_eligibility_slots(text: str) -> dict[str, Any]:
    lowered = text.lower().replace(",", "")
    normalized = _normalize_vi_text(lowered)
    slots: dict[str, Any] = {}
    age_match = re.search(r"(?:age|i am|i'm|toi|tuoi)\s*(\d{2})|(\d{2})\s*tuoi", normalized)
    if age_match:
        slots["age"] = int(next(group for group in age_match.groups() if group is not None))
    income_match = re.search(
        r"(?:income|salary|monthly income|thu nhap|luong)\s*(?:is|=|:)?\s*(\d+(?:\.\d+)?)",
        normalized,
    )
    if income_match:
        slots["monthly_income"] = float(income_match.group(1))
    employment_match = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:months?|mos?|thang)\s*(?:employment|employed|work|working|lam viec|di lam|cong tac)",
        normalized,
    )
    if employment_match:
        slots["employment_months"] = int(float(employment_match.group(1)))
    return slots


def _normalize_vi_text(text: str) -> str:
    normalized = unicodedata.normalize("NFD", text.lower())
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn").replace("đ", "d")


def _classify_intent_text(text: str, memory_context: dict[str, Any] | None = None) -> str:
    q = text.lower()
    nq = _normalize_vi_text(q)
    pending_task = (memory_context or {}).get("pending_task")
    if pending_task and _looks_like_follow_up(q):
        return pending_task.get("intent", "product_qa")
    if (memory_context or {}).get("preferences", {}).get("product_interest") and any(
        term in nq for term in ["thu nhap", "luong", "income", "salary", "tuoi", "age", "trieu", "employment"]
    ):
        return "eligibility"
    if any(term in nq for term in ["opening balance", "minimum balance", "minimum opening balance", "so du mo tai khoan", "so du toi thieu"]):
        return "product_qa"
    if any(term in nq for term in ["interest calculated", "lai tien gui duoc tinh", "tinh lai tien gui"]):
        return "product_qa"
    if any(term in nq for term in ["reward points", "gambling transactions", "refunded transactions", "diem thuong"]):
        return "product_qa"
    if any(term in nq for term in ["third-party fees", "excluded from repayment estimates", "phi ben thu ba"]):
        return "product_qa"
    if any(term in nq for term in ["what account situations require human review", "which credit card actions need human review", "what sensitive home loan actions require human banker review", "what personal loan actions require human review"]):
        return "product_qa"
    if any(term in nq for term in ["spending amount waives", "annual fee waiver", "mien phi thuong nien", "dieu kien mien phi thuong nien"]):
        return "fee_lookup"
    if any(term in nq for term in ["stable income history", "employment months", "months of employment", "minimum applicant age"]):
        return "eligibility"
    if any(term in nq for term in ["tranh chap giao dich", "disputed transaction", "dispute transaction", "chargeback"]):
        return "support"
    if any(term in nq for term in ["khieu nai", "gap nhan vien", "ho tro nguoi that", "gian lan", "nghi ngo lua dao"]):
        return "support"
    if any(term in nq for term in ["so du", "tai khoan cua toi", "thong tin tai khoan", "so du kha dung"]):
        return "account_summary"
    if any(term in nq for term in ["giao dich", "sao ke", "chi tieu", "lich su rut tien", "chuyen khoan gan day"]):
        return "transaction_search"
    if any(term in nq for term in ["trang thai khoan vay", "du no goc", "ngay den han", "qua han"]):
        return "loan_status"
    if any(term in nq for term in ["chi nhanh", "phong giao dich", "cay atm", "dat lich", "xep hang"]):
        return "branch_finder"
    if any(term in nq for term in ["du dieu kien", "dieu kien", "thu nhap toi thieu", "bao nhieu tuoi", "do tuoi"]):
        return "eligibility"
    if _looks_like_loan_calculation(nq):
        return "loan_calculation"
    if any(term in nq for term in ["phi", "bieu phi", "phi thuong nien", "phi tra cham", "ung tien mat", "phi tham dinh", "tra no truoc han", "rut tien atm"]):
        return "fee_lookup"
    if any(term in nq for term in ["tinh", "tra gop", "khoan phai tra hang thang", "tien tra hang thang", "lich tra no"]):
        return "loan_calculation"
    if any(term in q for term in ["ticket", "complaint", "human", "hardship", "suspected fraud", "khiếu nại", "gặp nhân viên", "hỗ trợ người thật", "gian lận", "nghi ngờ lừa đảo"]):
        return "support"
    if any(term in q for term in ["balance", "account summary", "available balance", "ledger balance", "show my account", "my account summary", "số dư", "tài khoản của tôi", "thông tin tài khoản", "số dư khả dụng"]):
        return "account_summary"
    if any(term in q for term in ["transaction", "statement", "spending", "recent transfer", "atm withdrawal history", "giao dịch", "sao kê", "chi tiêu", "lịch sử rút tiền", "chuyển khoản gần đây"]):
        return "transaction_search"
    if any(term in q for term in ["loan status", "outstanding principal", "next due", "days past due", "dpd", "trạng thái khoản vay", "dư nợ gốc", "ngày đến hạn", "quá hạn"]):
        return "loan_status"
    if any(term in q for term in ["branch", "atm near", "service center", "queue", "chi nhánh", "phòng giao dịch", "cây atm", "đặt lịch", "xếp hàng"]):
        return "branch_finder"
    if any(term in q for term in ["eligible", "eligibility", "qualify", "minimum income", "monthly income", "what age", "age is required", "đủ điều kiện", "điều kiện", "thu nhập tối thiểu", "bao nhiêu tuổi", "độ tuổi"]):
        return "eligibility"
    if any(term in q for term in ["fee", "charge", "atm withdrawal", "annual fee", "late payment", "cash advance", "appraisal", "early repayment", "phí", "biểu phí", "phí thường niên", "phí trả chậm", "ứng tiền mặt", "phí thẩm định", "trả nợ trước hạn", "rút tiền atm"]):
        return "fee_lookup"
    if any(term in q for term in ["repayment", "emi", "calculate", "monthly payment", "tính", "trả góp", "khoản phải trả hàng tháng", "tiền trả hàng tháng", "lịch trả nợ"]):
        return "loan_calculation"
    return "product_qa"


def _looks_like_loan_calculation(normalized_text: str) -> bool:
    if any(term in normalized_text for term in ["fee", "charge", "waive", "waived", "waiver", "phi", "mien phi"]):
        return False
    has_loan_term = any(term in normalized_text for term in ["loan", "vay", "khoan vay"])
    has_payment_term = any(
        term in normalized_text
        for term in [
            "repayment",
            "emi",
            "monthly payment",
            "calculate",
            "tinh",
            "tra gop",
            "tra bao nhieu",
            "moi thang",
            "hang thang",
            "lich tra no",
        ]
    )
    has_loan_numbers = bool(re.search(r"\d", normalized_text)) and any(
        term in normalized_text for term in ["%", "thang", "month", "months", "vnd", "dong"]
    )
    return has_loan_term and (has_payment_term or has_loan_numbers)


def _looks_like_follow_up(text: str) -> bool:
    normalized = _normalize_vi_text(text)
    if len(text.split()) <= 6 and re.search(r"\d", text):
        return True
    return any(
        term in normalized
        for term in ["yes", "no", "that one", "same", "for that", "months", "income", "salary", "age", "thang", "thu nhap", "luong", "tuoi"]
    )


def _looks_like_unsupported_rate_request(text: str) -> bool:
    normalized = _normalize_vi_text(text)
    asks_deposit_rate = any(term in normalized for term in ["deposit interest rate", "savings interest rate", "current rate", "lai suat tiet kiem", "lai suat tien gui"])
    references_account = any(term in normalized for term in ["savings", "deposit", "account", "tiet kiem", "tien gui", "tai khoan"])
    false_numeric_premise = bool(re.search(r"\d+(?:\.\d+)?\s*%", normalized)) and references_account and "loan" not in normalized and "vay" not in normalized
    return (asks_deposit_rate and references_account) or false_numeric_premise


def _tools_for_intent(intent: str) -> list[str]:
    return {
        "loan_calculation": ["loan_calculator"],
        "fee_lookup": ["fee_lookup"],
        "eligibility": ["eligibility_checker"],
        "support": ["create_support_ticket"],
        "account_summary": ["customer_account_summary"],
        "transaction_search": ["transaction_search"],
        "loan_status": ["loan_status_lookup"],
        "branch_finder": ["branch_finder"],
    }.get(intent, [])


def _clarifications_for_query(text: str) -> list[str]:
    intent = _classify_intent_text(text)
    if intent == "loan_calculation" and None in _extract_loan_terms(text):
        return ["Please provide loan amount, annual interest rate, and repayment months."]
    if intent in {"account_summary", "transaction_search", "loan_status"}:
        return ["Please confirm authenticated customer session and consent before personal-data access."]
    return []


def _clarification_message(intent: str, missing_fields: list[str], known: dict[str, Any]) -> str:
    if intent == "loan_calculation":
        labels = {
            "loan_amount": "loan amount",
            "annual_rate": "annual interest rate",
            "repayment_months": "repayment period in months",
            "amount": "loan amount",
            "months": "repayment period in months",
        }
        missing = ", ".join(labels.get(field, field) for field in missing_fields)
        known_bits = []
        if known.get("loan_amount") is not None:
            known_bits.append(f"amount {known['loan_amount']}")
        if known.get("annual_rate") is not None:
            known_bits.append(f"annual rate {known['annual_rate']}%")
        if known.get("repayment_months") is not None:
            known_bits.append(f"{known['repayment_months']} months")
        prefix = f"I have {', '.join(known_bits)}. " if known_bits else ""
        return f"{prefix}Please provide the missing {missing} so I can calculate repayment."
    if intent == "eligibility":
        return (
            "I can check preliminary eligibility, but I need "
            + ", ".join(missing_fields)
            + ". This remains a rule-based estimate and still requires human review."
        )
    return "Please provide the missing information so I can continue."


def _expected_arguments_for_intent(
    intent: str,
    question: str,
    user_profile: dict[str, Any],
) -> dict[str, Any]:
    if intent == "loan_calculation":
        amount, rate, months = _extract_loan_terms(question)
        return {"amount": amount, "rate": rate, "months": months}
    if intent == "fee_lookup":
        return {"product_type": _detect_product(question), "service_name": _detect_service(question)}
    if intent == "eligibility":
        return {"product": _detect_product(question), "user_profile_fields": sorted(user_profile.keys())}
    if intent in {"account_summary", "loan_status"}:
        return {"customer_id": user_profile.get("customer_id"), "consent_confirmed": user_profile.get("consent_confirmed", False)}
    if intent == "transaction_search":
        return {
            "customer_id": user_profile.get("customer_id"),
            "keyword": _detect_transaction_keyword(question),
            "consent_confirmed": user_profile.get("consent_confirmed", False),
        }
    if intent == "branch_finder":
        return {"province": _detect_province(question), "service": _detect_branch_service(question)}
    if intent == "support":
        return {"issue_type": "general", "human_review_required": True}
    return {}


def _detect_product(text: str, default: str = "savings_account") -> str:
    normalized = _normalize_vi_text(text)
    if "credit" in normalized or "card" in normalized or "the tin dung" in normalized:
        return "credit_card"
    if "home" in normalized or "mortgage" in normalized or "vay mua nha" in normalized or "vay nha" in normalized:
        return "home_loan"
    if "personal" in normalized or "vay tieu dung" in normalized or "vay ca nhan" in normalized:
        return "personal_loan"
    return default


def _detect_service(text: str) -> str:
    normalized = _normalize_vi_text(text)
    if "late" in normalized or "tra cham" in normalized:
        return "late_payment"
    if "cash advance" in normalized or "ung tien mat" in normalized:
        return "cash_advance"
    if "annual" in normalized or "thuong nien" in normalized:
        return "annual_fee"
    if "replacement" in normalized or "cap lai" in normalized or "thay the" in normalized:
        return "debit_card_replacement"
    if "atm" in normalized or "withdraw" in normalized or "rut tien" in normalized:
        return "atm_withdrawal_foreign_bank"
    if "appraisal" in normalized or "tham dinh" in normalized:
        return "appraisal"
    if "early repayment" in normalized or "tra no truoc han" in normalized:
        return "early_repayment"
    if "maintenance" in normalized or "duy tri" in normalized:
        return "monthly_maintenance"
    return "monthly_maintenance"


def _detect_transaction_keyword(text: str) -> str | None:
    text = _normalize_vi_text(text)
    if "atm" in text:
        return "atm"
    if "transfer" in text or "chuyen khoan" in text:
        return "transfer"
    if "card" in text or "spending" in text or "the tin dung" in text or "chi tieu" in text:
        return "card"
    if "supermarket" in text or "sieu thi" in text:
        return "supermarket"
    return None


def _detect_province(text: str) -> str | None:
    text = _normalize_vi_text(text)
    if "da nang" in text or "hai chau" in text:
        return "Da Nang"
    if "ho chi minh" in text or "hcm" in text or "sai gon" in text or "district 1" in text or "nguyen hue" in text:
        return "Ho Chi Minh City"
    if "hanoi" in text or "ha noi" in text or "cau giay" in text or "hoan kiem" in text:
        return "Hanoi"
    return None


def _detect_branch_service(text: str) -> str | None:
    text = _normalize_vi_text(text)
    if "loan" in text or "vay" in text:
        return "loan_consultation"
    if "card" in text or "the tin dung" in text:
        return "credit_card"
    if "cash" in text or "withdraw" in text or "tien mat" in text or "rut tien" in text:
        return "cash_service"
    if "open" in text or "account" in text or "mo tai khoan" in text:
        return "account_opening"
    return None
