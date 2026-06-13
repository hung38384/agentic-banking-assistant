from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class ModelProvider(Protocol):
    async def generate(
        self,
        question: str,
        context: str,
        tool_outputs: list[dict[str, Any]],
        guardrail_flags: list[str],
    ) -> tuple[str, dict[str, int]]:
        ...


@dataclass(frozen=True)
class ProviderResult:
    answer: str
    token_usage: dict[str, int]
    provider: str
    fallback_used: bool = False


class ModelGateway:
    """Provider abstraction for Gemini/OpenAI/local fallback.

    This keeps the agent independent from any single vendor and gives one
    place to add provider health checks, budgets, and routing rules.
    """

    def __init__(self, primary: ModelProvider, provider_name: str):
        self.primary = primary
        self.provider_name = provider_name

    async def generate(
        self,
        question: str,
        context: str,
        tool_outputs: list[dict[str, Any]],
        guardrail_flags: list[str],
    ) -> ProviderResult:
        answer, usage = await self.primary.generate(question, context, tool_outputs, guardrail_flags)
        fallback_used = "local fallback generator" in answer.lower()
        return ProviderResult(
            answer=answer,
            token_usage=usage,
            provider=self.provider_name,
            fallback_used=fallback_used,
        )

