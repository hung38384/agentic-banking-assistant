from __future__ import annotations

import asyncio
import re
from typing import Any

from openai import AsyncOpenAI

from app.core.config import Settings


class LLMClient:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = (
            AsyncOpenAI(api_key=settings.openai_api_key)
            if settings.llm_provider == "openai" and settings.openai_api_key
            else None
        )
        self._groq_client = (
            AsyncOpenAI(api_key=settings.groq_api_key, base_url=settings.groq_base_url)
            if settings.llm_provider == "groq" and settings.groq_api_key
            else None
        )
        self._gemini_client: Any | None = None
        if settings.llm_provider == "gemini" and settings.gemini_api_key:
            from google import genai

            self._gemini_client = genai.Client(api_key=settings.gemini_api_key)

    async def generate(
        self,
        question: str,
        context: str,
        tool_outputs: list[dict[str, Any]],
        guardrail_flags: list[str],
    ) -> tuple[str, dict[str, int]]:
        if self._client:
            prompt = self._build_prompt(question, context, tool_outputs, guardrail_flags)
            response = await self._client.chat.completions.create(
                model=self.settings.openai_model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a cautious banking assistant. Answer only from provided "
                            "context and tool outputs. Cite sources as [source: title]. "
                            "Refuse unsupported financial advice."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
            usage = response.usage
            return response.choices[0].message.content or "", {
                "prompt_tokens": usage.prompt_tokens if usage else 0,
                "completion_tokens": usage.completion_tokens if usage else 0,
                "total_tokens": usage.total_tokens if usage else 0,
            }
        if self._groq_client:
            return await self._generate_with_openai_compatible(
                client=self._groq_client,
                model=self.settings.groq_model,
                question=question,
                context=context,
                tool_outputs=tool_outputs,
                guardrail_flags=guardrail_flags,
                provider_name="Groq",
            )
        if self._gemini_client:
            return await self._generate_with_gemini(question, context, tool_outputs, guardrail_flags)
        return self._mock_generate(question, context, tool_outputs, guardrail_flags)

    async def _generate_with_openai_compatible(
        self,
        *,
        client: AsyncOpenAI,
        model: str,
        question: str,
        context: str,
        tool_outputs: list[dict[str, Any]],
        guardrail_flags: list[str],
        provider_name: str,
    ) -> tuple[str, dict[str, int]]:
        prompt = self._build_prompt(question, context, tool_outputs, guardrail_flags)
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a cautious banking assistant. Answer only from provided "
                            "context and tool outputs. Cite sources as [source: title]. "
                            "Refuse unsupported financial advice and never expose secrets or raw PII."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
            )
        except Exception:
            answer, tokens = self._mock_generate(
                question,
                context,
                tool_outputs,
                sorted(set(guardrail_flags + ["llm_provider_fallback"])),
            )
            return (
                answer
                + f" Note: live {provider_name} generation was unavailable, so this response used the local fallback generator.",
                tokens,
            )
        usage = response.usage
        return response.choices[0].message.content or "", {
            "prompt_tokens": usage.prompt_tokens if usage else 0,
            "completion_tokens": usage.completion_tokens if usage else 0,
            "total_tokens": usage.total_tokens if usage else 0,
        }

    async def _generate_with_gemini(
        self,
        question: str,
        context: str,
        tool_outputs: list[dict[str, Any]],
        guardrail_flags: list[str],
    ) -> tuple[str, dict[str, int]]:
        prompt = (
            "You are a cautious banking assistant. Answer only from the provided "
            "banking context and tool outputs. Include citations when context is used. "
            "Refuse unsupported personalized financial advice.\n\n"
            + self._build_prompt(question, context, tool_outputs, guardrail_flags)
        )

        def call_gemini():
            return self._gemini_client.models.generate_content(
                model=self.settings.gemini_model,
                contents=prompt,
                config={"temperature": 0.1},
            )

        try:
            response = await asyncio.to_thread(call_gemini)
        except Exception:
            answer, tokens = self._mock_generate(
                question,
                context,
                tool_outputs,
                sorted(set(guardrail_flags + ["llm_provider_fallback"])),
            )
            return (
                answer
                + " Note: live Gemini generation was unavailable, so this response used the local fallback generator.",
                tokens,
            )
        answer = getattr(response, "text", "") or ""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = getattr(usage, "prompt_token_count", 0) if usage else 0
        completion_tokens = getattr(usage, "candidates_token_count", 0) if usage else 0
        total_tokens = getattr(usage, "total_token_count", prompt_tokens + completion_tokens) if usage else 0
        return answer, {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }

    def _build_prompt(
        self,
        question: str,
        context: str,
        tool_outputs: list[dict[str, Any]],
        guardrail_flags: list[str],
    ) -> str:
        return (
            f"Question:\n{question}\n\nRetrieved context:\n{context}\n\n"
            f"Tool outputs:\n{tool_outputs}\n\nGuardrail flags:\n{guardrail_flags}\n"
            "Write a concise answer with citations and any needed caveats."
        )

    def _mock_generate(
        self,
        question: str,
        context: str,
        tool_outputs: list[dict[str, Any]],
        guardrail_flags: list[str],
    ) -> tuple[str, dict[str, int]]:
        if "unsupported_financial_advice" in guardrail_flags:
            answer = (
                "I cannot provide personalized financial advice or recommend a product as the "
                "best choice. I can explain documented features, fees, eligibility rules, and "
                "calculation assumptions so a human banker can review your situation."
            )
        elif not context and not tool_outputs:
            answer = (
                "I could not find supporting information in the banking documents. Please ask "
                "about a documented product, fee, eligibility rule, or calculation."
            )
        else:
            bullets: list[str] = []
            if context:
                first_lines = [line.strip("- ") for line in context.splitlines() if line.strip()]
                bullets.append(_most_relevant_context_line(question, first_lines)[:450])
            for output in tool_outputs:
                bullets.append(f"Tool result: {output}")
            answer = "Based on the available bank documents and tool results: " + " ".join(bullets)
        approx_prompt = max(1, (len(question) + len(context)) // 4)
        approx_completion = max(1, len(answer) // 4)
        return answer, {
            "prompt_tokens": approx_prompt,
            "completion_tokens": approx_completion,
            "total_tokens": approx_prompt + approx_completion,
        }


def _most_relevant_context_line(question: str, lines: list[str]) -> str:
    if not lines:
        return ""
    question_tokens = set(_content_tokens(question))
    candidates = [
        line
        for line in lines
        if not line.lstrip().startswith("#") and "] #" not in line and not line.startswith("[")
    ] or lines
    if not question_tokens:
        return candidates[0]
    lowered_question = question.lower()
    return max(candidates, key=lambda line: _line_relevance_score(lowered_question, question_tokens, line))


def _content_tokens(text: str) -> list[str]:
    stopwords = {"the", "is", "a", "an", "and", "or", "for", "to", "of", "in", "on", "when", "what", "how"}
    return [
        token
        for token in re.findall(r"[a-zA-Z][a-zA-Z0-9_%-]*|\d+(?:\.\d+)?%?", text.lower())
        if token not in stopwords and len(token) > 1
    ]


def _line_relevance_score(lowered_question: str, question_tokens: set[str], line: str) -> float:
    lowered_line = line.lower()
    score = len(question_tokens & set(_content_tokens(line))) / max(len(question_tokens), 1)
    for phrase in ["minimum opening balance", "opening balance", "annual fee", "early repayment", "deposit interest rate"]:
        if phrase in lowered_question and phrase in lowered_line:
            score += 1.0
    if any(term in lowered_question for term in ["minimum", "fee", "rate", "balance"]) and re.search(r"\d", line):
        score += 0.25
    return score
