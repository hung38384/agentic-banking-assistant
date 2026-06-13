from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    "mock": {"input": 0.0, "output": 0.0},
    "groq": {"input": 0.59, "output": 0.79},
    "gemini": {"input": 0.30, "output": 2.50},
    "openai": {"input": 0.15, "output": 0.60},
}


@dataclass
class CostTracker:
    counters: Counter = field(default_factory=Counter)

    def record(
        self,
        *,
        provider: str,
        prompt_tokens: int,
        completion_tokens: int,
        cache_hit: bool,
        context_chars_before: int = 0,
        context_chars_after: int = 0,
    ) -> None:
        pricing = MODEL_PRICING_PER_1M.get(provider, MODEL_PRICING_PER_1M["mock"])
        estimated_cost = (prompt_tokens / 1_000_000) * pricing["input"] + (completion_tokens / 1_000_000) * pricing["output"]
        self.counters["estimated_cost_usd_micros"] += int(estimated_cost * 1_000_000)
        self.counters["prompt_tokens"] += prompt_tokens
        self.counters["completion_tokens"] += completion_tokens
        self.counters["cache_hits"] += int(cache_hit)
        self.counters["requests"] += 1
        self.counters["context_chars_before"] += context_chars_before
        self.counters["context_chars_after"] += context_chars_after

    def as_dict(self) -> dict[str, float | int]:
        requests = max(self.counters["requests"], 1)
        before = self.counters["context_chars_before"]
        after = self.counters["context_chars_after"]
        return {
            "requests": self.counters["requests"],
            "cache_hit_rate": round(self.counters["cache_hits"] / requests, 4),
            "prompt_tokens": self.counters["prompt_tokens"],
            "completion_tokens": self.counters["completion_tokens"],
            "estimated_cost_usd": round(self.counters["estimated_cost_usd_micros"] / 1_000_000, 6),
            "avg_context_chars_before": round(before / requests, 2) if before else 0,
            "avg_context_chars_after": round(after / requests, 2) if after else 0,
            "compression_ratio": round(after / before, 4) if before else 0,
        }


cost_tracker = CostTracker()

