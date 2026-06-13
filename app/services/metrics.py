import time
from collections import Counter
from dataclasses import dataclass, field


@dataclass
class MetricsRegistry:
    started_at: float = field(default_factory=time.time)
    counters: Counter = field(default_factory=Counter)
    latencies_ms: list[float] = field(default_factory=list)
    token_usage: Counter = field(default_factory=Counter)

    def record_chat(self, latency_ms: float, tokens: dict[str, int], tool_successes: int) -> None:
        self.counters["chat_requests"] += 1
        self.counters["tool_successes"] += tool_successes
        self.latencies_ms.append(latency_ms)
        for key, value in tokens.items():
            self.token_usage[key] += value

    def as_dict(self) -> dict[str, float | int]:
        total = max(self.counters["chat_requests"], 1)
        return {
            "uptime_seconds": round(time.time() - self.started_at, 2),
            "chat_requests": self.counters["chat_requests"],
            "avg_latency_ms": round(sum(self.latencies_ms) / total, 2) if self.latencies_ms else 0,
            "tool_success_rate": round(self.counters["tool_successes"] / total, 4),
            "prompt_tokens": self.token_usage["prompt_tokens"],
            "completion_tokens": self.token_usage["completion_tokens"],
            "total_tokens": self.token_usage["total_tokens"],
        }


metrics = MetricsRegistry()

