import argparse
import asyncio
import json
import statistics
import time
from typing import Any

import httpx


QUESTIONS = [
    "What is the annual fee for the MB Visa Classic Credit Card?",
    "What is the minimum opening balance for the MB Digital Savings Account?",
    "Calculate a 100000000 VND personal loan at 12 percent for 24 months",
    "Can you tell me the current deposit interest rate for the savings account?",
    "Which Hanoi branch supports loan consultation?",
    "Phí thường niên thẻ tín dụng MB Visa Classic là bao nhiêu?",
]


async def main() -> None:
    parser = argparse.ArgumentParser(description="Lightweight /chat load test.")
    parser.add_argument("--url", default="http://localhost:8000/chat")
    parser.add_argument("--api-key", default="dev-secret")
    parser.add_argument("--requests", type=int, default=30)
    parser.add_argument("--concurrency", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    semaphore = asyncio.Semaphore(args.concurrency)
    results: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=args.timeout) as client:
        tasks = [
            _send_one(client, semaphore, args.url, args.api_key, idx)
            for idx in range(args.requests)
        ]
        for result in await asyncio.gather(*tasks):
            results.append(result)

    latencies = [item["latency_ms"] for item in results if item["ok"]]
    failed = [item for item in results if not item["ok"]]
    summary = {
        "requests": args.requests,
        "concurrency": args.concurrency,
        "success_rate": round((args.requests - len(failed)) / args.requests, 4),
        "failures": len(failed),
        "p50_latency_ms": _percentile(latencies, 50),
        "p95_latency_ms": _percentile(latencies, 95),
        "p99_latency_ms": _percentile(latencies, 99),
        "avg_latency_ms": round(statistics.mean(latencies), 2) if latencies else None,
        "errors": failed[:5],
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))


async def _send_one(
    client: httpx.AsyncClient,
    semaphore: asyncio.Semaphore,
    url: str,
    api_key: str,
    idx: int,
) -> dict[str, Any]:
    async with semaphore:
        payload = {
            "conversation_id": f"load-test-{idx % 7}",
            "message": QUESTIONS[idx % len(QUESTIONS)],
            "user_profile": {},
        }
        start = time.perf_counter()
        try:
            response = await client.post(url, headers={"X-API-Key": api_key}, json=payload)
            latency = round((time.perf_counter() - start) * 1000, 2)
            return {
                "ok": response.status_code < 500,
                "status_code": response.status_code,
                "latency_ms": latency,
            }
        except Exception as exc:
            return {"ok": False, "status_code": None, "latency_ms": 0, "error": str(exc)}


def _percentile(values: list[float], percentile: int) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return round(ordered[index], 2)


if __name__ == "__main__":
    asyncio.run(main())
