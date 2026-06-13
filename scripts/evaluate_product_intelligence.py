import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.actions import action_engine
from app.services.feedback import FeedbackStore
from app.services.insights import insight_engine
from app.services.token_optimization import token_budget_manager


def main() -> None:
    start = time.perf_counter()
    insights = insight_engine.generate({"customer_id": "CUST1001"})
    actions = action_engine.recommend(insights)
    original_payload = [
        {
            "name": "transaction_search",
            "output": {
                "found": True,
                "count": 10,
                "transactions": [
                    {
                        "posted_date": "2026-06-05",
                        "description": "ATM withdrawal foreign bank",
                        "amount": 3000,
                        "direction": "debit",
                        "channel": "atm",
                        "merchant_category": "cash",
                        "fee": 15,
                        "balance_after": 117791,
                        "raw_memo": "extra verbose memo that should not be sent to the model",
                    }
                ]
                * 5,
            },
        }
    ]
    compacted = token_budget_manager.compact_tool_outputs(original_payload)
    original_tokens = token_budget_manager.estimate_tokens(original_payload)
    compact_tokens = token_budget_manager.estimate_tokens(compacted)
    feedback = FeedbackStore()
    feedback.add(
        conversation_id="eval-product-intelligence",
        rating=1,
        category="missing_content",
        comment="Need clearer fraud dispute workflow.",
    )
    summary = {
        "insight_count": len(insights),
        "recommended_action_count": len(actions),
        "token_compaction_ratio": round(compact_tokens / max(original_tokens, 1), 4),
        "estimated_tokens_saved": max(original_tokens - compact_tokens, 0),
        "content_gap_categories": len(feedback.content_gaps()),
        "latency_ms": round((time.perf_counter() - start) * 1000, 2),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

