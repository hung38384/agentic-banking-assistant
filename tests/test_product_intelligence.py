from app.services.actions import ActionEngine
from app.services.feedback import FeedbackStore
from app.services.insights import ProactiveInsightEngine
from app.services.token_optimization import TokenBudgetManager


def test_token_budget_manager_uses_tool_first_policy():
    manager = TokenBudgetManager(default_top_k=4, default_context_chars=4500)

    budget = manager.plan("What is the annual fee?", "fee_lookup", has_tools=True)

    assert budget.top_k == 2
    assert budget.model_tier == "fast"


def test_token_budget_manager_compacts_personal_tool_output():
    manager = TokenBudgetManager()
    tool_outputs = [
        {
            "name": "customer_account_summary",
            "output": {
                "found": True,
                "accounts": [
                    {
                        "product_name": "Savings",
                        "available_balance": 1000,
                        "ledger_balance": 1200,
                        "currency": "VND",
                        "status": "active",
                        "extra": "remove-me",
                    }
                ],
            },
        }
    ]

    compacted = manager.compact_tool_outputs(tool_outputs)

    assert "extra" not in compacted[0]["output"]["accounts"][0]
    assert compacted[0]["output"]["account_count"] == 1


def test_insight_engine_generates_customer_insights_and_actions():
    insights = ProactiveInsightEngine().generate({"customer_id": "CUST1001"})
    actions = ActionEngine().recommend(insights)

    assert insights
    assert any(insight["category"] in {"fee_avoidance", "loan", "balance"} for insight in insights)
    assert actions


def test_action_engine_tracks_lifecycle_for_high_risk_action():
    engine = ActionEngine()

    action = engine.create("LOCK_CARD", "CUST1001", {"card_id": "CARD-9001"})
    transitioned = engine.transition(action["action_id"], "approved")

    assert action["status"] == "pending_approval"
    assert transitioned["status"] == "approved"


def test_feedback_store_reports_content_gaps():
    store = FeedbackStore()
    store.add(
        conversation_id="c1",
        rating=1,
        category="missing_content",
        comment="Need clearer fee explanation.",
    )

    gaps = store.content_gaps()

    assert gaps[0]["category"] == "missing_content"
    assert gaps[0]["count"] == 1

