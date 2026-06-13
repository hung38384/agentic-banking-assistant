import pytest

from app.agent.graph import BankingAgent
from app.core.config import get_settings
from app.services.hallucination import evaluate_groundedness
from app.services.memory import ConversationMemory
from app.services.vector_store import BankingVectorStore


@pytest.fixture()
def agent():
    settings = get_settings()
    settings.llm_provider = "mock"
    settings.semantic_cache_enabled = False
    store = BankingVectorStore(settings)
    store.ingest_path(settings.docs_dir, reset=True)
    return BankingAgent(settings, store, ConversationMemory())


def test_ragas_lite_detects_unsupported_numeric_claim():
    score = evaluate_groundedness(
        question="What is the savings account rate?",
        answer="The savings account rate is 9.9%.",
        retrieved_chunks=[
            {
                "text": "Interest is calculated daily and paid monthly according to the current published deposit rate table.",
                "metadata": {"chunk_id": "savings-account-0"},
            }
        ],
        citations=[{"chunk_id": "savings-account-0"}],
        tool_calls=[],
        guardrail_flags=[],
    )

    assert score.numeric_consistency == 0.0
    assert score.unsupported_numbers == ["9.9%"]
    assert score.hallucination_risk > 0.2


@pytest.mark.asyncio
async def test_current_deposit_rate_request_is_refused(agent):
    result = await agent.run(
        "hallucination-rate-test",
        "Can you tell me the current deposit interest rate for the savings account?",
        {},
    )

    assert "unsupported_rate_request" in result["guardrail_flags"]
    assert "must not invent a rate" in result["answer"]


@pytest.mark.asyncio
async def test_opening_balance_is_product_qa_not_personal_balance(agent):
    result = await agent.run(
        "opening-balance-test",
        "What is the minimum opening balance for the MB Digital Savings Account?",
        {},
    )

    assert result["intent"] == "product_qa"
    assert "authentication_required" not in result["guardrail_flags"]


@pytest.mark.asyncio
async def test_home_loan_early_repayment_routes_to_home_loan_fee(agent):
    result = await agent.run(
        "home-loan-early-repayment-test",
        "When is the home loan early repayment fee waived?",
        {},
    )

    assert result["intent"] == "fee_lookup"
    assert result["tool_calls"][0].name == "fee_lookup"
    assert result["tool_calls"][0].input["product_type"] == "home_loan"
    assert result["tool_calls"][0].input["service_name"] == "early_repayment"
