import pytest

from app.agent.graph import BankingAgent
from app.core.config import get_settings
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


@pytest.mark.asyncio
async def test_memory_fills_missing_loan_months(agent):
    first = await agent.run(
        "loan-memory-test",
        "Calculate a 100000 VND loan at 6 percent",
        {},
    )

    assert "clarification_required" in first["guardrail_flags"]

    second = await agent.run(
        "loan-memory-test",
        "24 months",
        {},
    )

    assert second["intent"] == "loan_calculation"
    assert second["tool_calls"][0].name == "loan_calculator"
    assert "monthly_payment" in second["tool_calls"][0].output


@pytest.mark.asyncio
async def test_memory_fills_eligibility_profile(agent):
    first = await agent.run(
        "eligibility-memory-test",
        "Am I eligible for a home loan? My age 25 and income 30000000",
        {},
    )

    assert "clarification_required" in first["guardrail_flags"]

    second = await agent.run(
        "eligibility-memory-test",
        "12 months employed",
        {},
    )

    assert second["intent"] == "eligibility"
    assert second["tool_calls"][0].name == "eligibility_checker"
    assert second["tool_calls"][0].output["eligible"] is True
