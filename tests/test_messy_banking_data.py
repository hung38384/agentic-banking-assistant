import pytest

from app.agent.graph import BankingAgent
from app.core.config import get_settings
from app.services.memory import ConversationMemory
from app.services.mock_core import mock_core
from app.services.vector_store import BankingVectorStore


def test_messy_account_summary_contains_quality_flags():
    result = mock_core.account_summary("CUST2002")

    assert result["found"] is True
    assert "partial_kyc" == result["kyc_level"]
    assert "negative_available_balance" in result["data_quality_flags"]
    assert "account_dormant" in result["data_quality_flags"]


def test_messy_transactions_detect_pending_duplicate_and_fee_posting():
    result = mock_core.search_transactions("CUST2002")

    assert "pending_authorization" in result["data_quality_flags"]
    assert "duplicate_transaction_id" in result["data_quality_flags"]
    assert "standalone_fee_posting" in result["data_quality_flags"]


def test_messy_loan_status_detects_delinquency_and_missing_due():
    result = mock_core.loan_status("CUST2002")

    assert "loan_delinquent" in result["data_quality_flags"]
    assert "missing_next_due_amount" in result["data_quality_flags"]
    assert "loan_restructured" in result["data_quality_flags"]


def test_messy_branch_metadata_detects_stale_closed_service_point():
    result = mock_core.find_branches(province="Hanoi")

    assert "branch_temporarily_closed" in result["data_quality_flags"]
    assert "stale_branch_metadata" in result["data_quality_flags"]


@pytest.mark.asyncio
async def test_agent_adds_data_quality_caveat_for_messy_customer():
    settings = get_settings()
    settings.llm_provider = "mock"
    settings.semantic_cache_enabled = False
    store = BankingVectorStore(settings)
    store.ingest_path(settings.docs_dir, reset=True)
    agent = BankingAgent(settings, store, ConversationMemory())

    state = await agent.run(
        "messy-data-agent-test",
        "Show my account summary",
        {"role": "customer", "customer_id": "CUST2002", "consent_confirmed": True},
    )

    assert "data_quality_caveat" in state["guardrail_flags"]
    assert "quality flags" in state["answer"].lower()
