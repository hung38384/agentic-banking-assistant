from app.services.privacy import redact_text
from app.tools.banking_tools import customer_account_summary, loan_status_lookup, transaction_search


def test_redact_text_masks_common_banking_pii():
    result = redact_text("My phone is 0812345678 and card is 4111 1111 1111 1111")

    assert "[REDACTED_PHONE]" in result.redacted
    assert "[REDACTED_CARD_NUMBER]" in result.redacted
    assert "phone_number" in result.pii_types


def test_account_summary_requires_authenticated_customer_and_consent():
    anonymous = customer_account_summary({})

    assert anonymous["allowed"] is False

    customer_without_consent = customer_account_summary({"role": "customer", "customer_id": "CUST1001"})

    assert customer_without_consent["allowed"] is False
    assert customer_without_consent["human_review_required"] is True


def test_account_summary_returns_masked_mock_core_data_with_consent():
    result = customer_account_summary(
        {"role": "customer", "customer_id": "CUST1001", "consent_confirmed": True}
    )

    assert result["allowed"] is True
    assert result["found"] is True
    assert result["accounts"][0]["account_no"].startswith("*")


def test_transaction_and_loan_tools_return_detailed_mock_data():
    profile = {"role": "customer", "customer_id": "CUST1001", "consent_confirmed": True}

    transactions = transaction_search(profile, keyword="atm")
    loan_status = loan_status_lookup(profile)

    assert transactions["found"] is True
    assert transactions["transactions"][0]["channel"] == "atm"
    assert loan_status["found"] is True
    assert loan_status["loans"][0]["outstanding_principal"] > 0

