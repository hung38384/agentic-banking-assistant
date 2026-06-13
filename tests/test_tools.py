from app.tools.banking_tools import eligibility_checker, fee_lookup, loan_calculator


def test_loan_calculator_returns_amortized_payment():
    result = loan_calculator(amount=100000, rate=6, months=24)

    assert result["monthly_payment"] == 4432.06
    assert result["total_interest"] > 0
    assert result["currency"] == "VND"


def test_fee_lookup_finds_credit_card_late_payment_fee():
    result = fee_lookup("credit card", "late payment")

    assert result["found"] is True
    assert result["amount"] == "4% of overdue minimum payment"


def test_eligibility_checker_requests_missing_fields():
    result = eligibility_checker({"age": 21}, "credit_card")

    assert result["eligible"] is None
    assert "monthly_income" in result["missing_fields"]


def test_eligibility_checker_passes_basic_rules_with_human_review():
    result = eligibility_checker(
        {"age": 25, "monthly_income": 30000000, "employment_months": 12},
        "home_loan",
    )

    assert result["eligible"] is True
    assert result["human_review_required"] is True

