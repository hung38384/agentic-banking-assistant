import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "eval_questions.jsonl"


def main() -> None:
    cases: list[dict[str, str]] = []

    def add(question: str, expected_source: str, expected_intent: str) -> None:
        cases.append(
            {
                "id": f"q{len(cases) + 1:03d}",
                "question": question,
                "expected_source": expected_source,
                "expected_intent": expected_intent,
            }
        )

    savings = "savings-account"
    credit = "credit-card"
    home = "home-loan"
    personal = "personal-loan"
    digital = "digital-banking-and-core-integration"
    vi = "vietnamese-banking-services"

    for question in [
        "What is the minimum opening balance for the MB Digital Savings Account?",
        "Does the MB Digital Savings Account charge a monthly maintenance fee?",
        "Can the assistant invent the current VND deposit interest rate?",
        "How is savings account interest calculated?",
        "What identity document is needed for account opening?",
        "What is the debit card replacement fee?",
        "How many free NAPAS ATM withdrawals do I get per month?",
        "What is the interbank ATM withdrawal fee after free withdrawals?",
        "When does account closure require branch verification?",
        "What happens if the current deposit rate table is missing?",
        "Does the savings account support QR payments?",
        "Are domestic transfers supported in digital banking?",
        "What is the SMS banking fee condition?",
        "Can the assistant change my account data?",
        "What account situations require human review?",
    ]:
        add(question, savings, "fee_lookup" if "fee" in question.lower() or "withdrawal" in question.lower() or "replacement" in question.lower() else "product_qa")

    for question, intent in [
        ("What is the annual fee for the MB Visa Classic Credit Card?", "fee_lookup"),
        ("What spending amount waives the MB Visa Classic annual fee?", "fee_lookup"),
        ("What is the late payment fee for the credit card?", "fee_lookup"),
        ("What is the cash advance fee for the credit card?", "fee_lookup"),
        ("What minimum income is required for the MB Visa Classic Credit Card?", "eligibility"),
        ("What age is required for the credit card?", "eligibility"),
        ("Do reward points apply to cash advances?", "product_qa"),
        ("Can you guarantee credit card approval?", "product_qa"),
        ("Does the card require CIC review?", "product_qa"),
        ("Which credit card actions need human review?", "product_qa"),
        ("Are gambling transactions eligible for reward points?", "product_qa"),
        ("Can you recommend this card as the best option for me?", "product_qa"),
        ("What happens for refunded transactions and reward points?", "product_qa"),
        ("Does the card require fraud screening?", "product_qa"),
        ("What stable income history is needed for card applicants?", "eligibility"),
    ]:
        add(question, credit, intent)

    for question, intent in [
        ("What is the collateral appraisal fee for a home loan?", "fee_lookup"),
        ("When is the home loan early repayment fee waived?", "fee_lookup"),
        ("What is the minimum income for an MB home loan?", "eligibility"),
        ("How many months of employment are required for a home loan?", "eligibility"),
        ("Does home loan approval depend on collateral valuation?", "product_qa"),
        ("Can the assistant guarantee home loan approval?", "product_qa"),
        ("What third-party fees are excluded from repayment estimates?", "product_qa"),
        ("Can the assistant quote unpublished preferential interest rates?", "product_qa"),
        ("Does a home loan require CIC credit history review?", "product_qa"),
        ("What sensitive home loan actions require human banker review?", "product_qa"),
        ("Calculate a 500000000 VND home loan at 8 percent for 60 months", "loan_calculation"),
        ("Calculate repayment for a 300000000 VND loan at 7.5 percent over 36 months", "loan_calculation"),
        ("What is the early repayment fee range in the first 3 years?", "fee_lookup"),
        ("Can collateral valuation be disputed through the assistant alone?", "product_qa"),
        ("What is the minimum applicant age for a home loan?", "eligibility"),
    ]:
        add(question, home, intent)

    for question, intent in [
        ("What minimum income is required for an MB personal loan?", "eligibility"),
        ("How many employment months are required for a personal loan?", "eligibility"),
        ("Can the assistant state that a personal loan will be approved?", "product_qa"),
        ("Does personal loan approval depend on CIC review?", "product_qa"),
        ("Calculate a 100000000 VND personal loan at 12 percent for 24 months", "loan_calculation"),
        ("Calculate a 75000000 VND personal loan at 10 percent for 18 months", "loan_calculation"),
        ("What personal loan actions require human review?", "product_qa"),
        ("Can the assistant advise a customer to borrow more than they can repay?", "product_qa"),
        ("Does employer verification affect personal loan approval?", "product_qa"),
        ("What data is needed before checking personal loan eligibility?", "eligibility"),
    ]:
        add(question, personal, intent)

    for question, intent in [
        ("Show my account summary", "account_summary"),
        ("Find my recent ATM transactions", "transaction_search"),
        ("What is my loan status and next due date?", "loan_status"),
        ("Create a complaint ticket for suspected fraud", "support"),
        ("Which Hanoi branch supports loan consultation?", "branch_finder"),
        ("Which Ho Chi Minh City branch supports credit card services?", "branch_finder"),
        ("Which Da Nang branch supports cash service?", "branch_finder"),
        ("Can personal banking data be stored in prompts?", "product_qa"),
        ("What does the adapter layer do in core banking integration?", "product_qa"),
        ("Why is customer consent required before personal-data tools?", "product_qa"),
        ("What fields can account summary responses include?", "account_summary"),
        ("What fields can transaction responses include?", "transaction_search"),
        ("What fields can loan status responses include?", "loan_status"),
        ("Can the assistant access my balance without authentication?", "account_summary"),
        ("Can the assistant submit a support ticket without human review?", "support"),
    ]:
        add(question, digital, intent)

    for question, intent in [
        ("Số dư mở tài khoản tối thiểu là bao nhiêu?", "product_qa"),
        ("Phí rút tiền ATM NAPAS ngân hàng khác là bao nhiêu?", "fee_lookup"),
        ("Phí cấp lại thẻ ghi nợ là bao nhiêu?", "fee_lookup"),
        ("Phí thường niên thẻ tín dụng MB Visa Classic là bao nhiêu?", "fee_lookup"),
        ("Điều kiện miễn phí thường niên thẻ tín dụng là gì?", "fee_lookup"),
        ("Thu nhập tối thiểu để mở thẻ tín dụng là bao nhiêu?", "eligibility"),
        ("Phí trả chậm thẻ tín dụng là bao nhiêu?", "fee_lookup"),
        ("Phí ứng tiền mặt thẻ tín dụng là bao nhiêu?", "fee_lookup"),
        ("Vay mua nhà yêu cầu thu nhập tối thiểu bao nhiêu?", "eligibility"),
        ("Phí thẩm định tài sản vay mua nhà là bao nhiêu?", "fee_lookup"),
        ("Trả nợ trước hạn vay mua nhà bị tính phí thế nào?", "fee_lookup"),
        ("Tính khoản vay 100000000 VND lãi suất 10% trong 24 tháng", "loan_calculation"),
        ("Tôi có đủ điều kiện vay tín chấp không nếu thu nhập 9000000 VND?", "eligibility"),
        ("Trợ lý có được tự bịa lãi suất tiền gửi không?", "product_qa"),
        ("Khiếu nại gian lận thẻ cần xử lý thế nào?", "support"),
        ("Tôi muốn xem lịch sử giao dịch gần đây", "transaction_search"),
        ("Cho tôi xem trạng thái khoản vay", "loan_status"),
        ("Chi nhánh Hà Nội nào hỗ trợ tư vấn vay?", "branch_finder"),
        ("Tranh chấp giao dịch có cần nhân viên phê duyệt không?", "support"),
        ("Trợ lý có được đảm bảo phê duyệt khoản vay không?", "product_qa"),
    ]:
        add(question, vi, intent)

    OUT.write_text("\n".join(json.dumps(case, ensure_ascii=False) for case in cases) + "\n", encoding="utf-8")
    print(json.dumps({"path": str(OUT), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
