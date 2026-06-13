import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.services.mock_core import mock_core


def main() -> None:
    account = mock_core.account_summary("CUST2002")
    transactions = mock_core.search_transactions("CUST2002")
    loans = mock_core.loan_status("CUST2002")
    branches = mock_core.find_branches(province="Hanoi")
    all_flags = sorted(
        set(
            account.get("data_quality_flags", [])
            + transactions.get("data_quality_flags", [])
            + loans.get("data_quality_flags", [])
            + branches.get("data_quality_flags", [])
        )
    )
    summary = {
        "messy_customer_id": "CUST2002",
        "account_quality_flags": account.get("data_quality_flags", []),
        "transaction_quality_flags": transactions.get("data_quality_flags", []),
        "loan_quality_flags": loans.get("data_quality_flags", []),
        "branch_quality_flags": branches.get("data_quality_flags", []),
        "unique_quality_flag_count": len(all_flags),
        "all_quality_flags": all_flags,
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
