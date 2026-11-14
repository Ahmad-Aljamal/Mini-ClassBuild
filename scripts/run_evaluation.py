#!/usr/bin/env python3
"""Step 10: Offline Evaluation Runner for Mini ClassBuild."""
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.evaluation import run_offline_evaluation


def main() -> int:
    print("=" * 76)
    print("MINI CLASSBUILD - STEP 10: OFFLINE VALIDATION & EVALUATION REPORT")
    print("=" * 76)
    results = run_offline_evaluation()
    passed_count = 0

    for res in results:
        status = "PASS" if res.passed else "FAIL"
        if res.passed:
            passed_count += 1
        print(f"[{status}] {res.name}")
        print(f"       Description : {res.description}")
        print(f"       Expected    : {res.expected_outcome}")
        print(f"       Actual      : {res.actual_outcome}")
        print(f"       Details     : {res.details}")
        print("-" * 76)

    total = len(results)
    failed_count = total - passed_count
    print(f"SUMMARY: {passed_count}/{total} scenarios passed, {failed_count} failed.")
    print("=" * 76)
    return 0 if failed_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
