#!/usr/bin/env python3
"""Run the Kanzlei core eval set against an enhanced-answer endpoint."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.retrieval.eval_runner import format_eval_report, load_eval_set, run_eval_set


def main() -> int:
    parser = argparse.ArgumentParser(description="Run Kanzlei source/audit regression evals")
    parser.add_argument("--eval-set", default=str(ROOT / "evals" / "kanzlei_core.json"))
    parser.add_argument("--endpoint", default=None, help="Override eval-set default endpoint")
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--skip-known-gaps", action="store_true")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON report")
    parser.add_argument(
        "--no-fail-on-regression",
        action="store_true",
        help="Always exit 0, even when blocking regression guards fail",
    )
    args = parser.parse_args()

    eval_set = load_eval_set(args.eval_set)
    summary = run_eval_set(
        eval_set,
        endpoint=args.endpoint,
        top_k=args.top_k,
        include_known_gaps=not args.skip_known_gaps,
        timeout=args.timeout,
    )

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(format_eval_report(summary))

    if not args.no_fail_on_regression and summary["blocking_failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
