from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.agent_pipeline.common import load_json, save_json


@dataclass(frozen=True)
class AssessmentRules:
    min_positive_test_ratio: float = 0.50
    min_test_outperformed_hold_ratio: float = 0.50
    min_median_test_score: float = 0.0
    min_mean_test_return: float = 0.0
    max_mean_test_drawdown: float = 0.20


def build_default_rules() -> AssessmentRules:
    return AssessmentRules()


def _check(condition: bool, passed_message: str, failed_message: str) -> dict[str, Any]:
    return {
        "passed": bool(condition),
        "message": passed_message if condition else failed_message,
    }


def evaluate_run_directory(run_dir: str | Path, rules: AssessmentRules | None = None) -> dict[str, Any]:
    run_dir = Path(run_dir)
    rules = rules or build_default_rules()

    summary = load_json(run_dir / "summary.json")
    selected_params = pd.read_csv(run_dir / "selected_params.csv")
    aggregate = summary["aggregate"]

    mean_test_drawdown = float(selected_params["test_max_drawdown"].abs().mean())
    checks = {
        "median_test_score": _check(
            aggregate["median_test_score"] is not None
            and aggregate["median_test_score"] >= rules.min_median_test_score,
            "Median test score cleared the minimum threshold.",
            "Median test score stayed below the minimum threshold.",
        ),
        "positive_test_ratio": _check(
            aggregate["positive_test_ratio"] is not None
            and aggregate["positive_test_ratio"] >= rules.min_positive_test_ratio,
            "Enough test windows stayed above zero.",
            "Too few test windows stayed above zero.",
        ),
        "test_outperformed_hold_ratio": _check(
            aggregate["test_outperformed_hold_ratio"] is not None
            and aggregate["test_outperformed_hold_ratio"] >= rules.min_test_outperformed_hold_ratio,
            "The strategy beat buy-and-hold often enough.",
            "The strategy lost to buy-and-hold too often.",
        ),
        "mean_test_return": _check(
            aggregate["mean_test_return"] is not None
            and aggregate["mean_test_return"] >= rules.min_mean_test_return,
            "Mean out-of-sample return stayed positive.",
            "Mean out-of-sample return stayed negative.",
        ),
        "mean_test_drawdown": _check(
            mean_test_drawdown <= rules.max_mean_test_drawdown,
            "Mean test drawdown stayed within the risk budget.",
            "Mean test drawdown exceeded the risk budget.",
        ),
    }

    failed_checks = [name for name, check in checks.items() if not check["passed"]]
    verdict = "pass" if not failed_checks else "fail"

    selected_candidates = (
        selected_params[["candidate_id", "family", "candidate_label", "params_json"]]
        .drop_duplicates()
        .to_dict(orient="records")
    )
    selected_parameter_pairs: list[list[int]] = []
    if {"fast_window", "slow_window"}.issubset(selected_params.columns):
        selected_parameter_pairs = (
            selected_params[["fast_window", "slow_window"]]
            .drop_duplicates()
            .astype(int)
            .values
            .tolist()
        )

    assessment = {
        "run_dir": str(run_dir),
        "verdict": verdict,
        "selection_metric": aggregate["selection_metric"],
        "rules": asdict(rules),
        "aggregate_snapshot": {
            **aggregate,
            "mean_test_drawdown": mean_test_drawdown,
        },
        "checks": checks,
        "failed_checks": failed_checks,
        "selected_candidates": selected_candidates,
        "selected_parameter_pairs": selected_parameter_pairs,
    }
    return assessment


def save_assessment(assessment: dict[str, Any], output_path: str | Path) -> Path:
    return save_json(assessment, output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a completed walk-forward run directory.")
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--min-positive-test-ratio", type=float, default=0.50)
    parser.add_argument("--min-test-outperformed-hold-ratio", type=float, default=0.50)
    parser.add_argument("--min-median-test-score", type=float, default=0.0)
    parser.add_argument("--min-mean-test-return", type=float, default=0.0)
    parser.add_argument("--max-mean-test-drawdown", type=float, default=0.20)
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    rules = AssessmentRules(
        min_positive_test_ratio=args.min_positive_test_ratio,
        min_test_outperformed_hold_ratio=args.min_test_outperformed_hold_ratio,
        min_median_test_score=args.min_median_test_score,
        min_mean_test_return=args.min_mean_test_return,
        max_mean_test_drawdown=args.max_mean_test_drawdown,
    )
    assessment = evaluate_run_directory(args.run_dir, rules=rules)

    if args.output:
        save_assessment(assessment, args.output)

    print(json.dumps(assessment, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
