from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.agent_pipeline.common import save_json, timestamped_run_dir
from examples.agent_pipeline.data_stage import build_data_spec, save_data_spec
from examples.agent_pipeline.result_stage import build_default_rules, evaluate_run_directory, save_assessment
from examples.agent_pipeline.strategy_stage import (
    build_strategy_spec,
    build_walk_forward_config,
    save_strategy_spec,
)
from examples.walk_forward_engine import run_walk_forward


DEFAULT_OUTPUT_ROOT = Path("examples") / "agent_pipeline_runs"


def run_pipeline_once(
    symbol: str = "BTC-USD",
    period: str = "60d",
    interval: str = "5m",
    freq: str | None = None,
    train_days: int = 35,
    validation_days: int = 10,
    test_days: int = 10,
    n_splits: int = 4,
    window_spec: str | None = None,
    candidate_count: int = 8,
    min_trades: int = 1,
    selection_metric: str = "sharpe_ratio",
    direction: str = "longonly",
    fees: float = 0.001,
    init_cash: float = 1_000.0,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, object]:
    run_dir = timestamped_run_dir(output_root, symbol)

    data_spec = build_data_spec(
        symbol=symbol,
        period=period,
        interval=interval,
        freq=freq,
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
        n_splits=n_splits,
    )
    strategy_spec = build_strategy_spec(
        data_spec=data_spec,
        window_spec=window_spec,
        candidate_count=candidate_count,
        min_trades=min_trades,
        selection_metric=selection_metric,
        direction=direction,
        fees=fees,
        init_cash=init_cash,
    )

    data_spec_path = save_data_spec(data_spec, run_dir / "data_spec.json")
    strategy_spec_path = save_strategy_spec(strategy_spec, run_dir / "strategy_spec.json")

    walk_forward_config = build_walk_forward_config(data_spec, strategy_spec)
    engine_result = run_walk_forward(walk_forward_config, output_dir=run_dir)

    assessment = evaluate_run_directory(run_dir, rules=build_default_rules())
    assessment_path = save_assessment(assessment, run_dir / "assessment.json")

    manifest = {
        "run_dir": str(run_dir),
        "verdict": assessment["verdict"],
        "files": {
            "data_spec_json": str(data_spec_path),
            "strategy_spec_json": str(strategy_spec_path),
            "summary_json": engine_result["files"]["summary_json"],
            "selected_params_csv": engine_result["files"]["selected_params_csv"],
            "grid_metrics_csv": engine_result["files"]["grid_metrics_csv"],
            "assessment_json": str(assessment_path),
        },
        "aggregate": engine_result["aggregate"],
        "failed_checks": assessment["failed_checks"],
    }
    save_json(manifest, run_dir / "run_manifest.json")
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one full pass of the agent-ready walk-forward pipeline."
    )
    parser.add_argument("--symbol", default="BTC-USD")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--freq", default=None)
    parser.add_argument("--train-days", type=int, default=35)
    parser.add_argument("--validation-days", type=int, default=10)
    parser.add_argument("--test-days", type=int, default=10)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--window-spec", default=None)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--selection-metric", default="sharpe_ratio", choices=("sharpe_ratio", "total_return"))
    parser.add_argument("--direction", default="longonly", choices=("longonly", "shortonly", "both"))
    parser.add_argument("--fees", type=float, default=0.001)
    parser.add_argument("--init-cash", type=float, default=1_000.0)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    manifest = run_pipeline_once(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
        freq=args.freq,
        train_days=args.train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        n_splits=args.n_splits,
        window_spec=args.window_spec,
        candidate_count=args.candidate_count,
        min_trades=args.min_trades,
        selection_metric=args.selection_metric,
        direction=args.direction,
        fees=args.fees,
        init_cash=args.init_cash,
        output_root=args.output_root,
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
