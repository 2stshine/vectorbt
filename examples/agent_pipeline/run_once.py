from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.agent_pipeline.common import save_json, timestamped_run_dir, update_latest_pointer
from examples.agent_pipeline.data_stage import build_data_spec, save_data_spec
from examples.agent_pipeline.reporting import generate_run_report
from examples.agent_pipeline.result_stage import build_default_rules, evaluate_run_directory, save_assessment
from examples.agent_pipeline.strategy_stage import (
    build_strategy_spec,
    build_walk_forward_config,
    save_strategy_spec,
)
from examples.agent_pipeline.strategy_registry import SUPPORTED_STRATEGY_FAMILIES
from examples.walk_forward_engine import run_walk_forward


DEFAULT_OUTPUT_ROOT = Path("examples") / "agent_pipeline_runs"


def run_pipeline_once(
    symbol: str | None = None,
    data_source: str = "ccxt",
    exchange: str = "binanceusdm",
    period: str | None = "540d",
    start: str | None = None,
    end: str | None = None,
    interval: str = "5m",
    freq: str | None = None,
    train_days: int = 45,
    validation_days: int = 15,
    test_days: int = 15,
    n_splits: int = 6,
    family: str = "ma_crossover_ls",
    strategy_name: str | None = None,
    window_spec: str | None = None,
    search_space: dict[str, object] | None = None,
    candidate_count: int = 8,
    min_trades: int = 1,
    selection_metric: str = "sharpe_ratio",
    direction: str = "both",
    fees: float = 0.00015,
    init_cash: float = 1_000.0,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, object]:
    data_spec = build_data_spec(
        symbol=symbol,
        data_source=data_source,
        exchange=exchange,
        period=period,
        start=start,
        end=end,
        interval=interval,
        freq=freq,
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
        n_splits=n_splits,
    )
    run_dir = timestamped_run_dir(output_root, data_spec.symbol)
    strategy_spec = build_strategy_spec(
        data_spec=data_spec,
        family=family,
        strategy_name=strategy_name,
        window_spec=window_spec,
        search_space=search_space,
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
    report_artifacts = generate_run_report(run_dir, assessment=assessment)

    manifest = {
        "run_dir": str(run_dir),
        "verdict": assessment["verdict"],
        "files": {
            "data_spec_json": str(data_spec_path),
            "strategy_spec_json": str(strategy_spec_path),
            "summary_json": engine_result["files"]["summary_json"],
            "selected_params_csv": engine_result["files"]["selected_params_csv"],
            "grid_metrics_csv": engine_result["files"]["grid_metrics_csv"],
            "test_timeline_csv": engine_result["files"]["test_timeline_csv"],
            "overfitting_timeline_csv": engine_result["files"]["overfitting_timeline_csv"],
            "assessment_json": str(assessment_path),
            "summary_ko_md": report_artifacts["summary_ko_path"],
            "hold_comparison_png": report_artifacts["hold_comparison"],
            "return_timeline_png": report_artifacts["return_timeline"],
            "overfitting_timeline_png": report_artifacts["overfitting_timeline"],
        },
        "aggregate": engine_result["aggregate"],
        "failed_checks": assessment["failed_checks"],
    }
    save_json(manifest, run_dir / "run_manifest.json")
    latest_pointer = update_latest_pointer(output_root, run_dir)
    manifest["latest"] = latest_pointer
    save_json(manifest, run_dir / "run_manifest.json")
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one full pass of the agent-ready walk-forward pipeline."
    )
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--data-source", default="ccxt", choices=("yahoo", "ccxt"))
    parser.add_argument("--exchange", default="binanceusdm")
    parser.add_argument("--period", default="540d")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--freq", default=None)
    parser.add_argument("--train-days", type=int, default=45)
    parser.add_argument("--validation-days", type=int, default=15)
    parser.add_argument("--test-days", type=int, default=15)
    parser.add_argument("--n-splits", type=int, default=6)
    parser.add_argument("--family", default="ma_crossover_ls", choices=SUPPORTED_STRATEGY_FAMILIES)
    parser.add_argument("--strategy-name", default=None)
    parser.add_argument("--window-spec", default=None)
    parser.add_argument("--search-space-json", default=None)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--selection-metric", default="sharpe_ratio", choices=("sharpe_ratio", "total_return"))
    parser.add_argument("--direction", default="both", choices=("longonly", "shortonly", "both"))
    parser.add_argument("--fees", type=float, default=0.00015)
    parser.add_argument("--init-cash", type=float, default=1_000.0)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    search_space = None if args.search_space_json is None else json.loads(args.search_space_json)

    manifest = run_pipeline_once(
        symbol=args.symbol,
        data_source=args.data_source,
        exchange=args.exchange,
        period=args.period,
        start=args.start,
        end=args.end,
        interval=args.interval,
        freq=args.freq,
        train_days=args.train_days,
        validation_days=args.validation_days,
        test_days=args.test_days,
        n_splits=args.n_splits,
        family=args.family,
        strategy_name=args.strategy_name,
        window_spec=args.window_spec,
        search_space=search_space,
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
