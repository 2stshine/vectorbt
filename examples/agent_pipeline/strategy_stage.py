from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.agent_pipeline.common import load_json, save_json
from examples.agent_pipeline.data_stage import DataSpec, build_data_spec
from examples.agent_pipeline.strategy_registry import (
    SUPPORTED_STRATEGY_FAMILIES,
    default_search_space_for_family,
    default_window_spec_for_interval,
    normalize_family,
    sanitize_search_space,
)
from examples.walk_forward_engine import WalkForwardConfig


@dataclass(frozen=True)
class StrategySpec:
    strategy_name: str = "ma_crossover_ls"
    family: str = "ma_crossover_ls"
    window_spec: str | None = "6,12,18,24,36,48,72,96,144"
    search_space: dict[str, list[int | float]] | None = None
    candidate_count: int = 8
    min_trades: int = 1
    selection_metric: str = "sharpe_ratio"
    direction: str = "both"
    fees: float = 0.00015
    init_cash: float = 1_000.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_search_space(
    family: str,
    interval: str,
    *,
    window_spec: str | None,
    search_space: dict[str, Any] | None,
) -> dict[str, list[int | float]]:
    return sanitize_search_space(
        family,
        search_space,
        interval=interval,
        window_spec=window_spec,
    )


def build_strategy_spec(
    data_spec: DataSpec,
    family: str = "ma_crossover_ls",
    strategy_name: str | None = None,
    window_spec: str | None = None,
    search_space: dict[str, Any] | None = None,
    candidate_count: int = 8,
    min_trades: int = 1,
    selection_metric: str = "sharpe_ratio",
    direction: str = "both",
    fees: float = 0.00015,
    init_cash: float = 1_000.0,
) -> StrategySpec:
    normalized_family = normalize_family(family)
    resolved_window_spec = window_spec
    if normalized_family in {"ma_crossover_ls", "ma_rsi_filter_ls"} and resolved_window_spec is None:
        resolved_window_spec = default_window_spec_for_interval(data_spec.interval)

    resolved_search_space = _resolve_search_space(
        normalized_family,
        data_spec.interval,
        window_spec=resolved_window_spec,
        search_space=search_space,
    )
    return StrategySpec(
        strategy_name=strategy_name or normalized_family,
        family=normalized_family,
        window_spec=resolved_window_spec,
        search_space=resolved_search_space,
        candidate_count=candidate_count,
        min_trades=min_trades,
        selection_metric=selection_metric,
        direction=direction,
        fees=fees,
        init_cash=init_cash,
    )


def strategy_spec_from_dict(payload: dict[str, object]) -> StrategySpec:
    return StrategySpec(**payload)


def build_walk_forward_config(data_spec: DataSpec, strategy_spec: StrategySpec) -> WalkForwardConfig:
    return WalkForwardConfig(
        symbol=data_spec.symbol,
        data_source=data_spec.data_source,
        exchange=data_spec.exchange,
        period=data_spec.period,
        start=data_spec.start,
        end=data_spec.end,
        interval=data_spec.interval,
        freq=data_spec.freq,
        train_bars=data_spec.train_bars,
        validation_bars=data_spec.validation_bars,
        test_bars=data_spec.test_bars,
        n_splits=data_spec.n_splits,
        strategy_name=strategy_spec.strategy_name,
        strategy_family=strategy_spec.family,
        search_space=strategy_spec.search_space,
        window_spec=strategy_spec.window_spec,
        candidate_count=strategy_spec.candidate_count,
        min_trades=strategy_spec.min_trades,
        selection_metric=strategy_spec.selection_metric,
        init_cash=strategy_spec.init_cash,
        fees=strategy_spec.fees,
        direction=strategy_spec.direction,
    )


def save_strategy_spec(strategy_spec: StrategySpec, output_path: str | Path) -> Path:
    return save_json(strategy_spec.to_dict(), output_path)


def data_spec_from_json(path: str | Path) -> DataSpec:
    payload = load_json(path)
    return DataSpec(
        symbol=payload["symbol"],
        data_source=payload.get("data_source", "yahoo"),
        exchange=payload.get("exchange", "binanceusdm"),
        period=payload.get("period"),
        start=payload.get("start"),
        end=payload.get("end"),
        interval=payload["interval"],
        freq=payload["freq"],
        train_days=payload["train_days"],
        validation_days=payload["validation_days"],
        test_days=payload["test_days"],
        n_splits=payload["n_splits"],
    )


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a strategy specification for the walk-forward pipeline.")
    parser.add_argument("--data-spec", default=None, help="Path to data_spec.json. If omitted, build a default BTC config.")
    parser.add_argument("--family", default="ma_crossover_ls", choices=SUPPORTED_STRATEGY_FAMILIES)
    parser.add_argument("--strategy-name", default=None)
    parser.add_argument("--window-spec", default=None)
    parser.add_argument(
        "--search-space-json",
        default=None,
        help="JSON object describing the search space. If omitted, family defaults are used.",
    )
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--selection-metric", default="sharpe_ratio", choices=("sharpe_ratio", "total_return"))
    parser.add_argument("--direction", default="both", choices=("longonly", "shortonly", "both"))
    parser.add_argument("--fees", type=float, default=0.00015)
    parser.add_argument("--init-cash", type=float, default=1_000.0)
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    if args.data_spec:
        data_spec = data_spec_from_json(args.data_spec)
    else:
        data_spec = build_data_spec()

    search_space = None if args.search_space_json is None else json.loads(args.search_space_json)
    strategy_spec = build_strategy_spec(
        data_spec=data_spec,
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
    )

    if args.output:
        save_strategy_spec(strategy_spec, args.output)

    print(json.dumps(strategy_spec.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
