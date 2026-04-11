from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.agent_pipeline.common import load_json, save_json
from examples.agent_pipeline.data_stage import DataSpec, build_data_spec
from examples.walk_forward_engine import WalkForwardConfig


DEFAULT_WINDOWS_BY_INTERVAL = {
    "1m": "120:961:120",
    "2m": "60:481:60",
    "5m": "24:193:24",
    "15m": "8:65:8",
    "30m": "4:33:4",
    "60m": "2:17:2",
    "1h": "2:17:2",
    "1d": "5:51:5",
}


@dataclass(frozen=True)
class StrategySpec:
    strategy_name: str = "ma_crossover"
    family: str = "moving_average_cross"
    window_spec: str = "24:193:24"
    candidate_count: int = 8
    min_trades: int = 1
    selection_metric: str = "sharpe_ratio"
    direction: str = "longonly"
    fees: float = 0.001
    init_cash: float = 1_000.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def default_window_spec_for_interval(interval: str) -> str:
    return DEFAULT_WINDOWS_BY_INTERVAL.get(interval, "24:193:24")


def build_strategy_spec(
    data_spec: DataSpec,
    window_spec: str | None = None,
    candidate_count: int = 8,
    min_trades: int = 1,
    selection_metric: str = "sharpe_ratio",
    direction: str = "longonly",
    fees: float = 0.001,
    init_cash: float = 1_000.0,
) -> StrategySpec:
    return StrategySpec(
        window_spec=window_spec or default_window_spec_for_interval(data_spec.interval),
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
        period=data_spec.period,
        interval=data_spec.interval,
        freq=data_spec.freq,
        train_bars=data_spec.train_bars,
        validation_bars=data_spec.validation_bars,
        test_bars=data_spec.test_bars,
        n_splits=data_spec.n_splits,
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
        period=payload["period"],
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
    parser.add_argument("--window-spec", default=None)
    parser.add_argument("--candidate-count", type=int, default=8)
    parser.add_argument("--min-trades", type=int, default=1)
    parser.add_argument("--selection-metric", default="sharpe_ratio", choices=("sharpe_ratio", "total_return"))
    parser.add_argument("--direction", default="longonly", choices=("longonly", "shortonly", "both"))
    parser.add_argument("--fees", type=float, default=0.001)
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

    strategy_spec = build_strategy_spec(
        data_spec=data_spec,
        window_spec=args.window_spec,
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
