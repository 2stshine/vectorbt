from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt


DEFAULT_WINDOW_SPEC = "24:193:24"
DEFAULT_OUTPUT_ROOT = Path("examples") / "walk_forward_runs"
# Keep the selection metric set intentionally small so the agent loop
# works with metrics that have an unambiguous "higher is better" meaning.
SUPPORTED_SELECTION_METRICS = ("sharpe_ratio", "total_return")


@dataclass(frozen=True)
class WalkForwardConfig:
    symbol: str = "BTC-USD"
    period: str = "60d"
    interval: str = "5m"
    freq: str = "5min"
    train_bars: int = 35 * 24 * 12
    validation_bars: int = 10 * 24 * 12
    test_bars: int = 10 * 24 * 12
    n_splits: int = 4
    window_spec: str = DEFAULT_WINDOW_SPEC
    candidate_count: int = 10
    min_trades: int = 1
    selection_metric: str = "sharpe_ratio"
    init_cash: float = 1_000.0
    fees: float = 0.001
    direction: str = "longonly"


def parse_window_spec(window_spec: str) -> np.ndarray:
    window_spec = window_spec.strip()
    if not window_spec:
        raise ValueError("Window spec must not be empty.")

    if ":" in window_spec:
        start_str, stop_str, step_str = window_spec.split(":")
        start = int(start_str)
        stop = int(stop_str)
        step = int(step_str)
        windows = np.arange(start, stop, step, dtype=int)
    else:
        windows = np.array([int(part.strip()) for part in window_spec.split(",") if part.strip()], dtype=int)

    windows = np.unique(windows)
    windows = windows[windows > 0]

    if len(windows) < 2:
        raise ValueError("Need at least two valid windows to build fast/slow pairs.")

    return windows


def load_price_series(symbol: str, period: str, interval: str) -> pd.Series:
    price = vbt.YFData.download(
        symbol,
        period=period,
        interval=interval,
        missing_index="drop",
    ).get("Close")

    if isinstance(price, pd.DataFrame):
        if price.shape[1] != 1:
            raise ValueError("This engine expects a single symbol per run.")
        price = price.iloc[:, 0]

    price = price.dropna()
    price.name = symbol

    if price.empty:
        raise ValueError(f"No price rows were downloaded for {symbol}.")

    return price


def split_price_series(price: pd.Series, config: WalkForwardConfig):
    window_len = config.train_bars + config.validation_bars + config.test_bars
    if len(price) < window_len:
        raise ValueError(
            f"Not enough rows for one split: need at least {window_len}, got {len(price)}."
        )

    split_kwargs = dict(
        n=config.n_splits,
        window_len=window_len,
        set_lens=(config.validation_bars, config.test_bars),
        # Keep the remaining bars on the left so the split order becomes
        # train (remaining) -> validation -> test.
        left_to_right=False,
    )
    return price.vbt.rolling_split(**split_kwargs)


def portfolio_kwargs(config: WalkForwardConfig) -> dict[str, Any]:
    return dict(
        init_cash=config.init_cash,
        fees=config.fees,
        freq=config.freq,
        direction=config.direction,
    )


def collect_metrics(portfolio: vbt.Portfolio) -> pd.DataFrame:
    metrics = pd.DataFrame(
        {
            "sharpe_ratio": portfolio.sharpe_ratio(),
            "total_return": portfolio.total_return(),
            "max_drawdown": portfolio.max_drawdown(),
            "trade_count": portfolio.trades.count(),
        }
    )
    metrics = metrics.replace([np.inf, -np.inf], np.nan)
    return metrics


def simulate_all_param_metrics(price: pd.Series | pd.DataFrame, windows: np.ndarray, config: WalkForwardConfig) -> pd.DataFrame:
    fast_ma, slow_ma = vbt.MA.run_combs(
        price,
        windows,
        r=2,
        short_names=["fast", "slow"],
    )
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    portfolio = vbt.Portfolio.from_signals(price, entries, exits, **portfolio_kwargs(config))
    return collect_metrics(portfolio)


def simulate_holding_metrics(price: pd.Series | pd.DataFrame, config: WalkForwardConfig) -> pd.DataFrame:
    portfolio = vbt.Portfolio.from_holding(
        price,
        init_cash=config.init_cash,
        fees=config.fees,
        freq=config.freq,
    )
    return collect_metrics(portfolio)


def build_selection_scores(metrics: pd.DataFrame, selection_metric: str, min_trades: int) -> pd.Series:
    scores = metrics[selection_metric].copy()
    scores = scores.where(metrics["trade_count"] >= min_trades)
    return scores


def select_best_indices(
    train_metrics: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    selection_metric: str,
    candidate_count: int,
    min_trades: int,
) -> tuple[pd.MultiIndex, dict[int, str], dict[int, int]]:
    selected_indices: list[tuple[Any, ...]] = []
    selection_status: dict[int, str] = {}
    candidate_sizes: dict[int, int] = {}

    train_scores = build_selection_scores(train_metrics, selection_metric, min_trades)
    validation_scores = build_selection_scores(validation_metrics, selection_metric, min_trades)
    split_level = "split_idx"

    for split_idx in train_metrics.index.get_level_values(split_level).unique():
        split_train = train_metrics.xs(split_idx, level=split_level, drop_level=False).copy()
        split_train["selection_score"] = train_scores.loc[split_train.index].values

        eligible_train = split_train.dropna(subset=["selection_score"]).sort_values(
            "selection_score",
            ascending=False,
        )

        if eligible_train.empty:
            candidate_pool = split_train.dropna(subset=[selection_metric]).sort_values(
                selection_metric,
                ascending=False,
            )
            status = "fallback_train_metric"
        else:
            candidate_pool = eligible_train
            status = "selected_from_train_eligible"

        if candidate_pool.empty:
            raise ValueError(f"No valid train candidates found for split {split_idx}.")

        candidate_pool = candidate_pool.head(max(candidate_count, 1))
        candidate_sizes[int(split_idx)] = int(len(candidate_pool))

        split_validation = validation_metrics.loc[candidate_pool.index].copy()
        split_validation["selection_score"] = validation_scores.loc[candidate_pool.index].values
        eligible_validation = split_validation.dropna(subset=["selection_score"]).sort_values(
            "selection_score",
            ascending=False,
        )

        if eligible_validation.empty:
            best_index = candidate_pool.index[0]
            if status == "selected_from_train_eligible":
                status = "fallback_validation_metric"
            else:
                status = "fallback_train_and_validation_metric"
        else:
            best_index = eligible_validation.index[0]

        selected_indices.append(tuple(best_index))
        selection_status[int(split_idx)] = status

    return pd.MultiIndex.from_tuples(selected_indices, names=train_metrics.index.names), selection_status, candidate_sizes


def flatten_metric_frame(
    metrics: pd.DataFrame,
    dataset: str,
    selection_metric: str,
    min_trades: int,
) -> pd.DataFrame:
    flat = metrics.reset_index()
    flat["dataset"] = dataset
    flat["selection_score"] = build_selection_scores(metrics, selection_metric, min_trades).reset_index(drop=True)
    flat["eligible_for_selection"] = flat["selection_score"].notna()
    return flat


def index_bounds(index: pd.Index) -> dict[str, Any]:
    return {
        "start": None if len(index) == 0 else str(index[0]),
        "end": None if len(index) == 0 else str(index[-1]),
        "bars": int(len(index)),
    }


def build_selected_summary(
    selected_index: pd.MultiIndex,
    train_metrics: pd.DataFrame,
    validation_metrics: pd.DataFrame,
    test_metrics: pd.DataFrame,
    train_hold: pd.DataFrame,
    validation_hold: pd.DataFrame,
    test_hold: pd.DataFrame,
    train_indexes: list[pd.Index],
    validation_indexes: list[pd.Index],
    test_indexes: list[pd.Index],
    config: WalkForwardConfig,
    selection_status: dict[int, str],
    candidate_sizes: dict[int, int],
) -> pd.DataFrame:
    selection_metric = config.selection_metric
    train_ranks = train_metrics[selection_metric].groupby(level="split_idx").rank(ascending=False, method="min")
    validation_ranks = validation_metrics[selection_metric].groupby(level="split_idx").rank(ascending=False, method="min")
    test_ranks = test_metrics[selection_metric].groupby(level="split_idx").rank(ascending=False, method="min")

    rows = []
    for params in selected_index:
        fast_window, slow_window, split_idx = params
        split_idx = int(split_idx)
        train_row = train_metrics.loc[params]
        validation_row = validation_metrics.loc[params]
        test_row = test_metrics.loc[params]
        train_hold_row = train_hold.loc[split_idx]
        validation_hold_row = validation_hold.loc[split_idx]
        test_hold_row = test_hold.loc[split_idx]

        row = {
            "split_idx": split_idx,
            "fast_window": int(fast_window),
            "slow_window": int(slow_window),
            "selection_status": selection_status[split_idx],
            "candidate_count_used": candidate_sizes[split_idx],
            "train_rank": float(train_ranks.loc[params]),
            "validation_rank": float(validation_ranks.loc[params]),
            "test_rank": float(test_ranks.loc[params]),
            "train_start": index_bounds(train_indexes[split_idx])["start"],
            "train_end": index_bounds(train_indexes[split_idx])["end"],
            "validation_start": index_bounds(validation_indexes[split_idx])["start"],
            "validation_end": index_bounds(validation_indexes[split_idx])["end"],
            "test_start": index_bounds(test_indexes[split_idx])["start"],
            "test_end": index_bounds(test_indexes[split_idx])["end"],
            "train_bars": index_bounds(train_indexes[split_idx])["bars"],
            "validation_bars": index_bounds(validation_indexes[split_idx])["bars"],
            "test_bars": index_bounds(test_indexes[split_idx])["bars"],
        }

        for prefix, values in (
            ("train", train_row),
            ("validation", validation_row),
            ("test", test_row),
        ):
            row[f"{prefix}_sharpe_ratio"] = values["sharpe_ratio"]
            row[f"{prefix}_total_return"] = values["total_return"]
            row[f"{prefix}_max_drawdown"] = values["max_drawdown"]
            row[f"{prefix}_trade_count"] = values["trade_count"]

        for prefix, values in (
            ("train_hold", train_hold_row),
            ("validation_hold", validation_hold_row),
            ("test_hold", test_hold_row),
        ):
            row[f"{prefix}_sharpe_ratio"] = values["sharpe_ratio"]
            row[f"{prefix}_total_return"] = values["total_return"]
            row[f"{prefix}_max_drawdown"] = values["max_drawdown"]

        row["train_validation_gap"] = row[f"train_{selection_metric}"] - row[f"validation_{selection_metric}"]
        row["validation_test_gap"] = row[f"validation_{selection_metric}"] - row[f"test_{selection_metric}"]
        row["test_vs_hold_gap"] = row[f"test_{selection_metric}"] - row[f"test_hold_{selection_metric}"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("split_idx").reset_index(drop=True)


def safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def build_aggregate_summary(selected_summary: pd.DataFrame, config: WalkForwardConfig) -> dict[str, Any]:
    selection_metric = config.selection_metric
    parameter_pairs = list(zip(selected_summary["fast_window"], selected_summary["slow_window"]))
    parameter_changes = sum(
        parameter_pairs[idx] != parameter_pairs[idx - 1]
        for idx in range(1, len(parameter_pairs))
    )

    return {
        "selection_metric": selection_metric,
        "split_count": int(len(selected_summary)),
        "candidate_count": int(config.candidate_count),
        "min_trades": int(config.min_trades),
        "mean_train_score": safe_float(selected_summary[f"train_{selection_metric}"].mean()),
        "mean_validation_score": safe_float(selected_summary[f"validation_{selection_metric}"].mean()),
        "mean_test_score": safe_float(selected_summary[f"test_{selection_metric}"].mean()),
        "median_test_score": safe_float(selected_summary[f"test_{selection_metric}"].median()),
        "positive_test_ratio": safe_float((selected_summary[f"test_{selection_metric}"] > 0).mean()),
        "test_outperformed_hold_ratio": safe_float(
            (selected_summary[f"test_{selection_metric}"] > selected_summary[f"test_hold_{selection_metric}"]).mean()
        ),
        "mean_train_validation_gap": safe_float(selected_summary["train_validation_gap"].mean()),
        "mean_validation_test_gap": safe_float(selected_summary["validation_test_gap"].mean()),
        "mean_test_return": safe_float(selected_summary["test_total_return"].mean()),
        "mean_test_hold_return": safe_float(selected_summary["test_hold_total_return"].mean()),
        "unique_parameter_pairs": int(selected_summary[["fast_window", "slow_window"]].drop_duplicates().shape[0]),
        "parameter_change_ratio": safe_float(
            parameter_changes / max(len(parameter_pairs) - 1, 1)
        ),
    }


def default_output_dir(symbol: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace("/", "_")
    return DEFAULT_OUTPUT_ROOT / f"{safe_symbol}_{timestamp}"


def save_results(
    output_dir: Path,
    config: WalkForwardConfig,
    price: pd.Series,
    selected_summary: pd.DataFrame,
    grid_metrics: pd.DataFrame,
    aggregate: dict[str, Any],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = output_dir / "selected_params.csv"
    grid_path = output_dir / "grid_metrics.csv"
    summary_path = output_dir / "summary.json"

    selected_summary.to_csv(selected_path, index=False)
    grid_metrics.to_csv(grid_path, index=False)

    summary_payload = {
        "config": asdict(config),
        "parsed_windows": parse_window_spec(config.window_spec).tolist(),
        "data": {
            "symbol": config.symbol,
            "rows": int(len(price)),
            "start": str(price.index[0]),
            "end": str(price.index[-1]),
        },
        "aggregate": aggregate,
        "files": {
            "selected_params_csv": str(selected_path),
            "grid_metrics_csv": str(grid_path),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    return {
        "selected_params_csv": str(selected_path),
        "grid_metrics_csv": str(grid_path),
        "summary_json": str(summary_path),
    }


def run_walk_forward(config: WalkForwardConfig, output_dir: str | Path | None = None) -> dict[str, Any]:
    if config.selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported selection metric {config.selection_metric!r}. "
            f"Choose one of {SUPPORTED_SELECTION_METRICS}."
        )

    windows = parse_window_spec(config.window_spec)
    price = load_price_series(config.symbol, config.period, config.interval)

    (train_price, train_indexes), (validation_price, validation_indexes), (test_price, test_indexes) = split_price_series(
        price,
        config,
    )

    train_metrics = simulate_all_param_metrics(train_price, windows, config)
    validation_metrics = simulate_all_param_metrics(validation_price, windows, config)
    test_metrics = simulate_all_param_metrics(test_price, windows, config)

    train_hold = simulate_holding_metrics(train_price, config)
    validation_hold = simulate_holding_metrics(validation_price, config)
    test_hold = simulate_holding_metrics(test_price, config)

    selected_index, selection_status, candidate_sizes = select_best_indices(
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        selection_metric=config.selection_metric,
        candidate_count=config.candidate_count,
        min_trades=config.min_trades,
    )

    selected_summary = build_selected_summary(
        selected_index=selected_index,
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        test_metrics=test_metrics,
        train_hold=train_hold,
        validation_hold=validation_hold,
        test_hold=test_hold,
        train_indexes=train_indexes,
        validation_indexes=validation_indexes,
        test_indexes=test_indexes,
        config=config,
        selection_status=selection_status,
        candidate_sizes=candidate_sizes,
    )

    grid_metrics = pd.concat(
        [
            flatten_metric_frame(train_metrics, "train", config.selection_metric, config.min_trades),
            flatten_metric_frame(validation_metrics, "validation", config.selection_metric, config.min_trades),
            flatten_metric_frame(test_metrics, "test", config.selection_metric, config.min_trades),
        ],
        ignore_index=True,
    )

    aggregate = build_aggregate_summary(selected_summary, config)
    resolved_output_dir = Path(output_dir) if output_dir is not None else default_output_dir(config.symbol)
    files = save_results(resolved_output_dir, config, price, selected_summary, grid_metrics, aggregate)

    return {
        "config": config,
        "price": price,
        "selected_summary": selected_summary,
        "grid_metrics": grid_metrics,
        "aggregate": aggregate,
        "files": files,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walk-forward train/validation/test engine for vectorbt strategy iteration."
    )
    parser.add_argument("--symbol", default="BTC-USD")
    parser.add_argument("--period", default="60d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--freq", default="5min")
    parser.add_argument("--train-bars", type=int, default=35 * 24 * 12)
    parser.add_argument("--validation-bars", type=int, default=10 * 24 * 12)
    parser.add_argument("--test-bars", type=int, default=10 * 24 * 12)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument("--windows", default=DEFAULT_WINDOW_SPEC, help="Python-style range start:stop:step or comma list.")
    parser.add_argument("--candidate-count", type=int, default=10, help="How many top train candidates reach validation.")
    parser.add_argument("--min-trades", type=int, default=1, help="Minimum trades required for a parameter set to be eligible.")
    parser.add_argument("--selection-metric", default="sharpe_ratio", choices=SUPPORTED_SELECTION_METRICS)
    parser.add_argument("--init-cash", type=float, default=1_000.0)
    parser.add_argument("--fees", type=float, default=0.001)
    parser.add_argument("--direction", default="longonly", choices=("longonly", "shortonly", "both"))
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    config = WalkForwardConfig(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
        freq=args.freq,
        train_bars=args.train_bars,
        validation_bars=args.validation_bars,
        test_bars=args.test_bars,
        n_splits=args.n_splits,
        window_spec=args.windows,
        candidate_count=args.candidate_count,
        min_trades=args.min_trades,
        selection_metric=args.selection_metric,
        init_cash=args.init_cash,
        fees=args.fees,
        direction=args.direction,
    )

    result = run_walk_forward(config=config, output_dir=args.output_dir)
    payload = {
        "aggregate": result["aggregate"],
        "files": result["files"],
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
