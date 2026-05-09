from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt

from examples.agent_pipeline.strategy_registry import (
    SUPPORTED_STRATEGY_FAMILIES,
    StrategyCandidate,
    apply_direction_to_signals,
    build_strategy_candidates,
    candidate_meta_frame,
    generate_candidate_signals,
    parse_window_spec,
    simulate_candidate_metrics,
)


DEFAULT_WINDOW_SPEC = "6,12,18,24,36,48,72,96,144"
DEFAULT_OUTPUT_ROOT = Path("examples") / "walk_forward_runs"
SUPPORTED_DATA_SOURCES = ("yahoo", "ccxt")
DEFAULT_SYMBOL_BY_SOURCE = {
    "yahoo": "BTC-USD",
    "ccxt": "BTC/USDT:USDT",
}
_PERIOD_PATTERN = re.compile(r"^\s*(\d+)\s*([A-Za-z]+)\s*$")
SUPPORTED_SELECTION_METRICS = ("sharpe_ratio", "total_return")


@dataclass(frozen=True)
class WalkForwardConfig:
    symbol: str = "BTC/USDT:USDT"
    data_source: str = "ccxt"
    exchange: str = "binanceusdm"
    period: str | None = "540d"
    start: str | None = None
    end: str | None = None
    interval: str = "5m"
    freq: str = "5min"
    train_bars: int = 45 * 24 * 12
    validation_bars: int = 15 * 24 * 12
    test_bars: int = 15 * 24 * 12
    n_splits: int = 6
    strategy_name: str = "ma_crossover_ls"
    strategy_family: str = "ma_crossover_ls"
    search_space: dict[str, list[int | float]] | None = None
    window_spec: str | None = DEFAULT_WINDOW_SPEC
    candidate_count: int = 10
    min_trades: int = 1
    selection_metric: str = "sharpe_ratio"
    init_cash: float = 1_000.0
    fees: float = 0.00015
    direction: str = "both"


def normalize_data_source(data_source: str) -> str:
    normalized = data_source.strip().lower()
    if normalized not in SUPPORTED_DATA_SOURCES:
        raise ValueError(
            f"Unsupported data source {data_source!r}. "
            f"Choose one of {SUPPORTED_DATA_SOURCES}."
        )
    return normalized


def parse_datetime_utc(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def parse_period_offset(period: str) -> pd.DateOffset | pd.Timedelta:
    match = _PERIOD_PATTERN.fullmatch(period)
    if match is None:
        raise ValueError(
            f"Unsupported period {period!r}. "
            "Use forms like '180d', '540d', '2y', '6mo', or '12h'."
        )

    quantity = int(match.group(1))
    unit = match.group(2).lower()

    if unit in {"min", "mins", "minute", "minutes"}:
        return pd.Timedelta(minutes=quantity)
    if unit in {"h", "hr", "hrs", "hour", "hours"}:
        return pd.Timedelta(hours=quantity)
    if unit in {"d", "day", "days"}:
        return pd.Timedelta(days=quantity)
    if unit in {"w", "wk", "wks", "week", "weeks"}:
        return pd.Timedelta(weeks=quantity)
    if unit in {"mo", "mon", "month", "months"}:
        return pd.DateOffset(months=quantity)
    if unit in {"y", "yr", "yrs", "year", "years"}:
        return pd.DateOffset(years=quantity)

    raise ValueError(
        f"Unsupported period unit {unit!r}. "
        "Use minute/hour/day/week/month/year style suffixes."
    )


def resolve_history_range(
    period: str | None,
    start: str | None,
    end: str | None,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    resolved_end = parse_datetime_utc(end) or pd.Timestamp.now(tz="UTC").floor("min")
    resolved_start = parse_datetime_utc(start)

    if resolved_start is None:
        if period is None:
            raise ValueError("Either period or start must be provided.")
        resolved_start = resolved_end - parse_period_offset(period)

    if resolved_start >= resolved_end:
        raise ValueError(
            f"Resolved start {resolved_start} must be earlier than end {resolved_end}."
        )

    return resolved_start, resolved_end


def format_timestamp(value: pd.Timestamp | None) -> str | None:
    if value is None:
        return None
    return value.isoformat()


def resolve_download_context(config: WalkForwardConfig) -> dict[str, Any]:
    data_source = normalize_data_source(config.data_source)
    context: dict[str, Any] = {
        "data_source": data_source,
        "symbol": config.symbol,
        "interval": config.interval,
        "period": config.period,
        "start": config.start,
        "end": config.end,
        "exchange": config.exchange if data_source == "ccxt" else None,
    }

    if data_source == "ccxt" or config.start is not None or config.end is not None:
        resolved_start, resolved_end = resolve_history_range(config.period, config.start, config.end)
        context["resolved_start"] = format_timestamp(resolved_start)
        context["resolved_end"] = format_timestamp(resolved_end)

    return context


def ensure_single_symbol_series(price: pd.Series | pd.DataFrame, symbol: str) -> pd.Series:
    if isinstance(price, pd.DataFrame):
        if price.shape[1] != 1:
            raise ValueError("This engine expects a single symbol per run.")
        price = price.iloc[:, 0]

    price = price.dropna()
    price.name = symbol

    if price.empty:
        raise ValueError(f"No price rows were downloaded for {symbol}.")

    return price


def load_price_series(config: WalkForwardConfig) -> tuple[pd.Series, dict[str, Any]]:
    download_context = resolve_download_context(config)
    data_source = download_context["data_source"]

    if data_source == "yahoo":
        download_kwargs: dict[str, Any] = {
            "interval": config.interval,
            "missing_index": "drop",
        }
        if "resolved_start" in download_context:
            download_kwargs["start"] = download_context["resolved_start"]
            download_kwargs["end"] = download_context["resolved_end"]
        else:
            if config.period is None:
                raise ValueError("Yahoo downloads require period or start/end.")
            download_kwargs["period"] = config.period

        price = vbt.YFData.download(config.symbol, **download_kwargs).get("Close")
        return ensure_single_symbol_series(price, config.symbol), download_context

    if data_source == "ccxt":
        try:
            price = vbt.CCXTData.download(
                config.symbol,
                exchange=config.exchange,
                timeframe=config.interval,
                start=download_context["resolved_start"],
                end=download_context["resolved_end"],
                limit=1000,
                show_progress=False,
                missing_index="drop",
            ).get("Close")
        except ModuleNotFoundError as exc:
            if exc.name == "ccxt":
                raise RuntimeError(
                    "CCXT data source requires the `ccxt` package. "
                    "Install it in the active environment first."
                ) from exc
            raise
        return ensure_single_symbol_series(price, config.symbol), download_context

    raise AssertionError(f"Unhandled data source: {data_source}")


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
        left_to_right=False,
    )
    return price.vbt.rolling_split(**split_kwargs)


def collect_metrics(portfolio: vbt.Portfolio) -> pd.DataFrame:
    metric_payload = {
        "sharpe_ratio": portfolio.sharpe_ratio(),
        "total_return": portfolio.total_return(),
        "max_drawdown": portfolio.max_drawdown(),
        "trade_count": portfolio.trades.count(),
    }
    normalized: dict[str, pd.Series] = {}
    for key, value in metric_payload.items():
        if isinstance(value, pd.DataFrame):
            if value.shape[0] == 1:
                series = value.iloc[0]
            elif value.shape[1] == 1:
                series = value.iloc[:, 0]
            else:
                raise ValueError(f"Metric {key!r} returned an unsupported DataFrame shape {value.shape}.")
        elif isinstance(value, pd.Series):
            series = value
        else:
            series = pd.Series({"_single": value})
        normalized[key] = series
    metrics = pd.concat(normalized, axis=1)
    return metrics.replace([np.inf, -np.inf], np.nan)


def simulate_holding_metrics(price: pd.Series, config: WalkForwardConfig) -> dict[str, float | None]:
    portfolio = vbt.Portfolio.from_holding(
        price,
        init_cash=config.init_cash,
        fees=config.fees,
        freq=config.freq,
    )
    metrics = collect_metrics(portfolio)
    if isinstance(metrics, pd.DataFrame):
        if len(metrics) != 1:
            raise ValueError("Expected a single holding metric row.")
        row = metrics.iloc[0]
    else:
        row = metrics
    return {key: safe_float(row[key]) for key in ("sharpe_ratio", "total_return", "max_drawdown", "trade_count")}


def build_selection_scores(metrics: pd.DataFrame, selection_metric: str, min_trades: int) -> pd.Series:
    scores = metrics[selection_metric].copy()
    scores = scores.where(metrics["trade_count"] >= min_trades)
    return scores


def simulate_candidate_metrics_by_splits(
    price: pd.Series,
    split_indexes: list[pd.Index],
    candidates: list[StrategyCandidate],
    config: WalkForwardConfig,
) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for split_idx, split_index in enumerate(split_indexes):
        split_price = price.loc[split_index].copy()
        split_metrics = simulate_candidate_metrics(
            split_price,
            candidates,
            init_cash=config.init_cash,
            fees=config.fees,
            freq=config.freq,
            direction=config.direction,
        )
        split_metrics["candidate_id"] = split_metrics.index
        split_metrics["split_idx"] = split_idx
        frames.append(split_metrics.reset_index(drop=True).set_index(["candidate_id", "split_idx"]))
    return pd.concat(frames).sort_index()


def simulate_holding_metrics_by_splits(
    price: pd.Series,
    split_indexes: list[pd.Index],
    config: WalkForwardConfig,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for split_idx, split_index in enumerate(split_indexes):
        split_price = price.loc[split_index].copy()
        row = {"split_idx": split_idx}
        row.update(simulate_holding_metrics(split_price, config))
        rows.append(row)
    return pd.DataFrame(rows).set_index("split_idx").sort_index()


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
    candidate_meta: pd.DataFrame,
) -> pd.DataFrame:
    flat = metrics.reset_index()
    flat["dataset"] = dataset
    flat["selection_score"] = build_selection_scores(metrics, selection_metric, min_trades).reset_index(drop=True)
    flat["eligible_for_selection"] = flat["selection_score"].notna()
    return flat.merge(candidate_meta.reset_index(), on="candidate_id", how="left")


def index_bounds(index: pd.Index) -> dict[str, Any]:
    return {
        "start": None if len(index) == 0 else str(index[0]),
        "end": None if len(index) == 0 else str(index[-1]),
        "bars": int(len(index)),
    }


def build_selected_summary(
    selected_index: pd.MultiIndex,
    candidate_meta: pd.DataFrame,
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
        candidate_id, split_idx = params
        split_idx = int(split_idx)
        meta_row = candidate_meta.loc[candidate_id]
        train_row = train_metrics.loc[params]
        validation_row = validation_metrics.loc[params]
        test_row = test_metrics.loc[params]
        train_hold_row = train_hold.loc[split_idx]
        validation_hold_row = validation_hold.loc[split_idx]
        test_hold_row = test_hold.loc[split_idx]

        row = {
            "split_idx": split_idx,
            "candidate_id": candidate_id,
            "family": meta_row["family"],
            "strategy_name": meta_row["strategy_name"],
            "candidate_label": meta_row["candidate_label"],
            "params_json": meta_row["params_json"],
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

        for meta_key, meta_value in meta_row.items():
            if meta_key in {"family", "strategy_name", "candidate_label", "params_json"}:
                continue
            row[meta_key] = meta_value

        for prefix, values in (
            ("train", train_row),
            ("validation", validation_row),
            ("test", test_row),
        ):
            row[f"{prefix}_sharpe_ratio"] = scalar_value(values["sharpe_ratio"])
            row[f"{prefix}_total_return"] = scalar_value(values["total_return"])
            row[f"{prefix}_max_drawdown"] = scalar_value(values["max_drawdown"])
            row[f"{prefix}_trade_count"] = scalar_value(values["trade_count"])

        for prefix, values in (
            ("train_hold", train_hold_row),
            ("validation_hold", validation_hold_row),
            ("test_hold", test_hold_row),
        ):
            row[f"{prefix}_sharpe_ratio"] = scalar_value(values["sharpe_ratio"])
            row[f"{prefix}_total_return"] = scalar_value(values["total_return"])
            row[f"{prefix}_max_drawdown"] = scalar_value(values["max_drawdown"])

        row["train_validation_gap"] = row[f"train_{selection_metric}"] - row[f"validation_{selection_metric}"]
        row["validation_test_gap"] = row[f"validation_{selection_metric}"] - row[f"test_{selection_metric}"]
        row["test_vs_hold_gap"] = row[f"test_{selection_metric}"] - row[f"test_hold_{selection_metric}"]
        rows.append(row)

    return pd.DataFrame(rows).sort_values("split_idx").reset_index(drop=True)


def safe_float(value: Any) -> float | None:
    if pd.isna(value):
        return None
    return float(value)


def scalar_value(value: Any) -> Any:
    if isinstance(value, pd.DataFrame):
        if value.size != 1:
            raise ValueError("Expected a single scalar value, but received a DataFrame with multiple values.")
        return value.iloc[0, 0]
    if isinstance(value, pd.Series):
        if len(value) != 1:
            raise ValueError("Expected a single scalar value, but received a Series with multiple values.")
        return value.iloc[0]
    return value


def build_aggregate_summary(selected_summary: pd.DataFrame, config: WalkForwardConfig) -> dict[str, Any]:
    candidate_ids = selected_summary["candidate_id"].tolist()
    candidate_changes = sum(
        candidate_ids[idx] != candidate_ids[idx - 1]
        for idx in range(1, len(candidate_ids))
    )

    return {
        "selection_metric": config.selection_metric,
        "split_count": int(len(selected_summary)),
        "candidate_count": int(config.candidate_count),
        "candidate_grid_size": int(selected_summary["candidate_id"].nunique()),
        "strategy_family": config.strategy_family,
        "strategy_name": config.strategy_name,
        "min_trades": int(config.min_trades),
        "mean_train_score": safe_float(selected_summary[f"train_{config.selection_metric}"].mean()),
        "mean_validation_score": safe_float(selected_summary[f"validation_{config.selection_metric}"].mean()),
        "mean_test_score": safe_float(selected_summary[f"test_{config.selection_metric}"].mean()),
        "median_test_score": safe_float(selected_summary[f"test_{config.selection_metric}"].median()),
        "positive_test_ratio": safe_float((selected_summary[f"test_{config.selection_metric}"] > 0).mean()),
        "test_outperformed_hold_ratio": safe_float(
            (selected_summary[f"test_{config.selection_metric}"] > selected_summary[f"test_hold_{config.selection_metric}"]).mean()
        ),
        "mean_train_validation_gap": safe_float(selected_summary["train_validation_gap"].mean()),
        "mean_validation_test_gap": safe_float(selected_summary["validation_test_gap"].mean()),
        "mean_test_return": safe_float(selected_summary["test_total_return"].mean()),
        "mean_test_hold_return": safe_float(selected_summary["test_hold_total_return"].mean()),
        "unique_candidate_count": int(selected_summary["candidate_id"].nunique()),
        "candidate_change_ratio": safe_float(candidate_changes / max(len(candidate_ids) - 1, 1)),
    }


def build_test_timeline(
    price: pd.Series,
    selected_summary: pd.DataFrame,
    test_indexes: list[pd.Index],
    candidate_lookup: dict[str, StrategyCandidate],
    config: WalkForwardConfig,
) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    strategy_multiplier = 1.0
    hold_multiplier = 1.0

    for row in selected_summary.sort_values("split_idx").itertuples(index=False):
        split_idx = int(row.split_idx)
        split_price = price.loc[test_indexes[split_idx]].copy()
        candidate = candidate_lookup[row.candidate_id]
        signals = apply_direction_to_signals(
            generate_candidate_signals(split_price, candidate.family, candidate.params),
            config.direction,
        )
        strategy_pf = vbt.Portfolio.from_signals(
            split_price,
            entries=signals["long_entries"],
            exits=signals["long_exits"],
            short_entries=signals["short_entries"],
            short_exits=signals["short_exits"],
            init_cash=config.init_cash,
            fees=config.fees,
            freq=config.freq,
        )
        hold_pf = vbt.Portfolio.from_holding(
            split_price,
            init_cash=config.init_cash,
            fees=config.fees,
            freq=config.freq,
        )

        strategy_split_return = strategy_pf.value() / config.init_cash - 1.0
        hold_split_return = hold_pf.value() / config.init_cash - 1.0

        split_frame = pd.DataFrame(
            {
                "timestamp": split_price.index,
                "split_idx": split_idx,
                "candidate_id": candidate.candidate_id,
                "family": candidate.family,
                "candidate_label": candidate.label,
                "close": split_price.to_numpy(),
                "strategy_split_return": strategy_split_return.to_numpy(),
                "hold_split_return": hold_split_return.to_numpy(),
                "strategy_cumulative_return": strategy_multiplier * (1.0 + strategy_split_return.to_numpy()) - 1.0,
                "hold_cumulative_return": hold_multiplier * (1.0 + hold_split_return.to_numpy()) - 1.0,
            }
        )
        rows.append(split_frame)

        strategy_multiplier *= 1.0 + float(strategy_split_return.iloc[-1])
        hold_multiplier *= 1.0 + float(hold_split_return.iloc[-1])

    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def build_overfitting_timeline(selected_summary: pd.DataFrame, selection_metric: str) -> pd.DataFrame:
    base_columns = [
        "split_idx",
        "candidate_id",
        "family",
        "candidate_label",
        "params_json",
        "train_start",
        "validation_start",
        "test_start",
        "test_end",
        f"train_{selection_metric}",
        f"validation_{selection_metric}",
        f"test_{selection_metric}",
        "train_validation_gap",
        "validation_test_gap",
        "test_vs_hold_gap",
    ]
    overfitting_timeline = selected_summary[base_columns].copy()
    overfitting_timeline["test_end"] = pd.to_datetime(overfitting_timeline["test_end"], utc=True)
    overfitting_timeline["test_start"] = pd.to_datetime(overfitting_timeline["test_start"], utc=True)
    overfitting_timeline["selection_metric"] = selection_metric
    return overfitting_timeline.sort_values("split_idx").reset_index(drop=True)


def default_output_dir(symbol: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_symbol = symbol.replace("/", "_").replace(":", "_")
    return DEFAULT_OUTPUT_ROOT / f"{safe_symbol}_{timestamp}"


def save_results(
    output_dir: Path,
    config: WalkForwardConfig,
    price: pd.Series,
    download_context: dict[str, Any],
    selected_summary: pd.DataFrame,
    grid_metrics: pd.DataFrame,
    test_timeline: pd.DataFrame,
    overfitting_timeline: pd.DataFrame,
    aggregate: dict[str, Any],
    candidate_meta: pd.DataFrame,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)

    selected_path = output_dir / "selected_params.csv"
    grid_path = output_dir / "grid_metrics.csv"
    test_timeline_path = output_dir / "test_timeline.csv"
    overfitting_timeline_path = output_dir / "overfitting_timeline.csv"
    summary_path = output_dir / "summary.json"

    selected_summary.to_csv(selected_path, index=False)
    grid_metrics.to_csv(grid_path, index=False)
    test_timeline.to_csv(test_timeline_path, index=False)
    overfitting_timeline.to_csv(overfitting_timeline_path, index=False)

    summary_payload = {
        "config": asdict(config),
        "strategy": {
            "strategy_name": config.strategy_name,
            "strategy_family": config.strategy_family,
            "window_spec": config.window_spec,
            "search_space": config.search_space,
            "candidate_grid_size": int(candidate_meta.shape[0]),
            "candidate_labels": candidate_meta["candidate_label"].tolist(),
        },
        "data": {
            "symbol": config.symbol,
            "data_source": download_context["data_source"],
            "exchange": download_context.get("exchange"),
            "period": download_context.get("period"),
            "requested_start": download_context.get("start"),
            "requested_end": download_context.get("end"),
            "resolved_start": download_context.get("resolved_start"),
            "resolved_end": download_context.get("resolved_end"),
            "interval": config.interval,
            "rows": int(len(price)),
            "start": str(price.index[0]),
            "end": str(price.index[-1]),
        },
        "aggregate": aggregate,
        "files": {
            "selected_params_csv": str(selected_path),
            "grid_metrics_csv": str(grid_path),
            "test_timeline_csv": str(test_timeline_path),
            "overfitting_timeline_csv": str(overfitting_timeline_path),
        },
    }
    summary_path.write_text(json.dumps(summary_payload, indent=2))

    return {
        "selected_params_csv": str(selected_path),
        "grid_metrics_csv": str(grid_path),
        "test_timeline_csv": str(test_timeline_path),
        "overfitting_timeline_csv": str(overfitting_timeline_path),
        "summary_json": str(summary_path),
    }


def run_walk_forward(config: WalkForwardConfig, output_dir: str | Path | None = None) -> dict[str, Any]:
    if config.selection_metric not in SUPPORTED_SELECTION_METRICS:
        raise ValueError(
            f"Unsupported selection metric {config.selection_metric!r}. "
            f"Choose one of {SUPPORTED_SELECTION_METRICS}."
        )

    candidates = build_strategy_candidates(
        config.strategy_family,
        interval=config.interval,
        strategy_name=config.strategy_name,
        search_space=config.search_space,
        window_spec=config.window_spec,
    )
    candidate_lookup = {candidate.candidate_id: candidate for candidate in candidates}
    candidate_meta = candidate_meta_frame(candidates)

    price, download_context = load_price_series(config)
    (_, train_indexes), (_, validation_indexes), (_, test_indexes) = split_price_series(price, config)

    train_metrics = simulate_candidate_metrics_by_splits(price, train_indexes, candidates, config)
    validation_metrics = simulate_candidate_metrics_by_splits(price, validation_indexes, candidates, config)
    test_metrics = simulate_candidate_metrics_by_splits(price, test_indexes, candidates, config)

    train_hold = simulate_holding_metrics_by_splits(price, train_indexes, config)
    validation_hold = simulate_holding_metrics_by_splits(price, validation_indexes, config)
    test_hold = simulate_holding_metrics_by_splits(price, test_indexes, config)

    selected_index, selection_status, candidate_sizes = select_best_indices(
        train_metrics=train_metrics,
        validation_metrics=validation_metrics,
        selection_metric=config.selection_metric,
        candidate_count=config.candidate_count,
        min_trades=config.min_trades,
    )

    selected_summary = build_selected_summary(
        selected_index=selected_index,
        candidate_meta=candidate_meta,
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
            flatten_metric_frame(train_metrics, "train", config.selection_metric, config.min_trades, candidate_meta),
            flatten_metric_frame(validation_metrics, "validation", config.selection_metric, config.min_trades, candidate_meta),
            flatten_metric_frame(test_metrics, "test", config.selection_metric, config.min_trades, candidate_meta),
        ],
        ignore_index=True,
    )
    test_timeline = build_test_timeline(price, selected_summary, test_indexes, candidate_lookup, config)
    overfitting_timeline = build_overfitting_timeline(selected_summary, config.selection_metric)

    aggregate = build_aggregate_summary(selected_summary, config)
    resolved_output_dir = Path(output_dir) if output_dir is not None else default_output_dir(config.symbol)
    files = save_results(
        resolved_output_dir,
        config,
        price,
        download_context,
        selected_summary,
        grid_metrics,
        test_timeline,
        overfitting_timeline,
        aggregate,
        candidate_meta,
    )

    return {
        "config": config,
        "price": price,
        "candidate_meta": candidate_meta,
        "selected_summary": selected_summary,
        "grid_metrics": grid_metrics,
        "test_timeline": test_timeline,
        "overfitting_timeline": overfitting_timeline,
        "aggregate": aggregate,
        "files": files,
    }


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Walk-forward train/validation/test engine for vectorbt strategy iteration."
    )
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--data-source", default="ccxt", choices=SUPPORTED_DATA_SOURCES)
    parser.add_argument("--exchange", default="binanceusdm")
    parser.add_argument("--period", default="540d")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--freq", default="5min")
    parser.add_argument("--train-bars", type=int, default=45 * 24 * 12)
    parser.add_argument("--validation-bars", type=int, default=15 * 24 * 12)
    parser.add_argument("--test-bars", type=int, default=15 * 24 * 12)
    parser.add_argument("--n-splits", type=int, default=6)
    parser.add_argument("--strategy-name", default="ma_crossover_ls")
    parser.add_argument("--family", default="ma_crossover_ls", choices=SUPPORTED_STRATEGY_FAMILIES)
    parser.add_argument("--windows", default=DEFAULT_WINDOW_SPEC, help="Python-style range start:stop:step or comma list.")
    parser.add_argument("--search-space-json", default=None, help="Override strategy search space as JSON.")
    parser.add_argument("--candidate-count", type=int, default=10, help="How many top train candidates reach validation.")
    parser.add_argument("--min-trades", type=int, default=1, help="Minimum trades required for a parameter set to be eligible.")
    parser.add_argument("--selection-metric", default="sharpe_ratio", choices=SUPPORTED_SELECTION_METRICS)
    parser.add_argument("--init-cash", type=float, default=1_000.0)
    parser.add_argument("--fees", type=float, default=0.00015)
    parser.add_argument("--direction", default="both", choices=("longonly", "shortonly", "both"))
    parser.add_argument("--output-dir", default=None)
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    symbol = args.symbol or DEFAULT_SYMBOL_BY_SOURCE[args.data_source]
    search_space = None if args.search_space_json is None else json.loads(args.search_space_json)
    config = WalkForwardConfig(
        symbol=symbol,
        data_source=args.data_source,
        exchange=args.exchange,
        period=args.period,
        start=args.start,
        end=args.end,
        interval=args.interval,
        freq=args.freq,
        train_bars=args.train_bars,
        validation_bars=args.validation_bars,
        test_bars=args.test_bars,
        n_splits=args.n_splits,
        strategy_name=args.strategy_name,
        strategy_family=args.family,
        search_space=search_space,
        window_spec=args.windows,
        candidate_count=args.candidate_count,
        min_trades=args.min_trades,
        selection_metric=args.selection_metric,
        init_cash=args.init_cash,
        fees=args.fees,
        direction=args.direction,
    )
    result = run_walk_forward(config, output_dir=args.output_dir)
    print(json.dumps(result["aggregate"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
