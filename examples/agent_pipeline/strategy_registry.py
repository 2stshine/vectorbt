from __future__ import annotations

import json
from dataclasses import dataclass
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import vectorbt as vbt


DEFAULT_WINDOW_SPEC_BY_INTERVAL = {
    "1m": "120:961:120",
    "2m": "60:481:60",
    "5m": "6,12,18,24,36,48,72,96,144",
    "15m": "8:65:8",
    "30m": "4:33:4",
    "60m": "2:17:2",
    "1h": "2:17:2",
    "1d": "5:51:5",
}
SUPPORTED_STRATEGY_FAMILIES = (
    "ma_crossover_ls",
    "rsi_reversion_ls",
    "channel_breakout_ls",
    "ma_rsi_filter_ls",
    "bbands_reversion_ls",
    "macd_signal_ls",
    "stoch_reversion_ls",
)
FAMILY_PARAMETER_ORDER = {
    "ma_crossover_ls": ("fast_window", "slow_window"),
    "rsi_reversion_ls": (
        "rsi_window",
        "long_entry_threshold",
        "long_exit_threshold",
        "short_entry_threshold",
    ),
    "channel_breakout_ls": ("breakout_window", "exit_window"),
    "ma_rsi_filter_ls": (
        "fast_window",
        "slow_window",
        "rsi_window",
        "long_filter_max",
        "short_filter_min",
    ),
    "bbands_reversion_ls": (
        "bb_window",
        "bb_alpha",
    ),
    "macd_signal_ls": (
        "fast_window",
        "slow_window",
        "signal_window",
    ),
    "stoch_reversion_ls": (
        "k_window",
        "d_window",
        "lower_threshold",
        "upper_threshold",
    ),
}
FAMILY_DESCRIPTIONS = {
    "ma_crossover_ls": "Explicit long/short moving-average crossover strategy.",
    "rsi_reversion_ls": "RSI mean-reversion strategy that buys oversold and shorts overbought.",
    "channel_breakout_ls": "Donchian-style breakout strategy with explicit long/short channel exits.",
    "ma_rsi_filter_ls": "MA crossover entries filtered by RSI regime for long and short trades.",
    "bbands_reversion_ls": "Bollinger-band mean-reversion strategy with symmetric long/short entries.",
    "macd_signal_ls": "MACD signal-line crossover strategy for directional momentum.",
    "stoch_reversion_ls": "Stochastic-oscillator mean-reversion strategy for overbought and oversold swings.",
}


@dataclass(frozen=True)
class StrategyCandidate:
    candidate_id: str
    family: str
    strategy_name: str
    params: dict[str, int | float]

    @property
    def label(self) -> str:
        ordered_keys = FAMILY_PARAMETER_ORDER[self.family]
        return ", ".join(f"{key}={format_param_value(self.params[key])}" for key in ordered_keys)

    def to_record(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "candidate_id": self.candidate_id,
            "family": self.family,
            "strategy_name": self.strategy_name,
            "candidate_label": self.label,
            "params_json": json.dumps(self.params, sort_keys=True),
        }
        payload.update(self.params)
        return payload


def format_param_value(value: int | float) -> str:
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(int(value))


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
        raise ValueError("Need at least two valid windows.")
    return windows


def normalize_family(family: str) -> str:
    normalized = family.strip().lower()
    if normalized not in SUPPORTED_STRATEGY_FAMILIES:
        raise ValueError(
            f"Unsupported strategy family {family!r}. "
            f"Choose one of {SUPPORTED_STRATEGY_FAMILIES}."
        )
    return normalized


def default_window_spec_for_interval(interval: str) -> str:
    return DEFAULT_WINDOW_SPEC_BY_INTERVAL.get(interval, "6,12,18,24,36,48,72,96,144")


def default_search_space_for_family(
    family: str,
    interval: str,
    window_spec: str | None = None,
) -> dict[str, list[int | float]]:
    normalized_family = normalize_family(family)
    resolved_window_spec = window_spec or default_window_spec_for_interval(interval)
    windows = parse_window_spec(resolved_window_spec).tolist()

    if normalized_family == "ma_crossover_ls":
        return {
            "fast_window": windows,
            "slow_window": windows,
        }

    if normalized_family == "rsi_reversion_ls":
        return {
            "rsi_window": [7, 14, 21],
            "long_entry_threshold": [20, 25, 30],
            "long_exit_threshold": [45, 50, 55],
            "short_entry_threshold": [70, 75, 80],
        }

    if normalized_family == "channel_breakout_ls":
        return {
            "breakout_window": [24, 48, 96, 144],
            "exit_window": [6, 12, 24, 48],
        }

    if normalized_family == "ma_rsi_filter_ls":
        ma_windows = windows[: min(len(windows), 7)]
        return {
            "fast_window": ma_windows,
            "slow_window": ma_windows,
            "rsi_window": [7, 14],
            "long_filter_max": [55, 60],
            "short_filter_min": [40, 45],
        }

    if normalized_family == "bbands_reversion_ls":
        return {
            "bb_window": [14, 20, 28, 40],
            "bb_alpha": [1.5, 2.0, 2.5],
        }

    if normalized_family == "macd_signal_ls":
        ma_windows = windows[: min(len(windows), 7)]
        return {
            "fast_window": ma_windows,
            "slow_window": ma_windows,
            "signal_window": [5, 9, 12, 18],
        }

    if normalized_family == "stoch_reversion_ls":
        return {
            "k_window": [9, 14, 21],
            "d_window": [3, 5, 7],
            "lower_threshold": [15, 20, 25],
            "upper_threshold": [75, 80, 85],
        }

    raise AssertionError(f"Unhandled family: {normalized_family}")


def sanitize_search_space(
    family: str,
    search_space: dict[str, Any] | None,
    *,
    interval: str,
    window_spec: str | None = None,
) -> dict[str, list[int | float]]:
    normalized_family = normalize_family(family)
    if search_space is None:
        search_space = default_search_space_for_family(normalized_family, interval, window_spec=window_spec)

    expected_keys = FAMILY_PARAMETER_ORDER[normalized_family]
    sanitized: dict[str, list[int | float]] = {}
    for key in expected_keys:
        if key not in search_space:
            raise ValueError(
                f"Missing parameter {key!r} in search space for family {normalized_family!r}."
            )
        raw_values = search_space[key]
        if not isinstance(raw_values, list) or len(raw_values) == 0:
            raise ValueError(f"Search space {key!r} must be a non-empty list.")
        values: list[int | float] = []
        for raw_value in raw_values:
            if isinstance(raw_value, bool):
                raise ValueError(f"Boolean values are not supported in search space {key!r}.")
            if isinstance(raw_value, int):
                values.append(int(raw_value))
            elif isinstance(raw_value, float):
                values.append(float(raw_value))
            else:
                raise ValueError(
                    f"Search space {key!r} must contain only numbers, got {type(raw_value).__name__}."
                )
        unique_values = sorted(set(values))
        if len(unique_values) == 0:
            raise ValueError(f"Search space {key!r} became empty after sanitizing.")
        sanitized[key] = unique_values

    extra_keys = set(search_space) - set(expected_keys)
    if extra_keys:
        raise ValueError(
            f"Unexpected search-space parameters for family {normalized_family!r}: {sorted(extra_keys)}."
        )
    return sanitized


def strategy_family_catalog(interval: str) -> dict[str, dict[str, Any]]:
    catalog: dict[str, dict[str, Any]] = {}
    for family in SUPPORTED_STRATEGY_FAMILIES:
        defaults = default_search_space_for_family(family, interval)
        catalog[family] = {
            "description": FAMILY_DESCRIPTIONS[family],
            "parameter_order": list(FAMILY_PARAMETER_ORDER[family]),
            "default_search_space": defaults,
        }
    return catalog


def _candidate_is_valid(family: str, params: dict[str, int | float]) -> bool:
    if family == "ma_crossover_ls":
        return int(params["fast_window"]) < int(params["slow_window"])

    if family == "rsi_reversion_ls":
        return (
            float(params["long_entry_threshold"])
            < float(params["long_exit_threshold"])
            < float(params["short_entry_threshold"])
        )

    if family == "channel_breakout_ls":
        return int(params["exit_window"]) < int(params["breakout_window"])

    if family == "ma_rsi_filter_ls":
        return (
            int(params["fast_window"]) < int(params["slow_window"])
            and float(params["short_filter_min"]) < float(params["long_filter_max"])
        )

    if family == "bbands_reversion_ls":
        return float(params["bb_alpha"]) > 0

    if family == "macd_signal_ls":
        return (
            int(params["fast_window"]) < int(params["slow_window"])
            and int(params["signal_window"]) > 0
        )

    if family == "stoch_reversion_ls":
        return (
            int(params["k_window"]) > 0
            and int(params["d_window"]) > 0
            and float(params["lower_threshold"]) < float(params["upper_threshold"])
        )

    raise AssertionError(f"Unhandled family: {family}")


def build_candidate_id(family: str, params: dict[str, int | float]) -> str:
    ordered_keys = FAMILY_PARAMETER_ORDER[family]
    param_str = ",".join(f"{key}={format_param_value(params[key])}" for key in ordered_keys)
    return f"{family}|{param_str}"


def build_strategy_candidates(
    family: str,
    *,
    interval: str,
    strategy_name: str | None = None,
    search_space: dict[str, Any] | None = None,
    window_spec: str | None = None,
) -> list[StrategyCandidate]:
    normalized_family = normalize_family(family)
    strategy_name = strategy_name or normalized_family
    sanitized_search_space = sanitize_search_space(
        normalized_family,
        search_space,
        interval=interval,
        window_spec=window_spec,
    )
    ordered_keys = FAMILY_PARAMETER_ORDER[normalized_family]
    candidates: list[StrategyCandidate] = []
    for values in product(*(sanitized_search_space[key] for key in ordered_keys)):
        params = dict(zip(ordered_keys, values))
        if not _candidate_is_valid(normalized_family, params):
            continue
        candidates.append(
            StrategyCandidate(
                candidate_id=build_candidate_id(normalized_family, params),
                family=normalized_family,
                strategy_name=strategy_name,
                params={key: params[key] for key in ordered_keys},
            )
        )
    if not candidates:
        raise ValueError(
            f"Search space for family {normalized_family!r} produced no valid candidates."
        )
    return candidates


def candidate_meta_frame(candidates: list[StrategyCandidate]) -> pd.DataFrame:
    return pd.DataFrame([candidate.to_record() for candidate in candidates]).set_index("candidate_id")


def generate_candidate_signals(
    price: pd.Series,
    family: str,
    params: dict[str, int | float],
) -> dict[str, pd.Series]:
    if family == "ma_crossover_ls":
        fast_ma = vbt.MA.run(price, window=int(params["fast_window"]))
        slow_ma = vbt.MA.run(price, window=int(params["slow_window"]))
        long_entries = fast_ma.ma_crossed_above(slow_ma)
        long_exits = fast_ma.ma_crossed_below(slow_ma)
        short_entries = fast_ma.ma_crossed_below(slow_ma)
        short_exits = fast_ma.ma_crossed_above(slow_ma)
        return {
            "long_entries": long_entries,
            "long_exits": long_exits,
            "short_entries": short_entries,
            "short_exits": short_exits,
        }

    if family == "rsi_reversion_ls":
        rsi = vbt.RSI.run(price, window=int(params["rsi_window"])).rsi
        long_entries = rsi < float(params["long_entry_threshold"])
        long_exits = rsi > float(params["long_exit_threshold"])
        short_entries = rsi > float(params["short_entry_threshold"])
        short_exits = rsi < float(params["long_exit_threshold"])
        return {
            "long_entries": long_entries,
            "long_exits": long_exits,
            "short_entries": short_entries,
            "short_exits": short_exits,
        }

    if family == "channel_breakout_ls":
        breakout_window = int(params["breakout_window"])
        exit_window = int(params["exit_window"])
        upper_breakout = price.rolling(breakout_window).max().shift(1)
        lower_breakout = price.rolling(breakout_window).min().shift(1)
        upper_exit = price.rolling(exit_window).max().shift(1)
        lower_exit = price.rolling(exit_window).min().shift(1)
        long_entries = price > upper_breakout
        long_exits = price < lower_exit
        short_entries = price < lower_breakout
        short_exits = price > upper_exit
        return {
            "long_entries": long_entries.fillna(False),
            "long_exits": long_exits.fillna(False),
            "short_entries": short_entries.fillna(False),
            "short_exits": short_exits.fillna(False),
        }

    if family == "ma_rsi_filter_ls":
        fast_ma = vbt.MA.run(price, window=int(params["fast_window"]))
        slow_ma = vbt.MA.run(price, window=int(params["slow_window"]))
        rsi = vbt.RSI.run(price, window=int(params["rsi_window"])).rsi
        cross_above = fast_ma.ma_crossed_above(slow_ma)
        cross_below = fast_ma.ma_crossed_below(slow_ma)
        long_entries = cross_above & (rsi < float(params["long_filter_max"]))
        long_exits = cross_below
        short_entries = cross_below & (rsi > float(params["short_filter_min"]))
        short_exits = cross_above
        return {
            "long_entries": long_entries,
            "long_exits": long_exits,
            "short_entries": short_entries,
            "short_exits": short_exits,
        }

    if family == "bbands_reversion_ls":
        bbands = vbt.BBANDS.run(
            price,
            window=int(params["bb_window"]),
            alpha=float(params["bb_alpha"]),
        )
        long_entries = price.vbt.crossed_below(bbands.lower)
        long_exits = price.vbt.crossed_above(bbands.middle)
        short_entries = price.vbt.crossed_above(bbands.upper)
        short_exits = price.vbt.crossed_below(bbands.middle)
        return {
            "long_entries": long_entries.fillna(False),
            "long_exits": long_exits.fillna(False),
            "short_entries": short_entries.fillna(False),
            "short_exits": short_exits.fillna(False),
        }

    if family == "macd_signal_ls":
        macd = vbt.MACD.run(
            price,
            fast_window=int(params["fast_window"]),
            slow_window=int(params["slow_window"]),
            signal_window=int(params["signal_window"]),
        )
        long_entries = macd.macd_crossed_above(macd.signal)
        long_exits = macd.macd_crossed_below(macd.signal)
        short_entries = macd.macd_crossed_below(macd.signal)
        short_exits = macd.macd_crossed_above(macd.signal)
        return {
            "long_entries": long_entries.fillna(False),
            "long_exits": long_exits.fillna(False),
            "short_entries": short_entries.fillna(False),
            "short_exits": short_exits.fillna(False),
        }

    if family == "stoch_reversion_ls":
        stoch = vbt.STOCH.run(price, price, price, k_window=int(params["k_window"]), d_window=int(params["d_window"]))
        lower_threshold = float(params["lower_threshold"])
        upper_threshold = float(params["upper_threshold"])
        long_entries = stoch.percent_k < lower_threshold
        long_exits = stoch.percent_k > 50.0
        short_entries = stoch.percent_k > upper_threshold
        short_exits = stoch.percent_k < 50.0
        return {
            "long_entries": long_entries.fillna(False),
            "long_exits": long_exits.fillna(False),
            "short_entries": short_entries.fillna(False),
            "short_exits": short_exits.fillna(False),
        }

    raise AssertionError(f"Unhandled family: {family}")


def apply_direction_to_signals(
    signals: dict[str, pd.Series],
    direction: str,
) -> dict[str, pd.Series]:
    resolved_direction = direction.strip().lower()
    if resolved_direction == "both":
        return signals
    if resolved_direction == "longonly":
        return {
            "long_entries": signals["long_entries"],
            "long_exits": signals["long_exits"],
            "short_entries": pd.Series(False, index=signals["short_entries"].index),
            "short_exits": pd.Series(False, index=signals["short_exits"].index),
        }
    if resolved_direction == "shortonly":
        return {
            "long_entries": pd.Series(False, index=signals["long_entries"].index),
            "long_exits": pd.Series(False, index=signals["long_exits"].index),
            "short_entries": signals["short_entries"],
            "short_exits": signals["short_exits"],
        }
    raise ValueError(f"Unsupported direction {direction!r}.")


def simulate_candidate_metrics(
    price: pd.Series,
    candidates: list[StrategyCandidate],
    *,
    init_cash: float,
    fees: float,
    freq: str,
    direction: str,
) -> pd.DataFrame:
    long_entries: dict[str, pd.Series] = {}
    long_exits: dict[str, pd.Series] = {}
    short_entries: dict[str, pd.Series] = {}
    short_exits: dict[str, pd.Series] = {}

    for candidate in candidates:
        signals = apply_direction_to_signals(
            generate_candidate_signals(price, candidate.family, candidate.params),
            direction,
        )
        long_entries[candidate.candidate_id] = signals["long_entries"].fillna(False).astype(bool)
        long_exits[candidate.candidate_id] = signals["long_exits"].fillna(False).astype(bool)
        short_entries[candidate.candidate_id] = signals["short_entries"].fillna(False).astype(bool)
        short_exits[candidate.candidate_id] = signals["short_exits"].fillna(False).astype(bool)

    portfolio = vbt.Portfolio.from_signals(
        price,
        entries=pd.DataFrame(long_entries, index=price.index),
        exits=pd.DataFrame(long_exits, index=price.index),
        short_entries=pd.DataFrame(short_entries, index=price.index),
        short_exits=pd.DataFrame(short_exits, index=price.index),
        init_cash=init_cash,
        fees=fees,
        freq=freq,
    )
    metrics = pd.DataFrame(
        {
            "sharpe_ratio": portfolio.sharpe_ratio(),
            "total_return": portfolio.total_return(),
            "max_drawdown": portfolio.max_drawdown(),
            "trade_count": portfolio.trades.count(),
        }
    )
    metrics.index.name = "candidate_id"
    return metrics.replace([np.inf, -np.inf], np.nan)
