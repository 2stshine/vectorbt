"""Run walk-forward optimization on CCXT crypto futures data.

This script mirrors the idea behind ``WalkForwardOptimization.ipynb`` but
switches the data source and defaults to a Binance USD-M futures setup that is
usable for medium-frequency crypto research.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import vectorbt as vbt


DEFAULT_MODE = "full"
DEFAULT_WINDOWS = (6, 12, 18, 24, 36, 48, 72, 96, 144)
DEFAULT_OUTPUT_ROOT = Path("examples") / "walk_forward_ccxt_runs"
PERIOD_PATTERN = re.compile(r"^\s*(\d+)\s*([A-Za-z]+)\s*$")
BARS_PER_DAY_BY_TIMEFRAME = {
    "1m": 24 * 60,
    "2m": 24 * 30,
    "5m": 24 * 12,
    "15m": 24 * 4,
    "30m": 24 * 2,
    "60m": 24,
    "90m": 16,
    "1h": 24,
    "1d": 1,
}
MODE_PRESETS = {
    "fast": {
        "period": "120d",
        "train_days": 20,
        "validation_days": 5,
        "test_days": 5,
        "n_splits": 3,
        "top_train_candidates": 3,
    },
    "full": {
        "period": "540d",
        "train_days": 45,
        "validation_days": 15,
        "test_days": 15,
        "n_splits": 6,
        "top_train_candidates": 5,
    },
}


@dataclass(frozen=True)
class WalkForwardConfig:
    mode: str = DEFAULT_MODE
    symbol: str = "BTC/USDT:USDT"
    exchange: str = "binanceusdm"
    period: str | None = "540d"
    start: str | None = None
    end: str | None = None
    timeframe: str = "5m"
    freq: str = "5min"
    train_days: int = 45
    validation_days: int = 15
    test_days: int = 15
    n_splits: int = 6
    windows: tuple[int, ...] = DEFAULT_WINDOWS
    top_train_candidates: int = 5
    direction: str = "both"
    fees: float = 0.00015
    init_cash: float = 1_000.0
    show_progress: bool = False

    @property
    def bars_per_day(self) -> int:
        try:
            return BARS_PER_DAY_BY_TIMEFRAME[self.timeframe]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported timeframe {self.timeframe!r}. "
                f"Choose one of {sorted(BARS_PER_DAY_BY_TIMEFRAME)}."
            ) from exc

    @property
    def train_bars(self) -> int:
        return self.train_days * self.bars_per_day

    @property
    def validation_bars(self) -> int:
        return self.validation_days * self.bars_per_day

    @property
    def test_bars(self) -> int:
        return self.test_days * self.bars_per_day

    @property
    def window_len(self) -> int:
        return self.train_bars + self.validation_bars + self.test_bars

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "bars_per_day": self.bars_per_day,
                "train_bars": self.train_bars,
                "validation_bars": self.validation_bars,
                "test_bars": self.test_bars,
                "window_len": self.window_len,
            }
        )
        return payload


def parse_datetime_utc(value: str | None) -> pd.Timestamp | None:
    if value is None:
        return None

    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        return timestamp.tz_localize("UTC")
    return timestamp.tz_convert("UTC")


def parse_period_offset(period: str) -> pd.DateOffset | pd.Timedelta:
    match = PERIOD_PATTERN.fullmatch(period)
    if match is None:
        raise ValueError(
            f"Unsupported period {period!r}. "
            "Use forms like '540d', '12h', '6mo', or '2y'."
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


def resolve_history_range(config: WalkForwardConfig) -> tuple[pd.Timestamp, pd.Timestamp]:
    resolved_end = parse_datetime_utc(config.end) or pd.Timestamp.now(tz="UTC").floor("min")
    resolved_start = parse_datetime_utc(config.start)

    if resolved_start is None:
        if config.period is None:
            raise ValueError("Either period or start must be provided.")
        resolved_start = resolved_end - parse_period_offset(config.period)

    if resolved_start >= resolved_end:
        raise ValueError(
            f"Resolved start {resolved_start} must be earlier than end {resolved_end}."
        )

    return resolved_start, resolved_end


def parse_windows(value: str) -> tuple[int, ...]:
    windows = tuple(sorted({int(piece.strip()) for piece in value.split(",") if piece.strip()}))
    if len(windows) < 2:
        raise ValueError("Provide at least two moving-average windows.")
    return windows


def resolve_mode_defaults(mode: str) -> dict[str, object]:
    try:
        return MODE_PRESETS[mode]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported mode {mode!r}. Choose one of {sorted(MODE_PRESETS)}."
        ) from exc


def with_mode_default(value: object | None, default: object) -> object:
    if value is None:
        return default
    return value


def ensure_single_symbol_series(price: pd.Series | pd.DataFrame, symbol: str) -> pd.Series:
    if isinstance(price, pd.DataFrame):
        if price.shape[1] != 1:
            raise ValueError("This script expects a single symbol per run.")
        price = price.iloc[:, 0]

    price = price.dropna()
    price.name = symbol
    if price.empty:
        raise ValueError(f"No price rows were downloaded for {symbol}.")
    return price


def load_price_series(config: WalkForwardConfig) -> tuple[pd.Series, dict[str, str]]:
    resolved_start, resolved_end = resolve_history_range(config)

    try:
        price = vbt.CCXTData.download(
            config.symbol,
            exchange=config.exchange,
            timeframe=config.timeframe,
            start=resolved_start,
            end=resolved_end,
            limit=1000,
            show_progress=config.show_progress,
            missing_index="drop",
        ).get("Close")
    except ModuleNotFoundError as exc:
        if exc.name == "ccxt":
            raise RuntimeError(
                "CCXT data source requires the `ccxt` package. "
                "Install it in the active environment first."
            ) from exc
        raise

    return ensure_single_symbol_series(price, config.symbol), {
        "resolved_start": resolved_start.isoformat(),
        "resolved_end": resolved_end.isoformat(),
    }


def split_price_series(price: pd.Series, config: WalkForwardConfig):
    if len(price) < config.window_len:
        raise ValueError(
            f"Not enough rows for one split: need at least {config.window_len}, got {len(price)}."
        )

    return price.vbt.rolling_split(
        n=config.n_splits,
        window_len=config.window_len,
        set_lens=(config.validation_bars, config.test_bars),
        left_to_right=False,
    )


def simulate_holding(price: pd.Series | pd.DataFrame, config: WalkForwardConfig) -> pd.Series:
    pf = vbt.Portfolio.from_holding(
        price,
        init_cash=config.init_cash,
        fees=config.fees,
        freq=config.freq,
    )
    return pf.sharpe_ratio()


def simulate_all_params(
    price: pd.Series | pd.DataFrame,
    config: WalkForwardConfig,
) -> pd.Series:
    fast_ma, slow_ma = vbt.MA.run_combs(
        price,
        np.asarray(config.windows),
        r=2,
        short_names=["fast", "slow"],
    )
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        init_cash=config.init_cash,
        fees=config.fees,
        freq=config.freq,
        direction=config.direction,
    )
    return pf.sharpe_ratio()


def simulate_selected_params(
    price: pd.Series | pd.DataFrame,
    fast_windows: np.ndarray,
    slow_windows: np.ndarray,
    config: WalkForwardConfig,
) -> pd.Series:
    fast_ma = vbt.MA.run(price, window=fast_windows, per_column=True)
    slow_ma = vbt.MA.run(price, window=slow_windows, per_column=True)
    entries = fast_ma.ma_crossed_above(slow_ma)
    exits = fast_ma.ma_crossed_below(slow_ma)
    pf = vbt.Portfolio.from_signals(
        price,
        entries,
        exits,
        init_cash=config.init_cash,
        fees=config.fees,
        freq=config.freq,
        direction=config.direction,
    )
    return pf.sharpe_ratio()


def get_best_index(performance: pd.Series, higher_better: bool = True) -> pd.MultiIndex:
    if higher_better:
        return performance[performance.groupby("split_idx").idxmax()].index
    return performance[performance.groupby("split_idx").idxmin()].index


def get_top_n_index(
    performance: pd.Series,
    n: int,
    higher_better: bool = True,
) -> pd.MultiIndex:
    metric_name = performance.name or "metric"
    perf_df = performance.rename(metric_name).reset_index()
    perf_df = perf_df.dropna(subset=[metric_name])
    perf_df = perf_df.sort_values(
        ["split_idx", metric_name],
        ascending=[True, not higher_better],
    )
    top_df = perf_df.groupby("split_idx", as_index=False).head(n)
    return pd.MultiIndex.from_tuples(
        [
            tuple(record)
            for record in top_df[["fast_window", "slow_window", "split_idx"]].itertuples(index=False, name=None)
        ],
        names=performance.index.names,
    )


def get_best_params(best_index: pd.MultiIndex, level_name: str) -> np.ndarray:
    return best_index.get_level_values(level_name).to_numpy()


def safe_symbol_name(symbol: str) -> str:
    return symbol.replace("/", "_").replace(":", "_")


def build_output_dir(root: Path, symbol: str) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    output_dir = root / f"{safe_symbol_name(symbol)}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir


def write_plot(fig, path: Path) -> None:
    fig.write_html(str(path), include_plotlyjs="cdn")


def write_table_html(
    df: pd.DataFrame,
    path: Path,
    title: str,
    *,
    index: bool = False,
    float_precision: int = 4,
) -> None:
    display_df = df.copy()
    float_cols = display_df.select_dtypes(include=["float", "float64", "float32"]).columns
    if len(float_cols) > 0:
        display_df[float_cols] = display_df[float_cols].round(float_precision)

    table_html = display_df.to_html(
        index=index,
        border=0,
        classes=["results-table"],
        justify="center",
    )
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      background: #f6f8fb;
      color: #152033;
    }}
    h1 {{
      margin: 0 0 16px;
      font-size: 22px;
    }}
    .table-wrap {{
      overflow-x: auto;
      background: #ffffff;
      border: 1px solid #d8e0ea;
      border-radius: 12px;
      padding: 12px;
      box-shadow: 0 8px 24px rgba(15, 23, 42, 0.06);
    }}
    table.results-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 14px;
    }}
    table.results-table thead th {{
      position: sticky;
      top: 0;
      background: #eef4fb;
      border-bottom: 1px solid #c8d4e3;
      padding: 10px 12px;
      text-align: right;
      white-space: nowrap;
    }}
    table.results-table tbody td {{
      border-top: 1px solid #edf1f5;
      padding: 9px 12px;
      text-align: right;
      white-space: nowrap;
    }}
    table.results-table tbody tr:nth-child(even) {{
      background: #fafcff;
    }}
    table.results-table thead th:first-child,
    table.results-table tbody td:first-child {{
      text-align: left;
    }}
  </style>
</head>
<body>
  <h1>{title}</h1>
  <div class="table-wrap">
    {table_html}
  </div>
</body>
</html>
"""
    path.write_text(html, encoding="utf-8")


def run_walk_forward(config: WalkForwardConfig, output_dir: Path) -> dict[str, object]:
    price, download_context = load_price_series(config)
    (train_price, _), (validation_price, _), (test_price, _) = split_price_series(price, config)

    train_hold = simulate_holding(train_price, config)
    validation_hold = simulate_holding(validation_price, config)
    test_hold = simulate_holding(test_price, config)

    train_sharpe = simulate_all_params(train_price, config)
    validation_sharpe = simulate_all_params(validation_price, config)

    train_best_index = get_best_index(train_sharpe)
    train_top_index = get_top_n_index(train_sharpe, config.top_train_candidates)
    validation_selected_index = get_best_index(validation_sharpe.loc[train_top_index])

    selected_fast_windows = get_best_params(validation_selected_index, "fast_window")
    selected_slow_windows = get_best_params(validation_selected_index, "slow_window")
    test_selected = simulate_selected_params(
        test_price,
        selected_fast_windows,
        selected_slow_windows,
        config,
    )

    selected_params_df = pd.DataFrame(
        {
            "split_idx": validation_selected_index.get_level_values("split_idx"),
            "fast_window": selected_fast_windows,
            "slow_window": selected_slow_windows,
            "train_sharpe": train_sharpe.loc[validation_selected_index].values,
            "validation_sharpe": validation_sharpe.loc[validation_selected_index].values,
            "test_sharpe": test_selected.values,
            "test_hold_sharpe": test_hold.values,
            "test_minus_hold": test_selected.values - test_hold.values,
        }
    )

    cv_results_df = pd.DataFrame(
        {
            "train_hold": train_hold.values,
            "train_median": train_sharpe.groupby("split_idx").median().values,
            "train_best": train_sharpe[train_best_index].values,
            "validation_hold": validation_hold.values,
            "validation_median": validation_sharpe.groupby("split_idx").median().values,
            "validation_selected": validation_sharpe[validation_selected_index].values,
            "test_hold": test_hold.values,
            "test_selected": test_selected.values,
        }
    )
    cv_results_df["test_minus_hold"] = cv_results_df["test_selected"] - cv_results_df["test_hold"]
    cv_results_df.index.name = "split_idx"

    config_path = output_dir / "config.json"
    selected_params_path = output_dir / "selected_params.csv"
    cv_results_path = output_dir / "cv_results.csv"
    selected_params_table_path = output_dir / "selected_params_table.html"
    cv_results_table_path = output_dir / "cv_results_table.html"
    price_path = output_dir / "close_price.csv"
    split_plot_path = output_dir / "rolling_split.html"
    results_plot_path = output_dir / "cv_results.html"
    selected_plot_path = output_dir / "selected_windows.html"
    summary_path = output_dir / "summary.json"

    config_payload = config.to_dict()
    config_payload["download_context"] = download_context
    config_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
    price.to_csv(price_path, header=True)
    selected_params_df.to_csv(selected_params_path, index=False)
    cv_results_df.to_csv(cv_results_path)
    write_table_html(selected_params_df, selected_params_table_path, "Selected Parameters by Split")
    write_table_html(cv_results_df.reset_index(), cv_results_table_path, "Walk-Forward Results by Split")

    split_fig = price.vbt.rolling_split(
        n=config.n_splits,
        window_len=config.window_len,
        set_lens=(config.validation_bars, config.test_bars),
        left_to_right=False,
        plot=True,
        trace_names=["train", "validation", "test"],
    )
    write_plot(split_fig, split_plot_path)

    color_schema = vbt.settings["plotting"]["color_schema"]
    results_fig = cv_results_df.vbt.plot(
        trace_kwargs=[
            dict(line_color=color_schema["blue"]),
            dict(line_color=color_schema["blue"], line_dash="dash"),
            dict(line_color=color_schema["blue"], line_dash="dot"),
            dict(line_color=color_schema["green"]),
            dict(line_color=color_schema["green"], line_dash="dash"),
            dict(line_color=color_schema["green"], line_dash="dot"),
            dict(line_color=color_schema["orange"]),
            dict(line_color=color_schema["orange"], line_dash="dot"),
            dict(line_color=color_schema["red"]),
        ]
    )
    write_plot(results_fig, results_plot_path)

    selected_fig = selected_params_df.set_index("split_idx")[
        ["fast_window", "slow_window"]
    ].vbt.plot()
    write_plot(selected_fig, selected_plot_path)

    summary = {
        "mode": config.mode,
        "symbol": config.symbol,
        "exchange": config.exchange,
        "timeframe": config.timeframe,
        "period": config.period,
        "resolved_start": download_context["resolved_start"],
        "resolved_end": download_context["resolved_end"],
        "rows": int(price.shape[0]),
        "splits": int(config.n_splits),
        "median_test_selected": float(np.nanmedian(test_selected.values)),
        "median_test_hold": float(np.nanmedian(test_hold.values)),
        "median_test_minus_hold": float(np.nanmedian(cv_results_df["test_minus_hold"].values)),
        "test_outperform_ratio": float((cv_results_df["test_minus_hold"] > 0).mean()),
        "mean_selected_fast_window": float(selected_params_df["fast_window"].mean()),
        "mean_selected_slow_window": float(selected_params_df["slow_window"].mean()),
        "files": {
            "config": str(config_path),
            "close_price_csv": str(price_path),
            "selected_params_csv": str(selected_params_path),
            "cv_results_csv": str(cv_results_path),
            "selected_params_table_html": str(selected_params_table_path),
            "cv_results_table_html": str(cv_results_table_path),
            "rolling_split_html": str(split_plot_path),
            "cv_results_html": str(results_plot_path),
            "selected_windows_html": str(selected_plot_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run walk-forward optimization on CCXT crypto futures data.",
    )
    parser.add_argument("--mode", default=DEFAULT_MODE, choices=tuple(MODE_PRESETS))
    parser.add_argument("--symbol", default="BTC/USDT:USDT")
    parser.add_argument("--exchange", default="binanceusdm")
    parser.add_argument("--period", default=None)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--freq", default="5min")
    parser.add_argument("--train-days", type=int, default=None)
    parser.add_argument("--validation-days", type=int, default=None)
    parser.add_argument("--test-days", type=int, default=None)
    parser.add_argument("--n-splits", type=int, default=None)
    parser.add_argument("--windows", default="6,12,18,24,36,48,72,96,144")
    parser.add_argument("--top-train-candidates", type=int, default=None)
    parser.add_argument("--direction", default="both", choices=("longonly", "shortonly", "both"))
    parser.add_argument("--fees", type=float, default=0.00015)
    parser.add_argument("--init-cash", type=float, default=1_000.0)
    parser.add_argument("--show-progress", action="store_true")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    mode_defaults = resolve_mode_defaults(args.mode)

    config = WalkForwardConfig(
        mode=args.mode,
        symbol=args.symbol,
        exchange=args.exchange,
        period=with_mode_default(args.period, mode_defaults["period"]),
        start=args.start,
        end=args.end,
        timeframe=args.timeframe,
        freq=args.freq,
        train_days=with_mode_default(args.train_days, mode_defaults["train_days"]),
        validation_days=with_mode_default(args.validation_days, mode_defaults["validation_days"]),
        test_days=with_mode_default(args.test_days, mode_defaults["test_days"]),
        n_splits=with_mode_default(args.n_splits, mode_defaults["n_splits"]),
        windows=parse_windows(args.windows),
        top_train_candidates=with_mode_default(
            args.top_train_candidates,
            mode_defaults["top_train_candidates"],
        ),
        direction=args.direction,
        fees=args.fees,
        init_cash=args.init_cash,
        show_progress=args.show_progress,
    )

    output_dir = build_output_dir(Path(args.output_root), config.symbol)
    summary = run_walk_forward(config, output_dir)
    print(json.dumps({"output_dir": str(output_dir), **summary}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
