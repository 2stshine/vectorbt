from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.agent_pipeline.common import save_json


SUPPORTED_DATA_SOURCES = ("yahoo", "ccxt")
DEFAULT_SYMBOL_BY_SOURCE = {
    "yahoo": "BTC-USD",
    "ccxt": "BTC/USDT:USDT",
}

BARS_PER_DAY_BY_INTERVAL = {
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

FREQ_BY_INTERVAL = {
    "1m": "1min",
    "2m": "2min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "60m": "60min",
    "90m": "90min",
    "1h": "1H",
    "1d": "1D",
}


@dataclass(frozen=True)
class DataSpec:
    symbol: str = "BTC/USDT:USDT"
    data_source: str = "ccxt"
    exchange: str = "binanceusdm"
    period: str | None = "540d"
    start: str | None = None
    end: str | None = None
    interval: str = "5m"
    freq: str = "5min"
    train_days: int = 45
    validation_days: int = 15
    test_days: int = 15
    n_splits: int = 6

    @property
    def bars_per_day(self) -> int:
        try:
            return BARS_PER_DAY_BY_INTERVAL[self.interval]
        except KeyError as exc:
            raise ValueError(
                f"Unsupported interval {self.interval!r}. "
                f"Choose one of {sorted(BARS_PER_DAY_BY_INTERVAL)}."
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

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "bars_per_day": self.bars_per_day,
                "train_bars": self.train_bars,
                "validation_bars": self.validation_bars,
                "test_bars": self.test_bars,
            }
        )
        return payload


def normalize_data_source(data_source: str) -> str:
    normalized = data_source.strip().lower()
    if normalized not in SUPPORTED_DATA_SOURCES:
        raise ValueError(
            f"Unsupported data source {data_source!r}. "
            f"Choose one of {SUPPORTED_DATA_SOURCES}."
        )
    return normalized


def default_symbol_for_source(data_source: str) -> str:
    return DEFAULT_SYMBOL_BY_SOURCE[normalize_data_source(data_source)]


def default_freq_for_interval(interval: str) -> str:
    try:
        return FREQ_BY_INTERVAL[interval]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported interval {interval!r}. "
            f"Choose one of {sorted(FREQ_BY_INTERVAL)}."
        ) from exc


def build_data_spec(
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
) -> DataSpec:
    normalized_source = normalize_data_source(data_source)
    return DataSpec(
        symbol=symbol or default_symbol_for_source(normalized_source),
        data_source=normalized_source,
        exchange=exchange,
        period=period,
        start=start,
        end=end,
        interval=interval,
        freq=freq or default_freq_for_interval(interval),
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
        n_splits=n_splits,
    )


def save_data_spec(data_spec: DataSpec, output_path: str | Path) -> Path:
    return save_json(data_spec.to_dict(), output_path)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create a data specification for the walk-forward pipeline.")
    parser.add_argument("--symbol", default=None)
    parser.add_argument("--data-source", default="ccxt", choices=SUPPORTED_DATA_SOURCES)
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
    parser.add_argument("--output", default=None)
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()

    data_spec = build_data_spec(
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
    )

    if args.output:
        save_data_spec(data_spec, args.output)

    print(json.dumps(data_spec.to_dict(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
