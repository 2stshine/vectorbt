from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.agent_pipeline.common import load_json


CHECK_LABELS_KO = {
    "median_test_score": "테스트 점수 중앙값",
    "positive_test_ratio": "플러스 테스트 비율",
    "test_outperformed_hold_ratio": "홀드 초과 비율",
    "mean_test_return": "평균 테스트 수익률",
    "mean_test_drawdown": "평균 테스트 낙폭",
}
METRIC_LABELS_KO = {
    "sharpe_ratio": "샤프 비율",
    "total_return": "총수익률",
}


def is_futures_context(symbol: str, exchange: str | None) -> bool:
    exchange_label = "" if exchange is None else exchange.lower()
    return ":" in symbol or "usdm" in exchange_label or "perp" in exchange_label or "future" in exchange_label


def benchmark_label(symbol: str, exchange: str | None) -> str:
    return "Long Hold" if is_futures_context(symbol, exchange) else "Buy & Hold"


def verdict_label_ko(verdict: str) -> str:
    return {
        "pass": "통과",
        "fail": "실패",
        "error": "오류",
    }.get(verdict, verdict)


def format_ratio_ko(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.1f}%"


def format_number_ko(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value:.3f}"


def format_return_ko(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.2f}%"


def format_fee_ko(value: float | None) -> str:
    if value is None:
        return "N/A"
    return f"{value * 100:.3f}%"


def relative_path_str(path: str | Path, base_dir: str | Path) -> str:
    return os.path.relpath(Path(path), Path(base_dir))


def build_failed_reasons_ko(assessment: dict[str, Any] | None) -> list[str]:
    if assessment is None:
        return ["실행 중 오류가 발생해서 성과 평가를 만들지 못했습니다."]

    failed_checks = assessment.get("failed_checks", [])
    if not failed_checks:
        return ["기본 통과 기준을 모두 만족했습니다."]

    aggregate = assessment["aggregate_snapshot"]
    rules = assessment["rules"]
    metric_label = METRIC_LABELS_KO.get(assessment["selection_metric"], assessment["selection_metric"])
    reasons: list[str] = []

    for check in failed_checks:
        if check == "median_test_score":
            reasons.append(
                f"테스트 {metric_label} 중앙값이 {format_number_ko(aggregate.get('median_test_score'))}로 "
                f"기준 {format_number_ko(rules.get('min_median_test_score'))}보다 낮았습니다."
            )
        elif check == "positive_test_ratio":
            reasons.append(
                f"플러스 테스트 비율이 {format_ratio_ko(aggregate.get('positive_test_ratio'))}로 "
                f"기준 {format_ratio_ko(rules.get('min_positive_test_ratio'))}보다 낮았습니다."
            )
        elif check == "test_outperformed_hold_ratio":
            reasons.append(
                f"홀드를 이긴 비율이 {format_ratio_ko(aggregate.get('test_outperformed_hold_ratio'))}로 "
                f"기준 {format_ratio_ko(rules.get('min_test_outperformed_hold_ratio'))}보다 낮았습니다."
            )
        elif check == "mean_test_return":
            reasons.append(
                f"평균 테스트 수익률이 {format_return_ko(aggregate.get('mean_test_return'))}로 "
                f"기준 {format_return_ko(rules.get('min_mean_test_return'))}보다 낮았습니다."
            )
        elif check == "mean_test_drawdown":
            reasons.append(
                f"평균 테스트 낙폭이 {format_return_ko(aggregate.get('mean_test_drawdown'))}로 "
                f"허용치 {format_return_ko(rules.get('max_mean_test_drawdown'))}를 넘었습니다."
            )
        else:
            reasons.append(f"{CHECK_LABELS_KO.get(check, check)} 기준을 통과하지 못했습니다.")

    return reasons


def plot_hold_comparison(selected_params: pd.DataFrame, output_path: Path) -> None:
    x = np.arange(len(selected_params))
    width = 0.35
    hold_label = benchmark_label(
        str(selected_params.attrs.get("symbol", "")),
        selected_params.attrs.get("exchange"),
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    axes[0].bar(x - width / 2, selected_params["test_total_return"] * 100, width, label="Strategy")
    axes[0].bar(x + width / 2, selected_params["test_hold_total_return"] * 100, width, label=hold_label)
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Test Return by Split")
    axes[0].set_xlabel("Split")
    axes[0].set_ylabel("Return (%)")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(selected_params["split_idx"].astype(int).tolist())
    axes[0].legend()

    axes[1].bar(x - width / 2, selected_params["test_sharpe_ratio"], width, label="Strategy")
    axes[1].bar(x + width / 2, selected_params["test_hold_sharpe_ratio"], width, label=hold_label)
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_title("Test Sharpe by Split")
    axes[1].set_xlabel("Split")
    axes[1].set_ylabel("Sharpe")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(selected_params["split_idx"].astype(int).tolist())
    axes[1].legend()

    fig.suptitle(f"Strategy vs {hold_label}", fontsize=14)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_return_timeline(test_timeline: pd.DataFrame, output_path: Path) -> None:
    timeline = test_timeline.copy()
    timeline["timestamp"] = pd.to_datetime(timeline["timestamp"], utc=True)
    timeline = timeline.sort_values(["split_idx", "timestamp"])
    hold_label = benchmark_label(
        str(timeline.attrs.get("symbol", "")),
        timeline.attrs.get("exchange"),
    )

    fig, ax = plt.subplots(figsize=(13, 5.5))
    plotted_strategy = False
    plotted_hold = False

    for split_idx, split_frame in timeline.groupby("split_idx", sort=True):
        ax.plot(
            split_frame["timestamp"],
            split_frame["strategy_cumulative_return"] * 100,
            color="#155eef",
            linewidth=2,
            label="Strategy" if not plotted_strategy else None,
        )
        ax.plot(
            split_frame["timestamp"],
            split_frame["hold_cumulative_return"] * 100,
            color="#12b76a",
            linewidth=2,
            linestyle="--",
            label=hold_label if not plotted_hold else None,
        )
        ax.axvline(split_frame["timestamp"].iloc[0], color="#98a2b3", linewidth=1, linestyle=":")
        plotted_strategy = True
        plotted_hold = True

    ax.axhline(0, color="black", linewidth=1)
    ax.set_title("Chronological Test Return Timeline")
    ax.set_xlabel("Date")
    ax.set_ylabel("Cumulative Return (%)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_overfitting_timeline(overfitting_timeline: pd.DataFrame, selection_metric: str, output_path: Path) -> None:
    timeline = overfitting_timeline.copy()
    timeline["test_end"] = pd.to_datetime(timeline["test_end"], utc=True)

    value_scale = 100.0 if selection_metric == "total_return" else 1.0
    ylabel = "Return (%)" if selection_metric == "total_return" else "Sharpe"

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)

    axes[0].plot(timeline["test_end"], timeline[f"train_{selection_metric}"] * value_scale, marker="o", label="Train")
    axes[0].plot(
        timeline["test_end"],
        timeline[f"validation_{selection_metric}"] * value_scale,
        marker="o",
        label="Validation",
    )
    axes[0].plot(timeline["test_end"], timeline[f"test_{selection_metric}"] * value_scale, marker="o", label="Test")
    axes[0].axhline(0, color="black", linewidth=1)
    axes[0].set_title("Overfitting Timeline")
    axes[0].set_ylabel(ylabel)
    axes[0].legend()

    axes[1].plot(
        timeline["test_end"],
        timeline["train_validation_gap"] * value_scale,
        marker="o",
        label="Train - Validation",
    )
    axes[1].plot(
        timeline["test_end"],
        timeline["validation_test_gap"] * value_scale,
        marker="o",
        label="Validation - Test",
    )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set_xlabel("Test End")
    axes[1].set_ylabel(ylabel)
    axes[1].legend()

    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def build_run_visuals(run_dir: str | Path, assessment: dict[str, Any] | None = None) -> dict[str, str]:
    run_dir = Path(run_dir)
    summary_payload = load_json(run_dir / "summary.json")
    selected_params = pd.read_csv(run_dir / "selected_params.csv").sort_values("split_idx")
    test_timeline = pd.read_csv(run_dir / "test_timeline.csv")
    overfitting_timeline = pd.read_csv(run_dir / "overfitting_timeline.csv")
    selected_params.attrs["symbol"] = summary_payload["data"]["symbol"]
    selected_params.attrs["exchange"] = summary_payload["data"].get("exchange")
    test_timeline.attrs["symbol"] = summary_payload["data"]["symbol"]
    test_timeline.attrs["exchange"] = summary_payload["data"].get("exchange")
    selection_metric = (
        summary_payload["aggregate"]["selection_metric"]
        if assessment is None
        else assessment["selection_metric"]
    )

    plots_dir = run_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    hold_plot_path = plots_dir / "hold_comparison.png"
    return_timeline_path = plots_dir / "return_timeline.png"
    overfitting_timeline_path = plots_dir / "overfitting_timeline.png"

    plot_hold_comparison(selected_params, hold_plot_path)
    plot_return_timeline(test_timeline, return_timeline_path)
    plot_overfitting_timeline(overfitting_timeline, selection_metric, overfitting_timeline_path)

    return {
        "hold_comparison": str(hold_plot_path),
        "return_timeline": str(return_timeline_path),
        "overfitting_timeline": str(overfitting_timeline_path),
        "hold_comparison_run_rel": relative_path_str(hold_plot_path, run_dir),
        "return_timeline_run_rel": relative_path_str(return_timeline_path, run_dir),
        "overfitting_timeline_run_rel": relative_path_str(overfitting_timeline_path, run_dir),
    }


def build_run_interpretation_ko(
    assessment: dict[str, Any],
    test_timeline: pd.DataFrame,
) -> list[str]:
    aggregate = assessment["aggregate_snapshot"]
    final_strategy_return = None if test_timeline.empty else float(test_timeline["strategy_cumulative_return"].dropna().iloc[-1])
    final_hold_return = None if test_timeline.empty else float(test_timeline["hold_cumulative_return"].dropna().iloc[-1])
    notes: list[str] = []

    if final_strategy_return is not None and final_hold_return is not None:
        if final_strategy_return > final_hold_return:
            notes.append(
                f"선택된 테스트 구간을 시간순으로 이어 보면 전략 누적 수익률이 {format_return_ko(final_strategy_return)}로 "
                f"홀드 {format_return_ko(final_hold_return)}를 웃돌았습니다."
            )
        else:
            notes.append(
                f"선택된 테스트 구간을 시간순으로 이어 보면 전략 누적 수익률이 {format_return_ko(final_strategy_return)}로 "
                f"홀드 {format_return_ko(final_hold_return)}보다 낮았습니다."
            )

    if aggregate.get("mean_train_validation_gap") is not None and aggregate.get("mean_validation_test_gap") is not None:
        train_gap = float(aggregate["mean_train_validation_gap"])
        test_gap = float(aggregate["mean_validation_test_gap"])
        if train_gap > 0 and test_gap > 0:
            notes.append(
                "train에서 좋았던 성과가 validation과 test로 갈수록 줄어드는 편이라, 현재 설정에는 오버피팅 신호가 있습니다."
            )
        else:
            notes.append(
                "train, validation, test 간 점수 격차가 아주 일방적으로 벌어지지는 않아, 오버피팅 양상은 상대적으로 약한 편입니다."
            )

    if not notes:
        notes.append("현재 실행은 요약 지표를 다시 확인해 추가 해석이 필요합니다.")
    return notes


def summarize_selected_candidates(selected_params: pd.DataFrame) -> list[str]:
    if {"candidate_label", "family"}.issubset(selected_params.columns):
        unique_rows = selected_params[["family", "candidate_label"]].drop_duplicates()
        return [
            f"{row.family}: {row.candidate_label}"
            for row in unique_rows.itertuples(index=False)
        ]
    if {"fast_window", "slow_window"}.issubset(selected_params.columns):
        return [str(values.tolist()) for values in selected_params[["fast_window", "slow_window"]].drop_duplicates().values]
    return []


def render_run_summary_ko(
    run_dir: str | Path,
    assessment: dict[str, Any],
    visuals: dict[str, str],
) -> str:
    run_dir = Path(run_dir)
    summary_payload = load_json(run_dir / "summary.json")
    config = summary_payload["config"]
    strategy = summary_payload.get("strategy", {})
    data = summary_payload["data"]
    aggregate = assessment["aggregate_snapshot"]
    selected_params = pd.read_csv(run_dir / "selected_params.csv")
    test_timeline = pd.read_csv(run_dir / "test_timeline.csv")
    selected_candidates = summarize_selected_candidates(selected_params)

    source_label = "CCXT" if data["data_source"] == "ccxt" else "Yahoo"
    exchange_suffix = f", 거래소 {data['exchange']}" if data.get("exchange") else ""
    market_label = "USDT-M perpetual futures" if is_futures_context(data["symbol"], data.get("exchange")) else "spot"
    hold_label = benchmark_label(data["symbol"], data.get("exchange"))
    family_label = strategy.get("strategy_family", config.get("strategy_family", "unknown"))
    search_space_label = strategy.get("search_space", config.get("search_space"))
    window_spec_label = strategy.get("window_spec", config.get("window_spec"))

    lines = [
        "# 전략 백테스트 요약",
        "",
        f"- 판정: {verdict_label_ko(assessment['verdict'])}",
        f"- 실행 폴더: {run_dir}",
        f"- 데이터 설정: {source_label}{exchange_suffix}, {market_label}, {data['symbol']}, {data['interval']}, 기간 {data.get('period')}, 실제 범위 {data['start']} ~ {data['end']}",
        f"- 워크포워드 설정: train {config['train_bars']} bars / validation {config['validation_bars']} bars / test {config['test_bars']} bars, split {config['n_splits']}개",
        f"- 전략 설정: family `{family_label}`, 방향 `{config['direction']}`, 선택 기준 `{METRIC_LABELS_KO.get(assessment['selection_metric'], assessment['selection_metric'])}`, 후보 {config['candidate_count']}개, 최소 거래 {config['min_trades']}회",
        f"- 탐색 공간: window `{window_spec_label}` / search space `{search_space_label}`",
        f"- 비용 가정: maker 수수료 {format_fee_ko(config['fees'])}, funding/slippage/leverage는 별도 모델링하지 않았습니다.",
        f"- 핵심 지표: 테스트 중앙값 {format_number_ko(aggregate.get('median_test_score'))}, 플러스 비율 {format_ratio_ko(aggregate.get('positive_test_ratio'))}, {hold_label} 초과 비율 {format_ratio_ko(aggregate.get('test_outperformed_hold_ratio'))}",
        f"- 보조 지표: 평균 테스트 수익률 {format_return_ko(aggregate.get('mean_test_return'))}, 평균 {hold_label} 수익률 {format_return_ko(aggregate.get('mean_test_hold_return'))}, 평균 테스트 낙폭 {format_return_ko(aggregate.get('mean_test_drawdown'))}",
        f"- 선택된 후보: {selected_candidates if selected_candidates else '없음'}",
    ]

    for note in build_run_interpretation_ko(assessment, test_timeline):
        lines.append(f"- 해석: {note}")

    for reason in build_failed_reasons_ko(assessment):
        lines.append(f"- 실패/판정 이유: {reason}")

    lines.extend(
        [
            "",
            "## 시각화",
            "",
            "### 홀드 비교",
            f"![홀드 비교]({visuals['hold_comparison_run_rel']})",
            "",
            "### 수익률 타임라인",
            f"![수익률 타임라인]({visuals['return_timeline_run_rel']})",
            "",
            "### 오버피팅 타임라인",
            f"![오버피팅 타임라인]({visuals['overfitting_timeline_run_rel']})",
        ]
    )

    return "\n".join(lines) + "\n"


def generate_run_report(run_dir: str | Path, assessment: dict[str, Any] | None = None) -> dict[str, str]:
    run_dir = Path(run_dir)
    if assessment is None:
        assessment = load_json(run_dir / "assessment.json")

    visuals = build_run_visuals(run_dir, assessment=assessment)
    summary_path = run_dir / "summary_ko.md"
    summary_path.write_text(render_run_summary_ko(run_dir, assessment, visuals))
    visuals["summary_ko_path"] = str(summary_path)
    return visuals
