from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.agent_pipeline.common import load_json, save_json, timestamped_run_dir, update_latest_pointer
from examples.agent_pipeline.data_stage import SUPPORTED_DATA_SOURCES, default_symbol_for_source
from examples.agent_pipeline.llm_strategy import (
    DEFAULT_OPENAI_BASE_URL,
    DEFAULT_OPENAI_MODEL,
    LLMStrategySettings,
    llm_key_available,
    request_llm_strategy_proposal,
)
from examples.agent_pipeline.reporting import generate_run_report
from examples.agent_pipeline.run_once import run_pipeline_once
from examples.agent_pipeline.strategy_registry import (
    SUPPORTED_STRATEGY_FAMILIES,
    default_search_space_for_family,
    default_window_spec_for_interval,
    normalize_family,
    sanitize_search_space,
)


DEFAULT_OUTPUT_ROOT = Path("examples") / "agent_loop_runs"
DEFAULT_MAX_ITERATIONS = 12
CHECK_LABELS_KO = {
    "median_test_score": "테스트 점수 중앙값",
    "positive_test_ratio": "플러스 테스트 비율",
    "test_outperformed_hold_ratio": "홀드 초과 비율",
    "mean_test_return": "평균 테스트 수익률",
    "mean_test_drawdown": "평균 테스트 낙폭",
}
STOP_REASON_LABELS_KO = {
    "passed": "통과 조건을 만족해서 종료했습니다.",
    "no_new_candidate": "새로운 후보 설정을 더 만들지 못해서 종료했습니다.",
    "max_iterations_reached": "최대 반복 횟수에 도달해서 종료했습니다.",
}
METRIC_LABELS_KO = {
    "sharpe_ratio": "샤프 비율",
    "total_return": "총수익률",
}


@dataclass(frozen=True)
class LoopProposal:
    symbol: str = "BTC/USDT:USDT"
    data_source: str = "ccxt"
    exchange: str = "binanceusdm"
    period: str | None = "540d"
    start: str | None = None
    end: str | None = None
    interval: str = "5m"
    freq: str | None = None
    train_days: int = 45
    validation_days: int = 15
    test_days: int = 15
    n_splits: int = 6
    family: str = "ma_crossover_ls"
    strategy_name: str = "ma_crossover_ls"
    window_spec: str | None = "6,12,18,24,36,48,72,96,144"
    search_space: dict[str, list[int | float]] | None = None
    candidate_count: int = 8
    min_trades: int = 1
    selection_metric: str = "sharpe_ratio"
    direction: str = "both"
    fees: float = 0.00015
    init_cash: float = 1_000.0
    llm_enabled: bool = False
    llm_model: str = DEFAULT_OPENAI_MODEL
    llm_api_key_env: str = "OPENAI_API_KEY"
    llm_base_url: str = DEFAULT_OPENAI_BASE_URL
    rationale: str = "Baseline proposal."
    mutation_notes: list[str] = field(default_factory=list)

    def to_pipeline_kwargs(self) -> dict[str, Any]:
        return {
            "symbol": self.symbol,
            "data_source": self.data_source,
            "exchange": self.exchange,
            "period": self.period,
            "start": self.start,
            "end": self.end,
            "interval": self.interval,
            "freq": self.freq,
            "train_days": self.train_days,
            "validation_days": self.validation_days,
            "test_days": self.test_days,
            "n_splits": self.n_splits,
            "family": self.family,
            "strategy_name": self.strategy_name,
            "window_spec": self.window_spec,
            "search_space": self.search_space,
            "candidate_count": self.candidate_count,
            "min_trades": self.min_trades,
            "selection_metric": self.selection_metric,
            "direction": self.direction,
            "fees": self.fees,
            "init_cash": self.init_cash,
        }

    def signature(self) -> tuple[Any, ...]:
        payload = self.to_pipeline_kwargs().copy()
        payload["search_space"] = json.dumps(payload["search_space"], sort_keys=True)
        return tuple(payload[key] for key in sorted(payload))


def proposal_from_record(record: dict[str, Any]) -> LoopProposal:
    return LoopProposal(**record["proposal"])


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


def relative_path_str(path: str | Path, base_dir: str | Path) -> str:
    return os.path.relpath(Path(path), Path(base_dir))


def describe_history_request_ko(proposal: LoopProposal) -> str:
    source_label = "CCXT" if proposal.data_source == "ccxt" else "Yahoo"
    market_label = "선물" if ":" in proposal.symbol or "usdm" in proposal.exchange.lower() else "현물"
    exchange_suffix = f", 거래소 {proposal.exchange}" if proposal.data_source == "ccxt" else ""
    if proposal.start is not None or proposal.end is not None:
        start_label = proposal.start or "자동 계산"
        end_label = proposal.end or "현재 시각"
        return (
            f"{source_label}{exchange_suffix}, {market_label}, {proposal.symbol}, {proposal.interval}, "
            f"범위 {start_label} ~ {end_label}"
        )
    return f"{source_label}{exchange_suffix}, {market_label}, {proposal.symbol}, {proposal.interval}, 기간 {proposal.period}"


def search_space_label_ko(search_space: dict[str, list[int | float]] | None) -> str:
    if not search_space:
        return "없음"
    return json.dumps(search_space, ensure_ascii=False, sort_keys=True)


def describe_proposal_changes_ko(previous: LoopProposal | None, current: LoopProposal) -> list[str]:
    if previous is None:
        return ["첫 실험이라 기준 설정으로 시작했습니다."]

    changes: list[str] = []
    if previous.family != current.family:
        changes.append(f"전략 패밀리를 `{previous.family}`에서 `{current.family}`로 바꿨습니다.")
    if previous.strategy_name != current.strategy_name:
        changes.append(f"전략 이름을 `{previous.strategy_name}`에서 `{current.strategy_name}`로 조정했습니다.")
    if previous.symbol != current.symbol:
        changes.append(f"심볼을 `{previous.symbol}`에서 `{current.symbol}`로 바꿨습니다.")
    if previous.period != current.period or previous.start != current.start or previous.end != current.end:
        changes.append(
            f"조회 범위를 `{describe_history_request_ko(previous)}`에서 "
            f"`{describe_history_request_ko(current)}`로 바꿨습니다."
        )
    if previous.window_spec != current.window_spec:
        changes.append(f"윈도우 기준 문자열을 `{previous.window_spec}`에서 `{current.window_spec}`로 바꿨습니다.")
    if previous.search_space != current.search_space:
        changes.append(
            f"탐색 공간을 `{search_space_label_ko(previous.search_space)}`에서 "
            f"`{search_space_label_ko(current.search_space)}`로 조정했습니다."
        )
    if previous.selection_metric != current.selection_metric:
        changes.append(
            f"검증 선택 기준을 `{METRIC_LABELS_KO.get(previous.selection_metric, previous.selection_metric)}`에서 "
            f"`{METRIC_LABELS_KO.get(current.selection_metric, current.selection_metric)}`로 바꿨습니다."
        )
    if previous.candidate_count != current.candidate_count:
        changes.append(f"train 상위 후보 수를 {previous.candidate_count}개에서 {current.candidate_count}개로 조정했습니다.")
    if previous.min_trades != current.min_trades:
        changes.append(f"최소 거래 수 기준을 {previous.min_trades}회에서 {current.min_trades}회로 조정했습니다.")
    if previous.train_days != current.train_days or previous.validation_days != current.validation_days or previous.test_days != current.test_days:
        changes.append(
            f"구간 길이를 train {previous.train_days}/{previous.validation_days}/{previous.test_days}일에서 "
            f"{current.train_days}/{current.validation_days}/{current.test_days}일로 바꿨습니다."
        )
    if previous.direction != current.direction:
        changes.append(f"포지션 방향을 `{previous.direction}`에서 `{current.direction}`로 바꿨습니다.")
    if previous.fees != current.fees:
        changes.append(f"수수료 가정을 {previous.fees:.4f}에서 {current.fees:.4f}로 바꿨습니다.")

    if not changes:
        changes.append("직전 실험과 같은 큰 설정을 유지하고 다시 평가했습니다.")
    return changes


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


def selected_candidates_ko(assessment: dict[str, Any] | None) -> list[str]:
    if assessment is None:
        return []
    records = assessment.get("selected_candidates", [])
    lines: list[str] = []
    for record in records:
        family = record.get("family", "unknown")
        label = record.get("candidate_label", "unknown")
        lines.append(f"{family}: {label}")
    return lines


def render_iteration_summary_ko(
    record: dict[str, Any],
    previous_record: dict[str, Any] | None = None,
    best_record: dict[str, Any] | None = None,
) -> str:
    proposal = proposal_from_record(record)
    previous_proposal = None if previous_record is None else proposal_from_record(previous_record)
    assessment = record.get("assessment")
    aggregate = None if assessment is None else assessment.get("aggregate_snapshot", {})
    selected_candidates = selected_candidates_ko(assessment)

    lines = [
        f"# Iteration {record['iteration']} 요약",
        "",
        f"- 판정: {verdict_label_ko(record['verdict'])}",
        f"- 실행 폴더: {record.get('run_dir') or '실행 실패'}",
        f"- 데이터 설정: {describe_history_request_ko(proposal)}",
        f"- 구간 설정: train {proposal.train_days}일 / validation {proposal.validation_days}일 / test {proposal.test_days}일, split {proposal.n_splits}개",
        f"- 전략 설정: family `{proposal.family}`, name `{proposal.strategy_name}`, 탐색 공간 `{search_space_label_ko(proposal.search_space)}`, 선택 기준 `{METRIC_LABELS_KO.get(proposal.selection_metric, proposal.selection_metric)}`, 후보 {proposal.candidate_count}개, 최소 거래 {proposal.min_trades}회",
        f"- 비용/집행 가정: 방향 `{proposal.direction}`, maker 수수료 {proposal.fees * 100:.3f}%, funding/slippage/leverage는 별도 모델링하지 않았습니다.",
    ]

    if proposal.llm_enabled:
        llm_status = "활성"
        if not llm_key_available(
            LLMStrategySettings(
                enabled=True,
                model=proposal.llm_model,
                api_key_env=proposal.llm_api_key_env,
                base_url=proposal.llm_base_url,
            )
        ):
            llm_status = "API 키 없음으로 휴리스틱 폴백"
        lines.append(f"- LLM 제안기: {llm_status}, 모델 `{proposal.llm_model}`")

    for change in describe_proposal_changes_ko(previous_proposal, proposal):
        lines.append(f"- 변경 설명: {change}")

    if record.get("rationale"):
        lines.append(f"- 다음 전략 제안 배경: {record['rationale']}")
    for note in record.get("proposal", {}).get("mutation_notes", []):
        lines.append(f"- 변이 메모: {note}")

    if assessment is not None:
        lines.extend(
            [
                f"- 핵심 지표: 테스트 중앙값 {format_number_ko(aggregate.get('median_test_score'))}, "
                f"플러스 비율 {format_ratio_ko(aggregate.get('positive_test_ratio'))}, "
                f"홀드 초과 비율 {format_ratio_ko(aggregate.get('test_outperformed_hold_ratio'))}",
                f"- 보조 지표: 평균 테스트 수익률 {format_return_ko(aggregate.get('mean_test_return'))}, "
                f"평균 테스트 낙폭 {format_return_ko(aggregate.get('mean_test_drawdown'))}",
                f"- 선택된 후보: {selected_candidates if selected_candidates else '없음'}",
            ]
        )
    if record.get("error_message") is not None:
        lines.append(f"- 오류 내용: {record['error_message']}")

    for reason in build_failed_reasons_ko(assessment):
        lines.append(f"- 실패/판정 이유: {reason}")

    if best_record is not None:
        lines.append(
            f"- 현재까지 최고 iteration: {best_record['iteration']} "
            f"({verdict_label_ko(best_record['verdict'])})"
        )

    visuals = record.get("visuals", {})
    if visuals:
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


def save_iteration_summary_ko(
    loop_dir: Path,
    record: dict[str, Any],
    previous_record: dict[str, Any] | None = None,
    best_record: dict[str, Any] | None = None,
) -> str:
    target_dir = loop_dir if record.get("run_dir") is None else Path(record["run_dir"])
    output_path = target_dir / "loop_iteration_summary_ko.md"
    output_path.write_text(render_iteration_summary_ko(record, previous_record=previous_record, best_record=best_record))
    return str(output_path)


def render_loop_summary_ko(summary: dict[str, Any]) -> str:
    lines = [
        "# 루프 요약",
        "",
        f"- 총 반복 횟수: {summary['iterations_completed']}",
        f"- 종료 이유: {STOP_REASON_LABELS_KO.get(summary['stop_reason'], summary['stop_reason'])}",
        f"- 최고 iteration: {summary['best_iteration']}",
        f"- 최고 실행 폴더: {summary['best_run_dir']}",
        f"- 최고 판정: {verdict_label_ko(summary['best_verdict']) if summary['best_verdict'] is not None else 'N/A'}",
    ]

    for record in summary["history"]:
        assessment = record.get("assessment")
        aggregate = None if assessment is None else assessment.get("aggregate_snapshot", {})
        lines.append(
            f"- Iteration {record['iteration']}: {verdict_label_ko(record['verdict'])}, "
            f"family `{record['proposal'].get('family')}`, "
            f"중앙값 {format_number_ko(None if aggregate is None else aggregate.get('median_test_score'))}, "
            f"플러스 비율 {format_ratio_ko(None if aggregate is None else aggregate.get('positive_test_ratio'))}, "
            f"홀드 초과 비율 {format_ratio_ko(None if aggregate is None else aggregate.get('test_outperformed_hold_ratio'))}"
        )
        if record.get("summary_ko_loop_rel"):
            lines.append(f"  자세히 보기: [{record['iteration']}회차 요약]({record['summary_ko_loop_rel']})")

    best_iteration = summary.get("best_iteration")
    if best_iteration is not None:
        best_record = next((record for record in summary["history"] if record["iteration"] == best_iteration), None)
        if best_record is not None and best_record.get("visuals"):
            visuals = best_record["visuals"]
            lines.extend(
                [
                    "",
                    "## 최고 iteration 시각화",
                    "",
                    "### 홀드 비교",
                    f"![최고 iteration 홀드 비교]({visuals['hold_comparison_loop_rel']})",
                    "",
                    "### 수익률 타임라인",
                    f"![최고 iteration 수익률 타임라인]({visuals['return_timeline_loop_rel']})",
                    "",
                    "### 오버피팅 타임라인",
                    f"![최고 iteration 오버피팅 타임라인]({visuals['overfitting_timeline_loop_rel']})",
                ]
            )

    return "\n".join(lines) + "\n"


def generate_iteration_visuals(
    loop_dir: Path,
    record: dict[str, Any],
) -> dict[str, str]:
    run_dir = record.get("run_dir")
    assessment = record.get("assessment")
    if run_dir is None or assessment is None:
        return {}

    run_dir_path = Path(run_dir)
    hold_plot_path = run_dir_path / "plots" / "hold_comparison.png"
    return_timeline_path = run_dir_path / "plots" / "return_timeline.png"
    overfitting_timeline_path = run_dir_path / "plots" / "overfitting_timeline.png"
    if not hold_plot_path.exists() or not return_timeline_path.exists() or not overfitting_timeline_path.exists():
        generate_run_report(run_dir_path, assessment=assessment)

    return {
        "hold_comparison": str(hold_plot_path),
        "return_timeline": str(return_timeline_path),
        "overfitting_timeline": str(overfitting_timeline_path),
        "hold_comparison_run_rel": relative_path_str(hold_plot_path, run_dir_path),
        "return_timeline_run_rel": relative_path_str(return_timeline_path, run_dir_path),
        "overfitting_timeline_run_rel": relative_path_str(overfitting_timeline_path, run_dir_path),
        "hold_comparison_loop_rel": relative_path_str(hold_plot_path, loop_dir),
        "return_timeline_loop_rel": relative_path_str(return_timeline_path, loop_dir),
        "overfitting_timeline_loop_rel": relative_path_str(overfitting_timeline_path, loop_dir),
    }


def infer_step(values: list[int | float]) -> float:
    sorted_values = sorted(set(float(value) for value in values))
    if len(sorted_values) < 2:
        return 1.0
    diffs = [sorted_values[idx] - sorted_values[idx - 1] for idx in range(1, len(sorted_values))]
    diffs = [diff for diff in diffs if diff > 0]
    return min(diffs) if diffs else 1.0


def infer_window_spec_from_search_space(family: str, search_space: dict[str, list[int | float]] | None) -> str | None:
    if search_space is None or family not in {"ma_crossover_ls", "ma_rsi_filter_ls"}:
        return None
    window_values: set[int] = set()
    for key in ("fast_window", "slow_window"):
        for value in search_space.get(key, []):
            window_values.add(int(value))
    if not window_values:
        return None
    return ",".join(str(value) for value in sorted(window_values))


def selected_params_payload(assessment: dict[str, Any] | None) -> list[dict[str, int | float]]:
    if assessment is None:
        return []
    payload: list[dict[str, int | float]] = []
    for item in assessment.get("selected_candidates", []):
        params_json = item.get("params_json")
        if not params_json:
            continue
        try:
            params = json.loads(params_json)
        except json.JSONDecodeError:
            continue
        if isinstance(params, dict):
            payload.append(params)
    return payload


def focus_search_space(
    family: str,
    search_space: dict[str, list[int | float]] | None,
    selected_params: list[dict[str, int | float]],
    interval: str,
) -> dict[str, list[int | float]] | None:
    if search_space is None or not selected_params:
        return search_space

    focused: dict[str, list[int | float]] = {}
    for key, values in search_space.items():
        step = infer_step(values)
        centered: set[int | float] = set()
        for params in selected_params:
            if key not in params:
                continue
            value = float(params[key])
            for candidate in (value - step, value, value + step):
                if candidate > 0:
                    if all(float(existing).is_integer() for existing in values):
                        centered.add(int(round(candidate)))
                    else:
                        centered.add(round(candidate, 4))
        focused[key] = sorted(centered) if centered else list(values)
    return sanitize_search_space(family, focused, interval=interval, window_spec=None)


def widen_search_space(
    family: str,
    search_space: dict[str, list[int | float]] | None,
    interval: str,
) -> dict[str, list[int | float]] | None:
    if search_space is None:
        return search_space

    widened: dict[str, list[int | float]] = {}
    for key, values in search_space.items():
        step = infer_step(values)
        minimum = float(min(values))
        maximum = float(max(values))
        expanded = list(values)
        lower = minimum - step
        upper = maximum + step
        if lower > 0:
            expanded.append(int(round(lower)) if all(float(existing).is_integer() for existing in values) else round(lower, 4))
        expanded.append(int(round(upper)) if all(float(existing).is_integer() for existing in values) else round(upper, 4))
        widened[key] = sorted(set(expanded))
    return sanitize_search_space(family, widened, interval=interval, window_spec=None)


def reallocate_days(proposal: LoopProposal, train_delta: int = 0, validation_delta: int = 0, test_delta: int = 0) -> LoopProposal:
    train_days = max(5, proposal.train_days + train_delta)
    validation_days = max(2, proposal.validation_days + validation_delta)
    test_days = max(2, proposal.test_days + test_delta)
    return replace(
        proposal,
        train_days=train_days,
        validation_days=validation_days,
        test_days=test_days,
    )


def mutate_proposal(proposal: LoopProposal, rationale: str, note: str, **changes: Any) -> LoopProposal:
    notes = list(proposal.mutation_notes)
    notes.append(note)
    return replace(proposal, rationale=rationale, mutation_notes=notes, **changes)


def toggle_selection_metric(selection_metric: str) -> str:
    return "total_return" if selection_metric == "sharpe_ratio" else "sharpe_ratio"


def next_family(family: str) -> str:
    normalized = normalize_family(family)
    families = list(SUPPORTED_STRATEGY_FAMILIES)
    idx = families.index(normalized)
    return families[(idx + 1) % len(families)]


def family_switch_mutation(proposal: LoopProposal, family: str, note: str, rationale: str) -> LoopProposal:
    search_space = default_search_space_for_family(family, proposal.interval)
    return mutate_proposal(
        proposal,
        rationale=rationale,
        note=note,
        family=family,
        strategy_name=family,
        search_space=search_space,
        window_spec=infer_window_spec_from_search_space(family, search_space),
    )


def build_candidate_mutations(
    proposal: LoopProposal,
    assessment: dict[str, Any] | None,
    error_message: str | None,
) -> list[LoopProposal]:
    candidates: list[LoopProposal] = []
    failed_checks = set() if assessment is None else set(assessment.get("failed_checks", []))
    selected_params = selected_params_payload(assessment)

    if error_message is not None:
        fallback = reallocate_days(proposal, train_delta=-2, validation_delta=-1, test_delta=-1)
        candidates.append(
            mutate_proposal(
                fallback,
                rationale="직전 실행이 오류였기 때문에 구간을 약간 줄여 유효한 탐색 공간을 다시 확보합니다.",
                note=f"실행 오류를 복구하기 위해 split 길이를 줄였습니다: {error_message}",
                candidate_count=max(3, proposal.candidate_count - 1),
            )
        )
        candidates.append(
            family_switch_mutation(
                proposal,
                family=next_family(proposal.family),
                note="실행 오류 후 다른 전략 패밀리로 전환했습니다.",
                rationale="직전 패밀리에서 오류가 났기 때문에 다른 구조의 전략 패밀리로 우회합니다.",
            )
        )
        return candidates

    if selected_params:
        focused_space = focus_search_space(proposal.family, proposal.search_space, selected_params, proposal.interval)
        if focused_space is not None and focused_space != proposal.search_space:
            candidates.append(
                mutate_proposal(
                    proposal,
                    rationale="선택된 후보 주변에 성과가 몰려 있어서 그 근방을 더 촘촘히 재탐색합니다.",
                    note="선택된 후보 파라미터 주변으로 탐색 공간을 좁혔습니다.",
                    search_space=focused_space,
                    window_spec=infer_window_spec_from_search_space(proposal.family, focused_space),
                )
            )

    if "test_outperformed_hold_ratio" in failed_checks:
        candidates.append(
            mutate_proposal(
                proposal,
                rationale="홀드를 자주 이기지 못했기 때문에 다른 순위 기준과 더 작은 후보 풀을 시험합니다.",
                note="홀드 초과 비율 개선을 위해 선택 기준을 바꾸고 후보 수를 줄였습니다.",
                selection_metric=toggle_selection_metric(proposal.selection_metric),
                candidate_count=max(3, proposal.candidate_count - 2),
            )
        )
        candidates.append(
            family_switch_mutation(
                proposal,
                family=next_family(proposal.family),
                note="홀드 대비 열세여서 다른 전략 패밀리로 전환했습니다.",
                rationale="현재 패밀리가 시장 구간과 맞지 않을 수 있어 다른 구조의 롱/숏 전략을 시험합니다.",
            )
        )

    if any(check in failed_checks for check in ("positive_test_ratio", "median_test_score", "mean_test_return")):
        widened_space = widen_search_space(proposal.family, proposal.search_space, proposal.interval)
        if widened_space is not None and widened_space != proposal.search_space:
            candidates.append(
                mutate_proposal(
                    proposal,
                    rationale="양수 테스트 비율과 중앙값이 약해서 탐색 범위를 넓혀 다른 파라미터 대역을 찾습니다.",
                    note="탐색 공간 상하단을 넓혔습니다.",
                    search_space=widened_space,
                    window_spec=infer_window_spec_from_search_space(proposal.family, widened_space),
                )
            )
        tighter_validation = reallocate_days(proposal, train_delta=-3, validation_delta=3)
        candidates.append(
            mutate_proposal(
                tighter_validation,
                rationale="아웃오브샘플 성과가 약해서 validation 구간을 더 엄격하게 만듭니다.",
                note="train 일부를 validation으로 옮겼습니다.",
            )
        )

    if "mean_test_drawdown" in failed_checks:
        candidates.append(
            mutate_proposal(
                proposal,
                rationale="낙폭이 커서 거래 밀도를 줄이고 더 보수적으로 선택합니다.",
                note="최소 거래 수를 올리고 후보 수를 줄였습니다.",
                min_trades=proposal.min_trades + 1,
                candidate_count=max(3, proposal.candidate_count - 1),
            )
        )

    candidates.append(
        mutate_proposal(
            proposal,
            rationale="직전 실험이 통과하지 못했기 때문에 train 후보 수를 늘려 더 넓게 탐색합니다.",
            note="validation으로 넘어가는 후보 수를 늘렸습니다.",
            candidate_count=min(proposal.candidate_count + 2, 20),
        )
    )
    candidates.append(
        mutate_proposal(
            proposal,
            rationale="검증 기준 자체가 현재 시장 구간과 맞지 않을 수 있어 다른 기준으로 재정렬합니다.",
            note="validation selection metric을 바꿨습니다.",
            selection_metric=toggle_selection_metric(proposal.selection_metric),
        )
    )
    return candidates


def score_iteration(record: dict[str, Any]) -> float:
    assessment = record.get("assessment")
    if assessment is None:
        return float("-inf")

    aggregate = assessment["aggregate_snapshot"]
    score = 0.0
    if assessment["verdict"] == "pass":
        score += 1_000.0
    score += 100.0 * float(aggregate.get("test_outperformed_hold_ratio", 0.0) or 0.0)
    score += 50.0 * float(aggregate.get("positive_test_ratio", 0.0) or 0.0)
    score += 10.0 * float(aggregate.get("median_test_score", 0.0) or 0.0)
    score += 25.0 * float(aggregate.get("mean_test_return", 0.0) or 0.0)
    score -= 10.0 * float(aggregate.get("mean_test_drawdown", 0.0) or 0.0)
    return score


def pick_next_proposal(
    current_record: dict[str, Any],
    history: list[dict[str, Any]],
    assessment: dict[str, Any] | None,
    error_message: str | None,
) -> LoopProposal | None:
    seen_signatures = {tuple(item["proposal_signature"]) for item in history}
    current_proposal = proposal_from_record(current_record)
    best_record = max(history, key=score_iteration) if history else current_record

    if current_proposal.llm_enabled:
        settings = LLMStrategySettings(
            enabled=True,
            model=current_proposal.llm_model,
            api_key_env=current_proposal.llm_api_key_env,
            base_url=current_proposal.llm_base_url,
        )
        try:
            llm_proposal = request_llm_strategy_proposal(
                current_proposal.to_pipeline_kwargs(),
                history,
                interval=current_proposal.interval,
                settings=settings,
            )
        except Exception as exc:
            llm_proposal = None
            llm_error_note = f"LLM 제안 실패로 휴리스틱 폴백: {type(exc).__name__}: {exc}"
        else:
            llm_error_note = None

        if llm_proposal is not None:
            candidate = replace(
                current_proposal,
                family=llm_proposal["family"],
                strategy_name=llm_proposal["strategy_name"],
                search_space=llm_proposal["search_space"],
                window_spec=llm_proposal["window_spec"],
                candidate_count=llm_proposal["candidate_count"],
                min_trades=llm_proposal["min_trades"],
                selection_metric=llm_proposal["selection_metric"],
                direction=llm_proposal["direction"],
                rationale=f"LLM 제안. {llm_proposal['rationale']}",
                mutation_notes=list(llm_proposal["mutation_notes"]),
            )
            if candidate.signature() not in seen_signatures:
                return candidate
        elif llm_error_note is not None:
            current_proposal = replace(
                current_proposal,
                mutation_notes=list(current_proposal.mutation_notes) + [llm_error_note],
            )

    ordered_sources: list[tuple[dict[str, Any], dict[str, Any] | None, str | None]] = []
    current_score = score_iteration(current_record)
    best_score = score_iteration(best_record)
    if best_record is not current_record and best_score > current_score:
        ordered_sources.append((best_record, best_record.get("assessment"), best_record.get("error_message")))
        ordered_sources.append((current_record, assessment, error_message))
    else:
        ordered_sources.append((current_record, assessment, error_message))
        if best_record is not current_record:
            ordered_sources.append((best_record, best_record.get("assessment"), best_record.get("error_message")))

    for source_record, source_assessment, source_error in ordered_sources:
        source_proposal = proposal_from_record(source_record)
        for candidate in build_candidate_mutations(source_proposal, assessment=source_assessment, error_message=source_error):
            if source_record is best_record and source_record is not current_record:
                candidate = replace(
                    candidate,
                    rationale=f"Best-of-history anchor. {candidate.rationale}",
                    mutation_notes=list(candidate.mutation_notes) + ["현재까지 최고 점수 iteration을 기준으로 다음 후보를 만들었습니다."],
                )
            if candidate.signature() not in seen_signatures:
                return candidate
    return None


def build_loop_summary(
    loop_dir: Path,
    history: list[dict[str, Any]],
    stop_reason: str,
) -> dict[str, Any]:
    best_record = None
    if history:
        best_record = max(history, key=score_iteration)

    return {
        "loop_dir": str(loop_dir),
        "iterations_completed": len(history),
        "stop_reason": stop_reason,
        "best_iteration": None if best_record is None else best_record["iteration"],
        "best_run_dir": None if best_record is None else best_record.get("run_dir"),
        "best_verdict": None if best_record is None else best_record.get("verdict"),
        "history": history,
    }


def save_loop_progress(loop_dir: Path, payload: dict[str, Any]) -> None:
    save_json(payload, loop_dir / "loop_progress.json")


def run_agentic_loop(
    proposal: LoopProposal,
    max_iterations: int = DEFAULT_MAX_ITERATIONS,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
) -> dict[str, Any]:
    if proposal.search_space is None:
        default_space = default_search_space_for_family(proposal.family, proposal.interval, window_spec=proposal.window_spec)
        proposal = replace(
            proposal,
            search_space=default_space,
            window_spec=infer_window_spec_from_search_space(proposal.family, default_space) or proposal.window_spec,
            rationale=f"{proposal.rationale} 선택된 family의 기본 탐색 공간을 채웠습니다.",
        )

    loop_dir = timestamped_run_dir(output_root, proposal.symbol)
    runs_root = loop_dir / "runs"
    history: list[dict[str, Any]] = []
    stop_reason = "max_iterations_reached"
    current_proposal = proposal

    for iteration in range(max_iterations):
        previous_record = None if not history else history[-1]
        run_dir: str | None = None
        manifest: dict[str, Any] | None = None
        assessment: dict[str, Any] | None = None
        error_message: str | None = None

        try:
            manifest = run_pipeline_once(**current_proposal.to_pipeline_kwargs(), output_root=runs_root)
            run_dir = str(manifest["run_dir"])
            assessment = load_json(manifest["files"]["assessment_json"])
            verdict = assessment["verdict"]
        except Exception as exc:  # pragma: no cover
            verdict = "error"
            error_message = f"{type(exc).__name__}: {exc}"

        record = {
            "iteration": iteration,
            "proposal": asdict(current_proposal),
            "proposal_signature": list(current_proposal.signature()),
            "rationale": current_proposal.rationale,
            "run_dir": run_dir,
            "verdict": verdict,
            "manifest": manifest,
            "assessment": assessment,
            "error_message": error_message,
        }
        history.append(record)
        record["visuals"] = generate_iteration_visuals(loop_dir, record)
        current_best_record = max(history, key=score_iteration)
        record["summary_ko_path"] = save_iteration_summary_ko(
            loop_dir,
            record,
            previous_record=previous_record,
            best_record=current_best_record,
        )
        if record.get("summary_ko_path") is not None:
            record["summary_ko_loop_rel"] = relative_path_str(record["summary_ko_path"], loop_dir)
        progress_payload = build_loop_summary(loop_dir, history, stop_reason=stop_reason)
        progress_payload["summary_ko_path"] = str(loop_dir / "loop_summary_ko.md")
        save_loop_progress(loop_dir, progress_payload)
        save_json({"markdown": render_loop_summary_ko(progress_payload)}, loop_dir / "loop_summary_ko.json")
        (loop_dir / "loop_summary_ko.md").write_text(render_loop_summary_ko(progress_payload))

        if verdict == "pass":
            stop_reason = "passed"
            break

        next_proposal = pick_next_proposal(
            current_record=record,
            history=history,
            assessment=assessment,
            error_message=error_message,
        )
        if next_proposal is None:
            stop_reason = "no_new_candidate"
            break
        current_proposal = next_proposal

    summary = build_loop_summary(loop_dir, history, stop_reason=stop_reason)
    summary["summary_ko_path"] = str(loop_dir / "loop_summary_ko.md")
    save_json(summary, loop_dir / "loop_manifest.json")
    (loop_dir / "loop_summary_ko.md").write_text(render_loop_summary_ko(summary))
    summary["latest"] = update_latest_pointer(
        output_root,
        loop_dir,
        summary_name="loop_summary_ko.md",
        manifest_name="loop_manifest.json",
    )
    save_json(summary, loop_dir / "loop_manifest.json")
    return summary


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run an agent-like iterative search loop over the walk-forward pipeline.")
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
    parser.add_argument("--max-iterations", type=int, default=DEFAULT_MAX_ITERATIONS)
    parser.add_argument("--llm-enable", action="store_true")
    parser.add_argument("--llm-model", default=DEFAULT_OPENAI_MODEL)
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-base-url", default=DEFAULT_OPENAI_BASE_URL)
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    symbol = args.symbol or default_symbol_for_source(args.data_source)
    family = normalize_family(args.family)
    search_space = None if args.search_space_json is None else json.loads(args.search_space_json)
    if search_space is None:
        search_space = default_search_space_for_family(family, args.interval, window_spec=args.window_spec or default_window_spec_for_interval(args.interval))
    else:
        search_space = sanitize_search_space(family, search_space, interval=args.interval, window_spec=None)

    proposal = LoopProposal(
        symbol=symbol,
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
        family=family,
        strategy_name=args.strategy_name or family,
        window_spec=infer_window_spec_from_search_space(family, search_space) or args.window_spec,
        search_space=search_space,
        candidate_count=args.candidate_count,
        min_trades=args.min_trades,
        selection_metric=args.selection_metric,
        direction=args.direction,
        fees=args.fees,
        init_cash=args.init_cash,
        llm_enabled=args.llm_enable,
        llm_model=args.llm_model,
        llm_api_key_env=args.llm_api_key_env,
        llm_base_url=args.llm_base_url,
        rationale="Baseline proposal from CLI inputs.",
    )
    summary = run_agentic_loop(
        proposal=proposal,
        max_iterations=args.max_iterations,
        output_root=args.output_root,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
