from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any

import requests

from examples.agent_pipeline.strategy_registry import (
    SUPPORTED_STRATEGY_FAMILIES,
    default_search_space_for_family,
    normalize_family,
    sanitize_search_space,
    strategy_family_catalog,
)
from examples.hyperagent_vbt.archive import top_records_for_context
from examples.hyperagent_vbt.memory import summarize_memory
from examples.hyperagent_vbt.models import GenerationRecord, MetaPolicy, RunConfig, StrategyProgram


TREND_FAMILIES = {
    "ma_crossover_ls",
    "channel_breakout_ls",
    "ma_rsi_filter_ls",
    "macd_signal_ls",
}
MEAN_REVERSION_FAMILIES = {
    "rsi_reversion_ls",
    "bbands_reversion_ls",
    "stoch_reversion_ls",
}


@dataclass(frozen=True)
class TaskAgentSettings:
    enabled: bool = False
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 90


def infer_step(values: list[int | float]) -> float:
    sorted_values = sorted(set(float(value) for value in values))
    if len(sorted_values) < 2:
        return 1.0
    diffs = [sorted_values[idx] - sorted_values[idx - 1] for idx in range(1, len(sorted_values))]
    diffs = [diff for diff in diffs if diff > 0]
    return min(diffs) if diffs else 1.0


def infer_window_spec_from_search_space(family: str, search_space: dict[str, list[int | float]] | None) -> str | None:
    if search_space is None or family not in {"ma_crossover_ls", "ma_rsi_filter_ls", "macd_signal_ls"}:
        return None
    window_values: set[int] = set()
    for key in ("fast_window", "slow_window"):
        for value in search_space.get(key, []):
            window_values.add(int(value))
    if not window_values:
        return None
    return ",".join(str(value) for value in sorted(window_values))


def _selected_params(record: GenerationRecord | None) -> list[dict[str, int | float]]:
    if record is None:
        return []
    params: list[dict[str, int | float]] = []
    for item in record.selected_candidates:
        params_json = item.get("params_json")
        if not params_json:
            continue
        try:
            payload = json.loads(params_json)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            params.append(payload)
    return params


def focus_search_space(
    family: str,
    search_space: dict[str, list[int | float]],
    selected_params: list[dict[str, int | float]],
    interval: str,
) -> dict[str, list[int | float]]:
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
    search_space: dict[str, list[int | float]],
    interval: str,
) -> dict[str, list[int | float]]:
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


def _family_score(family: str, memory_summary: dict[str, object], generation_idx: int) -> float:
    promising = float(dict(memory_summary["promising_families"]).get(family, 0))
    weak = float(dict(memory_summary["weak_families"]).get(family, 0))
    return promising - 0.75 * weak - 0.02 * generation_idx


def _family_rotation_target(
    current_family: str,
    memory_summary: dict[str, object],
    generation_idx: int,
) -> str:
    families = list(SUPPORTED_STRATEGY_FAMILIES)
    alternate_pool = [family for family in families if family != current_family]
    alternate_pool.sort(key=lambda family: _family_score(family, memory_summary, generation_idx), reverse=True)
    if current_family in TREND_FAMILIES:
        mean_rev = [family for family in alternate_pool if family in MEAN_REVERSION_FAMILIES]
        if mean_rev:
            return mean_rev[0]
    if current_family in MEAN_REVERSION_FAMILIES:
        trend = [family for family in alternate_pool if family in TREND_FAMILIES]
        if trend:
            return trend[0]
    return alternate_pool[0] if alternate_pool else current_family


def _llm_key_available(settings: TaskAgentSettings) -> bool:
    return bool(os.environ.get(settings.api_key_env))


def _build_schema() -> dict[str, Any]:
    return {
        "name": "hyperagent_strategy_program",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "family": {"type": "string", "enum": list(SUPPORTED_STRATEGY_FAMILIES)},
                "strategy_name": {"type": "string"},
                "search_space": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "number"},
                    },
                },
                "candidate_count": {"type": "integer", "minimum": 4, "maximum": 24},
                "min_trades": {"type": "integer", "minimum": 1, "maximum": 20},
                "selection_metric": {"type": "string", "enum": ["sharpe_ratio", "total_return"]},
                "direction": {"type": "string", "enum": ["both", "longonly", "shortonly"]},
                "hypothesis_ko": {"type": "string"},
                "rationale_ko": {"type": "string"},
                "notes_ko": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 4,
                },
            },
            "required": [
                "family",
                "strategy_name",
                "search_space",
                "candidate_count",
                "min_trades",
                "selection_metric",
                "direction",
                "hypothesis_ko",
                "rationale_ko",
                "notes_ko",
            ],
            "additionalProperties": False,
        },
    }


def _extract_message_text(payload: dict[str, Any]) -> str:
    choices = payload.get("choices", [])
    if not choices:
        raise RuntimeError("LLM response contained no choices.")
    message = choices[0].get("message", {})
    refusal = message.get("refusal")
    if refusal:
        raise RuntimeError(f"LLM refused to produce a strategy proposal: {refusal}")
    content = message.get("content")
    if isinstance(content, str):
        return content
    raise RuntimeError("LLM response content was not a plain string.")


def _request_llm_program(
    parent_program: StrategyProgram | None,
    parent_record: GenerationRecord | None,
    archive_records: list[GenerationRecord],
    memory_summary: dict[str, object],
    meta_policy: MetaPolicy,
    run_config: RunConfig,
    settings: TaskAgentSettings,
) -> StrategyProgram:
    api_key = os.environ.get(settings.api_key_env)
    if not api_key:
        raise RuntimeError("LLM API key is not configured.")

    archive_context = []
    for record in top_records_for_context(archive_records, meta_policy.top_k_archive_context):
        archive_context.append(
            {
                "gen_id": record.gen_id,
                "family": record.program["family"],
                "score_value": record.score_value,
                "full_verdict": record.full_verdict,
                "failed_checks": record.failed_checks,
                "selected_candidates": record.selected_candidates[:3],
            }
        )

    payload = {
        "run_config": run_config.to_dict(),
        "meta_policy": meta_policy.to_dict(),
        "memory_summary": memory_summary,
        "family_catalog": strategy_family_catalog(run_config.interval),
        "parent_program": None if parent_program is None else parent_program.to_dict(),
        "parent_record": None
        if parent_record is None
        else {
            "gen_id": parent_record.gen_id,
            "failed_checks": parent_record.failed_checks,
            "aggregate": parent_record.aggregate,
            "selected_candidates": parent_record.selected_candidates[:3],
        },
        "archive_context": archive_context,
        "instructions": {
            "goal": "다음 세대의 자동 매매 전략 프로그램을 제안하세요.",
            "priority": [
                "홀드 초과 성능 개선",
                "validation/test 일반화",
                "과도한 오버피팅 회피",
                "현실적인 거래 빈도",
            ],
            "style": "한국어로 가설과 제안 이유를 짧고 명확하게 설명하세요.",
        },
    }

    response = requests.post(
        f"{settings.base_url.rstrip('/')}/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": settings.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are the task agent in a HyperAgent-style quantitative research loop. "
                        "Propose the next strategy program for vectorbt walk-forward evaluation. "
                        "You must return only a valid JSON object that matches the schema."
                    ),
                },
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_schema", "json_schema": _build_schema()},
        },
        timeout=settings.timeout_seconds,
    )
    response.raise_for_status()
    raw = json.loads(_extract_message_text(response.json()))
    family = normalize_family(raw["family"])
    search_space = sanitize_search_space(family, raw["search_space"], interval=run_config.interval, window_spec=None)
    return StrategyProgram(
        family=family,
        strategy_name=str(raw["strategy_name"]),
        window_spec=infer_window_spec_from_search_space(family, search_space),
        search_space=search_space,
        candidate_count=int(raw["candidate_count"]),
        min_trades=int(raw["min_trades"]),
        selection_metric=str(raw["selection_metric"]),
        direction=str(raw["direction"]),
        hypothesis_ko=str(raw["hypothesis_ko"]),
        rationale_ko=str(raw["rationale_ko"]),
        notes_ko=[str(item) for item in raw["notes_ko"]],
        source="task_agent_llm",
    )


def _heuristic_program(
    parent_program: StrategyProgram | None,
    parent_record: GenerationRecord | None,
    archive_records: list[GenerationRecord],
    memory_summary: dict[str, object],
    meta_policy: MetaPolicy,
    run_config: RunConfig,
    *,
    seed_family: str,
    rng: random.Random,
) -> StrategyProgram:
    if parent_program is None:
        family = normalize_family(seed_family)
        search_space = default_search_space_for_family(family, run_config.interval)
        selection_metric = "total_return" if family in TREND_FAMILIES else "sharpe_ratio"
        return StrategyProgram(
            family=family,
            strategy_name=family,
            window_spec=infer_window_spec_from_search_space(family, search_space),
            search_space=search_space,
            candidate_count=max(meta_policy.min_candidate_count, 8),
            min_trades=1,
            selection_metric=selection_metric,
            direction=run_config.direction,
            hypothesis_ko=f"초기 세대에서는 `{family}` 패밀리로 기준선을 잡고 이후 세대가 여기서 분기합니다.",
            rationale_ko="첫 세대의 seed 전략을 평가해 이후 세대가 비교할 출발점을 만듭니다.",
            notes_ko=[f"seed 패밀리: {family}"],
            source="task_agent_seed",
        )

    failed_checks = set() if parent_record is None else set(parent_record.failed_checks)
    selected_params = _selected_params(parent_record)
    repeat_failures = int(dict(memory_summary["weak_families"]).get(parent_program.family, 0))
    family = parent_program.family
    search_space = dict(parent_program.search_space)
    candidate_count = parent_program.candidate_count
    min_trades = parent_program.min_trades
    selection_metric = parent_program.selection_metric
    rationale = "직전 결과를 바탕으로 다음 전략 프로그램을 조정합니다."
    hypothesis = "직전 실패 원인을 줄이는 방향으로 탐색 공간을 재설계합니다."
    notes: list[str] = []

    should_rotate = False
    if repeat_failures >= meta_policy.family_switch_after_failures:
        should_rotate = True
    if "test_outperformed_hold_ratio" in failed_checks and rng.random() < max(meta_policy.exploration_prob, 0.35):
        should_rotate = True
    if rng.random() < meta_policy.exploration_prob * 0.3:
        should_rotate = True

    if should_rotate:
        family = _family_rotation_target(parent_program.family, memory_summary, len(archive_records))
        search_space = default_search_space_for_family(family, run_config.interval)
        candidate_count = max(meta_policy.min_candidate_count, min(meta_policy.max_candidate_count, parent_program.candidate_count))
        selection_metric = "total_return" if family in TREND_FAMILIES else "sharpe_ratio"
        rationale = f"`{parent_program.family}` 패밀리가 반복적으로 약해서 `{family}` 패밀리로 전환합니다."
        hypothesis = "기존 패밀리의 구조적 약점을 피하기 위해 다른 신호 구조를 시험합니다."
        notes.append("패밀리 로테이션을 적용했습니다.")
    else:
        if meta_policy.focus_on_selected and selected_params:
            search_space = focus_search_space(parent_program.family, search_space, selected_params, run_config.interval)
            rationale = "직전 선택 후보 근처에서 성과가 모여 있어 탐색 공간을 좁혀 재검증합니다."
            hypothesis = "선택된 후보 주변의 미세 조정이 일반화 성능을 개선할 수 있습니다."
            notes.append("선택 후보 주변으로 탐색 공간을 집중했습니다.")

        if meta_policy.widen_after_failure and any(check in failed_checks for check in ("median_test_score", "positive_test_ratio", "mean_test_return")):
            search_space = widen_search_space(parent_program.family, search_space, run_config.interval)
            rationale = "아웃오브샘플 성과가 약해 탐색 공간 상하단을 넓혔습니다."
            hypothesis = "현재 범위 밖에 더 적합한 파라미터 대역이 있을 수 있습니다."
            notes.append("탐색 공간을 확장했습니다.")

        if "mean_test_drawdown" in failed_checks:
            min_trades += 1
            notes.append("낙폭을 줄이기 위해 최소 거래 수 기준을 높였습니다.")
        if "test_outperformed_hold_ratio" in failed_checks:
            selection_metric = "total_return"
            candidate_count = max(meta_policy.min_candidate_count, parent_program.candidate_count - 1)
            notes.append("홀드 초과 성능을 우선하기 위해 total_return 기준으로 전환했습니다.")
        elif "positive_test_ratio" in failed_checks:
            selection_metric = "sharpe_ratio"
            candidate_count = min(meta_policy.max_candidate_count, parent_program.candidate_count + 2)
            notes.append("양수 비율 회복을 위해 후보 수를 늘렸습니다.")

    search_space = sanitize_search_space(family, search_space, interval=run_config.interval, window_spec=None)
    candidate_count = max(meta_policy.min_candidate_count, min(meta_policy.max_candidate_count, int(candidate_count)))
    return StrategyProgram(
        family=family,
        strategy_name=family,
        window_spec=infer_window_spec_from_search_space(family, search_space),
        search_space=search_space,
        candidate_count=candidate_count,
        min_trades=max(1, int(min_trades)),
        selection_metric=selection_metric,
        direction=run_config.direction,
        hypothesis_ko=hypothesis,
        rationale_ko=rationale,
        notes_ko=notes or ["휴리스틱 기본 조정입니다."],
        source="task_agent_heuristic",
    )


class TaskAgent:
    def __init__(self, settings: TaskAgentSettings) -> None:
        self.settings = settings

    def propose(
        self,
        parent_program: StrategyProgram | None,
        parent_record: GenerationRecord | None,
        archive_records: list[GenerationRecord],
        meta_policy: MetaPolicy,
        run_config: RunConfig,
        memory_state: dict[str, object],
        *,
        seed_family: str = "ma_crossover_ls",
        rng: random.Random,
    ) -> tuple[StrategyProgram, bool]:
        if parent_program is None and not archive_records:
            program = _heuristic_program(
                parent_program,
                parent_record,
                archive_records,
                memory_state,
                meta_policy,
                run_config,
                seed_family=seed_family,
                rng=rng,
            )
            return program, False

        if self.settings.enabled and _llm_key_available(self.settings):
            try:
                program = _request_llm_program(
                    parent_program,
                    parent_record,
                    archive_records,
                    memory_state,
                    meta_policy,
                    run_config,
                    self.settings,
                )
            except Exception:
                pass
            else:
                return program, True

        program = _heuristic_program(
            parent_program,
            parent_record,
            archive_records,
            memory_state,
            meta_policy,
            run_config,
            seed_family=seed_family,
            rng=rng,
        )
        return program, False


def build_task_agent_settings(run_config: RunConfig) -> TaskAgentSettings:
    return TaskAgentSettings(
        enabled=run_config.llm_enabled,
        model=run_config.llm_model,
        api_key_env=run_config.llm_api_key_env,
        base_url=run_config.llm_base_url,
    )


def summarize_for_task_agent(memory_state: dict[str, object]) -> dict[str, object]:
    return memory_state
