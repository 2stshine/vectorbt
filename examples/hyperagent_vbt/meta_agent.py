from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from typing import Any

import requests

from examples.hyperagent_vbt.archive import top_records_for_context
from examples.hyperagent_vbt.models import GenerationRecord, MetaPolicy, RunConfig


@dataclass(frozen=True)
class MetaAgentSettings:
    enabled: bool = False
    model: str = "gpt-4o-mini"
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = "https://api.openai.com/v1"
    timeout_seconds: int = 90


def _llm_key_available(settings: MetaAgentSettings) -> bool:
    return bool(os.environ.get(settings.api_key_env))


def _build_schema() -> dict[str, Any]:
    return {
        "name": "hyperagent_meta_policy_update",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "reflection_ko": {"type": "string"},
                "parent_selection": {
                    "type": "string",
                    "enum": ["random", "latest", "best", "score_prop", "score_child_prop"],
                },
                "exploration_prob": {"type": "number", "minimum": 0.05, "maximum": 0.9},
                "family_switch_after_failures": {"type": "integer", "minimum": 1, "maximum": 5},
                "stage_min_median_test_score": {"type": "number", "minimum": -0.3, "maximum": 0.3},
                "notes_ko": {
                    "type": "array",
                    "items": {"type": "string"},
                    "minItems": 1,
                    "maxItems": 4,
                },
            },
            "required": [
                "reflection_ko",
                "parent_selection",
                "exploration_prob",
                "family_switch_after_failures",
                "stage_min_median_test_score",
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
        raise RuntimeError(f"LLM refused to produce a meta reflection: {refusal}")
    content = message.get("content")
    if isinstance(content, str):
        return content
    raise RuntimeError("LLM response content was not a plain string.")


def _request_llm_reflection(
    latest_record: GenerationRecord,
    archive_records: list[GenerationRecord],
    memory_summary: dict[str, object],
    meta_policy: MetaPolicy,
    run_config: RunConfig,
    settings: MetaAgentSettings,
) -> tuple[MetaPolicy, str]:
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
                "failed_checks": record.failed_checks,
                "full_verdict": record.full_verdict,
            }
        )

    payload = {
        "run_config": run_config.to_dict(),
        "current_meta_policy": meta_policy.to_dict(),
        "memory_summary": memory_summary,
        "latest_record": latest_record.to_dict(),
        "archive_context": archive_context,
        "instructions": {
            "goal": "다음 세대에서 전략 생성 방식 자체를 더 잘 작동하게 만드는 메타 정책 업데이트를 제안하세요.",
            "priority": [
                "성능 정체 시 탐색 강화",
                "좋은 계보가 보이면 적당한 활용 유지",
                "오버피팅 징후 반영",
                "한국어 한두 문장으로 메타 반성 작성",
            ],
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
                        "You are the meta agent in a HyperAgent-style quantitative research loop. "
                        "Update the meta-policy that controls future strategy generation. "
                        "Return only valid JSON matching the schema."
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
    updated = replace(
        meta_policy,
        parent_selection=str(raw["parent_selection"]),
        exploration_prob=float(raw["exploration_prob"]),
        family_switch_after_failures=int(raw["family_switch_after_failures"]),
        stage_min_median_test_score=float(raw["stage_min_median_test_score"]),
    )
    reflection = str(raw["reflection_ko"]).strip()
    notes = [str(item) for item in raw["notes_ko"]]
    if notes:
        reflection = reflection + " " + " ".join(notes)
    return updated, reflection.strip()


def _heuristic_reflection(
    latest_record: GenerationRecord,
    archive_records: list[GenerationRecord],
    meta_policy: MetaPolicy,
) -> tuple[MetaPolicy, str]:
    recent = archive_records[-3:]
    all_fail = recent and all(record.full_verdict != "pass" for record in recent)
    hold_ratio = None if latest_record.aggregate is None else latest_record.aggregate.get("test_outperformed_hold_ratio")
    val_gap = None if latest_record.aggregate is None else latest_record.aggregate.get("mean_validation_test_gap")
    updated = meta_policy
    notes: list[str] = []

    if latest_record.full_verdict == "pass":
        updated = replace(
            updated,
            exploration_prob=max(0.1, updated.exploration_prob - 0.05),
        )
        notes.append("최근 세대가 통과했기 때문에 탐색 비중을 약간 낮추고 활용 비중을 높였습니다.")
    elif all_fail:
        updated = replace(
            updated,
            exploration_prob=min(0.75, updated.exploration_prob + 0.08),
            family_switch_after_failures=max(1, updated.family_switch_after_failures - 1),
            parent_selection="score_child_prop",
        )
        notes.append("최근 여러 세대가 연속 실패해서 탐색 비중을 높이고 패밀리 전환을 더 빠르게 하도록 조정했습니다.")

    if hold_ratio is not None and float(hold_ratio) <= 0.0:
        updated = replace(updated, parent_selection="score_child_prop")
        notes.append("홀드 초과 성능이 없어서 성과 좋은 계보만 파는 대신 넓게 분기하는 선택 전략을 유지합니다.")

    if val_gap is not None and float(val_gap) > 1.0:
        updated = replace(updated, stage_min_median_test_score=max(-0.02, updated.stage_min_median_test_score))
        notes.append("validation 대비 test 하락폭이 커서 stage 기준을 조금 더 보수적으로 만들었습니다.")

    if not notes:
        notes.append("메타 정책은 현재 기본 균형을 유지합니다.")
    return updated, " ".join(notes)


class MetaAgent:
    def __init__(self, settings: MetaAgentSettings) -> None:
        self.settings = settings

    def reflect(
        self,
        latest_record: GenerationRecord,
        archive_records: list[GenerationRecord],
        memory_summary: dict[str, object],
        meta_policy: MetaPolicy,
        run_config: RunConfig,
    ) -> tuple[MetaPolicy, str, bool]:
        if self.settings.enabled and _llm_key_available(self.settings):
            try:
                updated_policy, reflection = _request_llm_reflection(
                    latest_record,
                    archive_records,
                    memory_summary,
                    meta_policy,
                    run_config,
                    self.settings,
                )
            except Exception:
                pass
            else:
                return updated_policy, reflection, True

        updated_policy, reflection = _heuristic_reflection(latest_record, archive_records, meta_policy)
        return updated_policy, reflection, False


def build_meta_agent_settings(run_config: RunConfig) -> MetaAgentSettings:
    return MetaAgentSettings(
        enabled=run_config.llm_enabled,
        model=run_config.llm_model,
        api_key_env=run_config.llm_api_key_env,
        base_url=run_config.llm_base_url,
    )
