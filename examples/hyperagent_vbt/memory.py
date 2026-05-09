from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

from examples.hyperagent_vbt.models import GenerationRecord, MemoryState


def load_memory(path: str | Path) -> MemoryState:
    path = Path(path)
    if not path.exists():
        return MemoryState()
    return MemoryState.from_dict(json.loads(path.read_text()))


def save_memory(memory: MemoryState, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(memory.to_dict(), indent=2, ensure_ascii=False))


def _push_limited(items: list[str], new_item: str, *, limit: int = 12) -> list[str]:
    if not new_item:
        return items
    deduped = [item for item in items if item != new_item]
    deduped.append(new_item)
    return deduped[-limit:]


def summarize_memory(memory: MemoryState) -> dict[str, object]:
    return {
        "promising_families": memory.promising_families,
        "weak_families": memory.weak_families,
        "recent_success_patterns_ko": memory.success_patterns_ko[-6:],
        "recent_failure_patterns_ko": memory.failure_patterns_ko[-6:],
        "recent_observations_ko": memory.observations_ko[-6:],
        "recent_meta_reflections_ko": memory.meta_reflections_ko[-6:],
    }


def update_memory(
    memory: MemoryState,
    record: GenerationRecord,
) -> MemoryState:
    family = str(record.program["family"])
    promising_families = dict(memory.promising_families)
    weak_families = dict(memory.weak_families)
    family_last_seen = dict(memory.family_last_seen)
    success_patterns = list(memory.success_patterns_ko)
    failure_patterns = list(memory.failure_patterns_ko)
    observations = list(memory.observations_ko)
    meta_reflections = list(memory.meta_reflections_ko)

    family_last_seen[family] = record.gen_id
    aggregate = record.aggregate or {}
    hold_ratio = aggregate.get("test_outperformed_hold_ratio")
    positive_ratio = aggregate.get("positive_test_ratio")
    val_gap = aggregate.get("mean_validation_test_gap")

    if record.full_verdict == "pass":
        promising_families[family] = promising_families.get(family, 0) + 1
        success_patterns = _push_limited(
            success_patterns,
            f"{family}: 테스트 기준을 통과한 사례가 누적되었습니다.",
        )
    else:
        weak_families[family] = weak_families.get(family, 0) + 1

    if hold_ratio is not None and float(hold_ratio) <= 0.0:
        failure_patterns = _push_limited(
            failure_patterns,
            f"{family}: Long Hold를 이기지 못하는 경향이 있습니다.",
        )
    if positive_ratio is not None and float(positive_ratio) < 0.5:
        failure_patterns = _push_limited(
            failure_patterns,
            f"{family}: 아웃오브샘플에서 양수 구간 비율이 낮았습니다.",
        )
    if val_gap is not None and float(val_gap) > 1.0:
        failure_patterns = _push_limited(
            failure_patterns,
            f"{family}: validation 대비 test 성과 하락폭이 커서 오버피팅 의심이 있습니다.",
        )
    if hold_ratio is not None and positive_ratio is not None and float(positive_ratio) >= 0.75 and float(hold_ratio) == 0.0:
        observations = _push_limited(
            observations,
            f"{family}: 방향은 어느 정도 맞지만 큰 추세를 충분히 먹지 못하는 패턴이 보였습니다.",
        )
    if record.selected_candidates:
        observations = _push_limited(
            observations,
            f"{family}: 최근 선택 후보는 {', '.join(item['candidate_label'] for item in record.selected_candidates[:3])} 쪽으로 모였습니다.",
        )
    if record.meta_reflection_ko:
        meta_reflections = _push_limited(meta_reflections, record.meta_reflection_ko)

    return replace(
        memory,
        promising_families=promising_families,
        weak_families=weak_families,
        family_last_seen=family_last_seen,
        success_patterns_ko=success_patterns,
        failure_patterns_ko=failure_patterns,
        observations_ko=observations,
        meta_reflections_ko=meta_reflections,
    )
