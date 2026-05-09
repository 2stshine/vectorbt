from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from examples.agent_pipeline.common import load_json
from examples.hyperagent_vbt.models import GenerationRecord, MemoryState, MetaPolicy, StrategyProgram


STOP_REASON_LABELS_KO = {
    "passed": "통과 세대를 찾았기 때문에 종료했습니다.",
    "no_new_candidate": "새로운 프로그램 제안을 만들지 못해서 종료했습니다.",
    "max_generations_reached": "최대 세대 수에 도달해서 종료했습니다.",
}


def _relative(path: str | Path, base_dir: str | Path) -> str:
    return os.path.relpath(Path(path), Path(base_dir))


def _load_manifest(path: str | Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    path = Path(path)
    if not path.exists():
        return None
    return load_json(path)


def _selected_candidate_labels(record: GenerationRecord) -> list[str]:
    return [item.get("candidate_label", "unknown") for item in record.selected_candidates]


def render_generation_summary_ko(
    loop_dir: str | Path,
    record: GenerationRecord,
    meta_policy: MetaPolicy,
) -> str:
    loop_dir = Path(loop_dir)
    program = StrategyProgram.from_dict(record.program)
    stage_manifest = _load_manifest(record.stage_manifest_path)
    full_manifest = _load_manifest(record.full_manifest_path)
    preferred_manifest = full_manifest or stage_manifest

    lines = [
        f"# Generation {record.gen_id} 요약",
        "",
        f"- 부모 generation: {record.parent_gen_id}",
        f"- 부모 선택 방식: `{record.parent_selection_method}`",
        f"- 전략 패밀리: `{program.family}`",
        f"- 전략 이름: `{program.strategy_name}`",
        f"- 가설: {program.hypothesis_ko}",
        f"- 제안 이유: {program.rationale_ko}",
        f"- 탐색 공간: `{json.dumps(program.search_space, ensure_ascii=False, sort_keys=True)}`",
        f"- 선택 기준: `{program.selection_metric}`, 후보 {program.candidate_count}개, 최소 거래 {program.min_trades}회, 방향 `{program.direction}`",
        f"- Stage 통과 여부: {'통과' if record.stage_passed else '실패'}",
        f"- Full verdict: {record.full_verdict or '미실행'}",
        f"- 점수: {record.score_value}",
        f"- 실패 체크: {record.failed_checks if record.failed_checks else '없음'}",
        f"- 선택 후보: {_selected_candidate_labels(record) if record.selected_candidates else '없음'}",
        f"- 메타 반성: {record.meta_reflection_ko}",
        f"- 메타 정책 스냅샷: parent_selection `{meta_policy.parent_selection}`, exploration {meta_policy.exploration_prob:.2f}, family_switch_after_failures {meta_policy.family_switch_after_failures}",
    ]

    for note in program.notes_ko:
        lines.append(f"- 전략 메모: {note}")

    if preferred_manifest is not None:
        run_dir = Path(preferred_manifest["run_dir"])
        summary_path = run_dir / "summary_ko.md"
        hold_plot = run_dir / "plots" / "hold_comparison.png"
        return_plot = run_dir / "plots" / "return_timeline.png"
        overfit_plot = run_dir / "plots" / "overfitting_timeline.png"
        if summary_path.exists():
            lines.append(f"- 백테스트 요약: [{summary_path.name}]({_relative(summary_path, loop_dir)})")
        if hold_plot.exists():
            lines.extend(
                [
                    "",
                    "## 시각화",
                    "",
                    "### 홀드 비교",
                    f"![홀드 비교]({_relative(hold_plot, loop_dir)})",
                ]
            )
        if return_plot.exists():
            lines.extend(
                [
                    "",
                    "### 수익률 타임라인",
                    f"![수익률 타임라인]({_relative(return_plot, loop_dir)})",
                ]
            )
        if overfit_plot.exists():
            lines.extend(
                [
                    "",
                    "### 오버피팅 타임라인",
                    f"![오버피팅 타임라인]({_relative(overfit_plot, loop_dir)})",
                ]
            )

    return "\n".join(lines) + "\n"


def render_loop_summary_ko(
    loop_dir: str | Path,
    records: list[GenerationRecord],
    memory: MemoryState,
    meta_policy: MetaPolicy,
    stop_reason: str,
) -> str:
    loop_dir = Path(loop_dir)
    best_record = None
    if records:
        scored = [record for record in records if record.score_value is not None]
        if scored:
            best_record = max(scored, key=lambda record: float(record.score_value or float("-inf")))

    lines = [
        "# HyperAgent VBT 루프 요약",
        "",
        f"- 총 세대 수: {len(records)}",
        f"- 종료 이유: {STOP_REASON_LABELS_KO.get(stop_reason, stop_reason)}",
        f"- 현재 메타 정책: parent_selection `{meta_policy.parent_selection}`, exploration {meta_policy.exploration_prob:.2f}, family_switch_after_failures {meta_policy.family_switch_after_failures}",
        f"- 기억된 강한 패밀리: {memory.promising_families if memory.promising_families else '없음'}",
        f"- 기억된 약한 패밀리: {memory.weak_families if memory.weak_families else '없음'}",
    ]

    if best_record is not None:
        lines.append(f"- 최고 세대: {best_record.gen_id}, family `{best_record.program['family']}`, score {best_record.score_value}")

    for record in records:
        lines.append(
            f"- Gen {record.gen_id}: family `{record.program['family']}`, stage {'통과' if record.stage_passed else '실패'}, full `{record.full_verdict or '미실행'}`, score {record.score_value}"
        )
        generation_summary = loop_dir / f"gen_{record.gen_id:04d}" / "generation_summary_ko.md"
        if generation_summary.exists():
            lines.append(f"  자세히 보기: [{generation_summary.name}]({_relative(generation_summary, loop_dir)})")

    if memory.observations_ko:
        lines.extend(["", "## 메모리 관찰", ""])
        for item in memory.observations_ko[-8:]:
            lines.append(f"- {item}")

    return "\n".join(lines) + "\n"
