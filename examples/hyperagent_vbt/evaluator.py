from __future__ import annotations

from pathlib import Path
from typing import Any

from examples.agent_pipeline.common import load_json
from examples.agent_pipeline.run_once import run_pipeline_once
from examples.hyperagent_vbt.models import EvaluationArtifacts, MetaPolicy, RunConfig, StrategyProgram


def score_from_assessment(assessment: dict[str, Any] | None, verdict: str | None) -> float | None:
    if assessment is None:
        return None
    aggregate = assessment["aggregate_snapshot"]
    score = 0.0
    if verdict == "pass":
        score += 1_000.0
    score += 100.0 * float(aggregate.get("test_outperformed_hold_ratio", 0.0) or 0.0)
    score += 50.0 * float(aggregate.get("positive_test_ratio", 0.0) or 0.0)
    score += 10.0 * float(aggregate.get("median_test_score", 0.0) or 0.0)
    score += 25.0 * float(aggregate.get("mean_test_return", 0.0) or 0.0)
    score -= 10.0 * float(aggregate.get("mean_test_drawdown", 0.0) or 0.0)
    return score


def _load_assessment_from_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    return load_json(manifest["files"]["assessment_json"])


def _run_manifest(
    program: StrategyProgram,
    run_config: RunConfig,
    output_root: Path,
    *,
    n_splits: int,
    candidate_count: int,
) -> dict[str, Any]:
    return run_pipeline_once(
        symbol=run_config.symbol,
        data_source=run_config.data_source,
        exchange=run_config.exchange,
        period=run_config.period,
        start=run_config.start,
        end=run_config.end,
        interval=run_config.interval,
        freq=run_config.freq,
        train_days=run_config.train_days,
        validation_days=run_config.validation_days,
        test_days=run_config.test_days,
        n_splits=n_splits,
        family=program.family,
        strategy_name=program.strategy_name,
        window_spec=program.window_spec,
        search_space=program.search_space,
        candidate_count=candidate_count,
        min_trades=program.min_trades,
        selection_metric=program.selection_metric,
        direction=program.direction,
        fees=run_config.fees,
        init_cash=run_config.init_cash,
        output_root=output_root,
    )


def evaluate_program(
    program: StrategyProgram,
    run_config: RunConfig,
    meta_policy: MetaPolicy,
    generation_dir: str | Path,
) -> EvaluationArtifacts:
    generation_dir = Path(generation_dir)
    stage_root = generation_dir / "stage_runs"
    full_root = generation_dir / "full_runs"

    stage_manifest = _run_manifest(
        program,
        run_config,
        stage_root,
        n_splits=max(1, min(run_config.n_splits, meta_policy.stage_n_splits)),
        candidate_count=min(program.candidate_count, meta_policy.stage_candidate_cap),
    )
    stage_assessment = _load_assessment_from_manifest(stage_manifest)
    stage_aggregate = stage_assessment["aggregate_snapshot"]
    stage_passed = (
        stage_manifest["verdict"] != "error"
        and stage_aggregate.get("median_test_score") is not None
        and float(stage_aggregate["median_test_score"]) >= meta_policy.stage_min_median_test_score
        and stage_aggregate.get("positive_test_ratio") is not None
        and float(stage_aggregate["positive_test_ratio"]) >= meta_policy.stage_min_positive_test_ratio
    )

    full_manifest: dict[str, Any] | None = None
    full_assessment: dict[str, Any] | None = None
    if stage_passed:
        full_manifest = _run_manifest(
            program,
            run_config,
            full_root,
            n_splits=run_config.n_splits,
            candidate_count=program.candidate_count,
        )
        full_assessment = _load_assessment_from_manifest(full_manifest)

    if full_manifest is not None and full_assessment is not None:
        score_value = score_from_assessment(full_assessment, full_assessment["verdict"])
        valid_parent = full_assessment["verdict"] in {"pass", "fail"}
        failed_checks = list(full_assessment.get("failed_checks", []))
        aggregate = dict(full_assessment["aggregate_snapshot"])
        selected_candidates = list(full_assessment.get("selected_candidates", []))
        full_verdict = full_assessment["verdict"]
    else:
        score_value = score_from_assessment(stage_assessment, stage_assessment["verdict"])
        if score_value is not None:
            score_value *= meta_policy.staged_score_frac
        valid_parent = False
        failed_checks = list(stage_assessment.get("failed_checks", []))
        aggregate = dict(stage_assessment["aggregate_snapshot"])
        selected_candidates = list(stage_assessment.get("selected_candidates", []))
        full_verdict = None

    return EvaluationArtifacts(
        stage_manifest_path=str(Path(stage_manifest["run_dir"]) / "run_manifest.json"),
        full_manifest_path=None if full_manifest is None else str(Path(full_manifest["run_dir"]) / "run_manifest.json"),
        stage_passed=stage_passed,
        full_verdict=full_verdict,
        score_value=score_value,
        valid_parent=valid_parent,
        failed_checks=failed_checks,
        aggregate=aggregate,
        selected_candidates=selected_candidates,
    )
