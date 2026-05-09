from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.agent_pipeline.common import save_json, timestamped_run_dir, update_latest_pointer
from examples.agent_pipeline.data_stage import SUPPORTED_DATA_SOURCES, default_symbol_for_source
from examples.hyperagent_vbt.archive import append_archive_record, archive_best_record, hydrate_child_counts, select_parent
from examples.hyperagent_vbt.evaluator import evaluate_program
from examples.hyperagent_vbt.memory import load_memory, save_memory, summarize_memory, update_memory
from examples.hyperagent_vbt.meta_agent import MetaAgent, build_meta_agent_settings
from examples.hyperagent_vbt.models import GenerationRecord, MemoryState, MetaPolicy, RunConfig, StrategyProgram, utc_now_iso
from examples.hyperagent_vbt.reporting import render_generation_summary_ko, render_loop_summary_ko
from examples.hyperagent_vbt.task_agent import TaskAgent, build_task_agent_settings


DEFAULT_OUTPUT_ROOT = Path("examples") / "hyperagent_vbt_runs"


def save_meta_policy(meta_policy: MetaPolicy, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(meta_policy.to_dict(), indent=2, ensure_ascii=False))


def build_manifest(
    loop_dir: Path,
    run_config: RunConfig,
    meta_policy: MetaPolicy,
    memory: MemoryState,
    records: list[GenerationRecord],
    stop_reason: str,
) -> dict[str, object]:
    best_record = archive_best_record(records)
    return {
        "loop_dir": str(loop_dir),
        "run_config": run_config.to_dict(),
        "meta_policy": meta_policy.to_dict(),
        "memory": memory.to_dict(),
        "stop_reason": stop_reason,
        "archive_path": str(loop_dir / "archive.jsonl"),
        "best_gen_id": None if best_record is None else best_record.gen_id,
        "best_score": None if best_record is None else best_record.score_value,
        "history": [record.to_dict() for record in records],
    }


def save_loop_reports(
    loop_dir: Path,
    records: list[GenerationRecord],
    memory: MemoryState,
    meta_policy: MetaPolicy,
    stop_reason: str,
) -> None:
    summary_path = loop_dir / "loop_summary_ko.md"
    summary_path.write_text(render_loop_summary_ko(loop_dir, records, memory, meta_policy, stop_reason))
    save_json({"markdown": summary_path.read_text()}, loop_dir / "loop_summary_ko.json")


def fallback_proposal_record(records: list[GenerationRecord]) -> GenerationRecord | None:
    if not records:
        return None
    scored = [record for record in records if record.score_value is not None]
    if scored:
        return max(scored, key=lambda record: float(record.score_value or float("-inf")))
    return records[-1]


def run_hyperagent_loop(
    run_config: RunConfig,
    *,
    output_root: str | Path = DEFAULT_OUTPUT_ROOT,
    seed_family: str = "ma_crossover_ls",
) -> dict[str, object]:
    loop_dir = timestamped_run_dir(output_root, run_config.symbol)
    archive_path = loop_dir / "archive.jsonl"
    memory_path = loop_dir / "memory.json"
    meta_policy_path = loop_dir / "meta_policy.json"

    memory = load_memory(memory_path)
    meta_policy = MetaPolicy()
    task_agent = TaskAgent(build_task_agent_settings(run_config))
    meta_agent = MetaAgent(build_meta_agent_settings(run_config))
    rng = random.Random(42)
    records: list[GenerationRecord] = []
    stop_reason = "max_generations_reached"

    for gen_id in range(run_config.max_generations):
        hydrated_records = hydrate_child_counts(records)
        selected_parent_record = select_parent(hydrated_records, meta_policy.parent_selection, rng=rng) if hydrated_records else None
        proposal_record = selected_parent_record or fallback_proposal_record(hydrated_records)
        parent_program = None if proposal_record is None else StrategyProgram.from_dict(proposal_record.program)
        memory_summary = summarize_memory(memory)

        proposed_program, llm_used_task = task_agent.propose(
            parent_program,
            proposal_record,
            hydrated_records,
            meta_policy,
            run_config,
            memory_summary,
            seed_family=seed_family,
            rng=rng,
        )

        generation_dir = loop_dir / f"gen_{gen_id:04d}"
        generation_dir.mkdir(parents=True, exist_ok=True)
        save_json(proposed_program.to_dict(), generation_dir / "proposal.json")

        evaluation = evaluate_program(proposed_program, run_config, meta_policy, generation_dir)

        provisional_record = GenerationRecord(
            gen_id=gen_id,
            created_at=utc_now_iso(),
            parent_gen_id=None if proposal_record is None else proposal_record.gen_id,
            parent_selection_method=meta_policy.parent_selection,
            program=proposed_program.to_dict(),
            task_rationale_ko=proposed_program.rationale_ko,
            meta_reflection_ko="",
            stage_manifest_path=evaluation.stage_manifest_path,
            full_manifest_path=evaluation.full_manifest_path,
            stage_passed=evaluation.stage_passed,
            full_verdict=evaluation.full_verdict,
            score_value=evaluation.score_value,
            valid_parent=evaluation.valid_parent,
            failed_checks=evaluation.failed_checks,
            aggregate=evaluation.aggregate,
            selected_candidates=evaluation.selected_candidates,
            llm_used_task=llm_used_task,
            llm_used_meta=False,
        )

        updated_policy, reflection, llm_used_meta = meta_agent.reflect(
            provisional_record,
            hydrated_records + [provisional_record],
            memory_summary,
            meta_policy,
            run_config,
        )
        final_record = replace(
            provisional_record,
            meta_reflection_ko=reflection,
            llm_used_meta=llm_used_meta,
        )
        records = hydrate_child_counts(records + [final_record])
        memory = update_memory(memory, final_record)
        meta_policy = updated_policy

        append_archive_record(archive_path, final_record)
        save_memory(memory, memory_path)
        save_meta_policy(meta_policy, meta_policy_path)

        generation_summary_path = generation_dir / "generation_summary_ko.md"
        generation_summary_path.write_text(render_generation_summary_ko(loop_dir, final_record, meta_policy))

        if final_record.full_verdict == "pass":
            stop_reason = "passed"
            save_loop_reports(loop_dir, records, memory, meta_policy, stop_reason)
            break
    else:
        stop_reason = "max_generations_reached"

    manifest = build_manifest(loop_dir, run_config, meta_policy, memory, records, stop_reason)
    save_json(manifest, loop_dir / "hyperagent_manifest.json")
    save_loop_reports(loop_dir, records, memory, meta_policy, stop_reason)
    manifest["latest"] = update_latest_pointer(
        output_root,
        loop_dir,
        summary_name="loop_summary_ko.md",
        manifest_name="hyperagent_manifest.json",
    )
    save_json(manifest, loop_dir / "hyperagent_manifest.json")
    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a HyperAgent-inspired automatic strategy generator on top of vectorbt.")
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
    parser.add_argument("--direction", default="both", choices=("longonly", "shortonly", "both"))
    parser.add_argument("--fees", type=float, default=0.00015)
    parser.add_argument("--init-cash", type=float, default=1_000.0)
    parser.add_argument("--max-generations", type=int, default=8)
    parser.add_argument("--llm-enable", action="store_true")
    parser.add_argument("--llm-model", default="gpt-4o-mini")
    parser.add_argument("--llm-api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--llm-base-url", default="https://api.openai.com/v1")
    parser.add_argument("--seed-family", default="ma_crossover_ls")
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser


def main() -> int:
    parser = build_argument_parser()
    args = parser.parse_args()
    symbol = args.symbol or default_symbol_for_source(args.data_source)
    run_config = RunConfig(
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
        direction=args.direction,
        fees=args.fees,
        init_cash=args.init_cash,
        max_generations=args.max_generations,
        llm_enabled=args.llm_enable,
        llm_model=args.llm_model,
        llm_api_key_env=args.llm_api_key_env,
        llm_base_url=args.llm_base_url,
    )
    manifest = run_hyperagent_loop(
        run_config,
        output_root=args.output_root,
        seed_family=args.seed_family,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
