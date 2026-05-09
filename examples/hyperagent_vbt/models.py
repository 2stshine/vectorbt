from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _stable_search_space(search_space: dict[str, list[int | float]]) -> dict[str, list[int | float]]:
    normalized: dict[str, list[int | float]] = {}
    for key, values in sorted(search_space.items()):
        unique_values = sorted(set(values))
        normalized[key] = [int(value) if isinstance(value, float) and value.is_integer() else value for value in unique_values]
    return normalized


@dataclass(frozen=True)
class StrategyProgram:
    family: str
    strategy_name: str
    window_spec: str | None
    search_space: dict[str, list[int | float]]
    candidate_count: int = 8
    min_trades: int = 1
    selection_metric: str = "sharpe_ratio"
    direction: str = "both"
    hypothesis_ko: str = "초기 전략 가설입니다."
    rationale_ko: str = "초기 전략 제안입니다."
    notes_ko: list[str] = field(default_factory=list)
    source: str = "seed"

    @property
    def program_id(self) -> str:
        payload = json.dumps(
            {
                "family": self.family,
                "strategy_name": self.strategy_name,
                "window_spec": self.window_spec,
                "search_space": _stable_search_space(self.search_space),
                "candidate_count": self.candidate_count,
                "min_trades": self.min_trades,
                "selection_metric": self.selection_metric,
                "direction": self.direction,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
        return f"{self.family}-{digest}"

    def to_pipeline_kwargs(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "strategy_name": self.strategy_name,
            "window_spec": self.window_spec,
            "search_space": _stable_search_space(self.search_space),
            "candidate_count": self.candidate_count,
            "min_trades": self.min_trades,
            "selection_metric": self.selection_metric,
            "direction": self.direction,
        }

    def signature(self) -> tuple[Any, ...]:
        payload = self.to_pipeline_kwargs().copy()
        payload["search_space"] = json.dumps(payload["search_space"], sort_keys=True)
        return tuple(payload[key] for key in sorted(payload))

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["search_space"] = _stable_search_space(self.search_space)
        payload["program_id"] = self.program_id
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StrategyProgram":
        payload = payload.copy()
        payload.pop("program_id", None)
        payload["search_space"] = _stable_search_space(payload["search_space"])
        return cls(**payload)


@dataclass(frozen=True)
class MetaPolicy:
    parent_selection: str = "score_child_prop"
    stage_min_median_test_score: float = -0.05
    stage_min_positive_test_ratio: float = 0.25
    stage_candidate_cap: int = 6
    stage_n_splits: int = 2
    staged_score_frac: float = 0.35
    exploration_prob: float = 0.35
    family_switch_after_failures: int = 2
    max_candidate_count: int = 16
    min_candidate_count: int = 4
    top_k_archive_context: int = 6
    focus_on_selected: bool = True
    widen_after_failure: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MetaPolicy":
        return cls(**payload)


@dataclass(frozen=True)
class MemoryState:
    promising_families: dict[str, int] = field(default_factory=dict)
    weak_families: dict[str, int] = field(default_factory=dict)
    family_last_seen: dict[str, int] = field(default_factory=dict)
    success_patterns_ko: list[str] = field(default_factory=list)
    failure_patterns_ko: list[str] = field(default_factory=list)
    observations_ko: list[str] = field(default_factory=list)
    meta_reflections_ko: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "MemoryState":
        return cls(**payload)


@dataclass(frozen=True)
class RunConfig:
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
    direction: str = "both"
    fees: float = 0.00015
    init_cash: float = 1_000.0
    max_generations: int = 12
    llm_enabled: bool = False
    llm_model: str = "gpt-4o-mini"
    llm_api_key_env: str = "OPENAI_API_KEY"
    llm_base_url: str = "https://api.openai.com/v1"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EvaluationArtifacts:
    stage_manifest_path: str | None
    full_manifest_path: str | None
    stage_passed: bool
    full_verdict: str | None
    score_value: float | None
    valid_parent: bool
    failed_checks: list[str]
    aggregate: dict[str, Any] | None
    selected_candidates: list[dict[str, Any]]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GenerationRecord:
    gen_id: int
    created_at: str
    parent_gen_id: int | None
    parent_selection_method: str
    program: dict[str, Any]
    task_rationale_ko: str
    meta_reflection_ko: str
    stage_manifest_path: str | None
    full_manifest_path: str | None
    stage_passed: bool
    full_verdict: str | None
    score_value: float | None
    valid_parent: bool
    failed_checks: list[str] = field(default_factory=list)
    aggregate: dict[str, Any] | None = None
    selected_candidates: list[dict[str, Any]] = field(default_factory=list)
    llm_used_task: bool = False
    llm_used_meta: bool = False
    child_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "GenerationRecord":
        return cls(**payload)
