from __future__ import annotations

import json
import math
import random
from dataclasses import replace
from pathlib import Path

from examples.hyperagent_vbt.models import GenerationRecord


def load_archive(path: str | Path) -> list[GenerationRecord]:
    path = Path(path)
    if not path.exists():
        return []
    records: list[GenerationRecord] = []
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(GenerationRecord.from_dict(json.loads(line)))
    return hydrate_child_counts(records)


def append_archive_record(path: str | Path, record: GenerationRecord) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps(record.to_dict(), ensure_ascii=False) + "\n")


def hydrate_child_counts(records: list[GenerationRecord]) -> list[GenerationRecord]:
    child_counts: dict[int, int] = {}
    for record in records:
        if record.parent_gen_id is not None:
            child_counts[record.parent_gen_id] = child_counts.get(record.parent_gen_id, 0) + 1
    return [replace(record, child_count=child_counts.get(record.gen_id, 0)) for record in records]


def archive_best_record(records: list[GenerationRecord]) -> GenerationRecord | None:
    candidates = [record for record in records if record.score_value is not None]
    if not candidates:
        return None
    return max(candidates, key=lambda record: float(record.score_value or float("-inf")))


def top_records_for_context(records: list[GenerationRecord], top_k: int) -> list[GenerationRecord]:
    candidates = [record for record in records if record.score_value is not None]
    candidates.sort(key=lambda record: float(record.score_value or float("-inf")), reverse=True)
    return candidates[:top_k]


def _valid_candidates(records: list[GenerationRecord]) -> list[GenerationRecord]:
    return [record for record in records if record.valid_parent and record.score_value is not None]


def _sigmoid_weighted_scores(records: list[GenerationRecord]) -> list[float]:
    raw_scores = [float(record.score_value or 0.0) for record in records]
    if not raw_scores:
        return []
    midpoint = sum(sorted(raw_scores, reverse=True)[: min(3, len(raw_scores))]) / min(3, len(raw_scores))
    return [1.0 / (1.0 + math.exp(-10.0 * (score - midpoint))) for score in raw_scores]


def select_parent(
    records: list[GenerationRecord],
    method: str,
    *,
    rng: random.Random,
) -> GenerationRecord | None:
    records = hydrate_child_counts(records)
    candidates = _valid_candidates(records)
    if not candidates:
        return None

    method = method.strip().lower()
    if method == "latest":
        return candidates[-1]
    if method == "random":
        return rng.choice(candidates)
    if method == "best":
        return max(candidates, key=lambda record: float(record.score_value or float("-inf")))
    if method == "score_prop":
        weights = _sigmoid_weighted_scores(candidates)
        return rng.choices(candidates, weights=weights, k=1)[0]
    if method == "score_child_prop":
        score_weights = _sigmoid_weighted_scores(candidates)
        penalties = [math.exp(-((record.child_count / 8.0) ** 3)) for record in candidates]
        weights = [score * penalty for score, penalty in zip(score_weights, penalties)]
        return rng.choices(candidates, weights=weights, k=1)[0]

    raise ValueError(
        f"Unsupported parent-selection method {method!r}. "
        "Choose one of random, latest, best, score_prop, score_child_prop."
    )
