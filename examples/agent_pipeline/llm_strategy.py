from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

import requests

from examples.agent_pipeline.strategy_registry import (
    SUPPORTED_STRATEGY_FAMILIES,
    normalize_family,
    sanitize_search_space,
    strategy_family_catalog,
)


DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


@dataclass(frozen=True)
class LLMStrategySettings:
    enabled: bool = False
    model: str = DEFAULT_OPENAI_MODEL
    api_key_env: str = "OPENAI_API_KEY"
    base_url: str = DEFAULT_OPENAI_BASE_URL
    timeout_seconds: int = 90


def llm_key_available(settings: LLMStrategySettings) -> bool:
    return bool(os.environ.get(settings.api_key_env))


def build_llm_history_payload(history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload: list[dict[str, Any]] = []
    for record in history[-6:]:
        assessment = record.get("assessment")
        aggregate = None if assessment is None else assessment.get("aggregate_snapshot", {})
        payload.append(
            {
                "iteration": record["iteration"],
                "verdict": record["verdict"],
                "proposal": {
                    "family": record["proposal"].get("family"),
                    "strategy_name": record["proposal"].get("strategy_name"),
                    "window_spec": record["proposal"].get("window_spec"),
                    "search_space": record["proposal"].get("search_space"),
                    "candidate_count": record["proposal"].get("candidate_count"),
                    "min_trades": record["proposal"].get("min_trades"),
                    "selection_metric": record["proposal"].get("selection_metric"),
                },
                "aggregate": aggregate,
                "failed_checks": [] if assessment is None else assessment.get("failed_checks", []),
                "selected_candidates": [] if assessment is None else assessment.get("selected_candidates", []),
                "error_message": record.get("error_message"),
                "rationale": record.get("rationale"),
            }
        )
    return payload


def build_schema() -> dict[str, Any]:
    return {
        "name": "strategy_proposal",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {
                "family": {
                    "type": "string",
                    "enum": list(SUPPORTED_STRATEGY_FAMILIES),
                },
                "strategy_name": {
                    "type": "string",
                },
                "search_space": {
                    "type": "object",
                    "additionalProperties": {
                        "type": "array",
                        "items": {
                            "type": "number",
                        },
                    },
                },
                "candidate_count": {
                    "type": "integer",
                    "minimum": 3,
                    "maximum": 30,
                },
                "min_trades": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 20,
                },
                "selection_metric": {
                    "type": "string",
                    "enum": ["sharpe_ratio", "total_return"],
                },
                "direction": {
                    "type": "string",
                    "enum": ["longonly", "shortonly", "both"],
                },
                "rationale_ko": {
                    "type": "string",
                },
                "mutation_notes_ko": {
                    "type": "array",
                    "items": {
                        "type": "string",
                    },
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
                "rationale_ko",
                "mutation_notes_ko",
            ],
            "additionalProperties": False,
        },
    }


def extract_message_text(payload: dict[str, Any]) -> str:
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


def infer_window_spec(search_space: dict[str, list[int | float]], family: str) -> str | None:
    if family not in {"ma_crossover_ls", "ma_rsi_filter_ls"}:
        return None
    window_values = set()
    for key in ("fast_window", "slow_window"):
        for value in search_space.get(key, []):
            window_values.add(int(value))
    if not window_values:
        return None
    return ",".join(str(value) for value in sorted(window_values))


def request_llm_strategy_proposal(
    proposal: dict[str, Any],
    history: list[dict[str, Any]],
    *,
    interval: str,
    settings: LLMStrategySettings,
) -> dict[str, Any] | None:
    if not settings.enabled:
        return None

    api_key = os.environ.get(settings.api_key_env)
    if not api_key:
        return None

    history_payload = build_llm_history_payload(history)
    family_catalog = strategy_family_catalog(interval)
    system_prompt = (
        "You are a quantitative crypto futures strategy research assistant. "
        "Given prior walk-forward validation results, propose the next strategy search space. "
        "Favor robust validation/test behavior over in-sample performance, reduce overfitting, and try to beat long hold more consistently. "
        "Return Korean rationale strings. Use only supported families and valid parameter keys from the provided catalog. "
        "Prefer explicit long/short strategies and keep candidate_count practical."
    )
    user_payload = {
        "current_proposal": proposal,
        "history": history_payload,
        "family_catalog": family_catalog,
        "instructions": {
            "goal": "Suggest the next strategy family and search space for the next walk-forward experiment.",
            "must_consider": [
                "hold underperformance",
                "train/validation/test gap",
                "validation robustness",
                "trade count realism",
            ],
            "preferred_direction": "both",
            "language": "Korean rationale and notes",
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
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
            "response_format": {
                "type": "json_schema",
                "json_schema": build_schema(),
            },
        },
        timeout=settings.timeout_seconds,
    )
    response.raise_for_status()

    raw_proposal = json.loads(extract_message_text(response.json()))
    family = normalize_family(raw_proposal["family"])
    search_space = sanitize_search_space(
        family,
        raw_proposal["search_space"],
        interval=interval,
        window_spec=None,
    )
    return {
        "family": family,
        "strategy_name": raw_proposal["strategy_name"],
        "search_space": search_space,
        "window_spec": infer_window_spec(search_space, family),
        "candidate_count": int(raw_proposal["candidate_count"]),
        "min_trades": int(raw_proposal["min_trades"]),
        "selection_metric": raw_proposal["selection_metric"],
        "direction": raw_proposal["direction"],
        "rationale": raw_proposal["rationale_ko"],
        "mutation_notes": [str(item) for item in raw_proposal["mutation_notes_ko"]],
        "llm_model": settings.model,
    }
