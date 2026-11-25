"""Reward for a grounded, structured site-facts answer.

The score is a weighted sum of:

* schema validity of the JSON object
* fact F1 against the ground-truth claims and values
* provenance agreement on the facts that matched

A wrong ``site_id`` drops fact and provenance credit. Facts about a different
site are not partial credit.

Confidence is required by the schema and is not scored. The sample targets use
placeholder confidences. TODO(labels): score calibration only after labels
carry a real confidence definition.

This module is pure Python so tests and offline eval can run without an API key.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from sitefacts.schema import validate_answer

SCHEMA_WEIGHT = 0.15
FACT_WEIGHT = 0.70
PROVENANCE_WEIGHT = 0.15
# Parsed JSON that is not the schema still gets a small signal so RL can
# tell "some object" from "not JSON". It is far below a valid answer.
INVALID_SCHEMA_SCORE = 0.05
# A schema-valid answer whose top-level timestamp disagrees is scaled, not zeroed.
OBSERVED_AT_MISMATCH_SCALE = 0.9

_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL)


@dataclass
class Score:
    total: float
    json_parsed: bool
    schema_valid: bool
    fact_f1: float
    provenance: float
    site_id_match: bool
    observed_at_match: bool
    errors: list[str] = field(default_factory=list)

    def metrics(self) -> dict[str, float]:
        """Numeric metrics safe to log from an RL step."""
        return {
            "schema_valid": float(self.schema_valid),
            "fact_f1": self.fact_f1,
            "provenance": self.provenance,
            "site_id_match": float(self.site_id_match),
            "observed_at_match": float(self.observed_at_match),
            "correct": float(self.total >= 0.99),
        }


def score_response(
    text: str,
    *,
    site_id: str,
    observed_at: str,
    facts: list[dict[str, Any]],
) -> Score:
    payload, parse_error = extract_json_object(text)
    if payload is None:
        return Score(
            total=0.0,
            json_parsed=False,
            schema_valid=False,
            fact_f1=0.0,
            provenance=0.0,
            site_id_match=False,
            observed_at_match=False,
            errors=[parse_error or "no JSON object found"],
        )
    errors = validate_answer(payload)
    if errors:
        return Score(
            total=INVALID_SCHEMA_SCORE,
            json_parsed=True,
            schema_valid=False,
            fact_f1=0.0,
            provenance=0.0,
            site_id_match=payload.get("site_id") == site_id,
            observed_at_match=payload.get("observed_at") == observed_at,
            errors=errors,
        )
    site_match = payload["site_id"] == site_id
    time_match = payload["observed_at"] == observed_at
    if site_match:
        fact_f1, provenance = _match_facts(payload["facts"], facts)
    else:
        # Do not credit claims once the answer is about a different site.
        fact_f1, provenance = 0.0, 0.0
        errors = ["site_id does not match the example"]
    total = SCHEMA_WEIGHT + FACT_WEIGHT * fact_f1 + PROVENANCE_WEIGHT * provenance
    if not time_match:
        total *= OBSERVED_AT_MISMATCH_SCALE
        errors = [*errors, "observed_at does not match the example"]
    return Score(
        total=total,
        json_parsed=True,
        schema_valid=True,
        fact_f1=fact_f1,
        provenance=provenance,
        site_id_match=site_match,
        observed_at_match=time_match,
        errors=errors,
    )


def extract_json_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    """Pull the answer object out of a model response.

    Prefers a full-string parse, then a fenced block, then the last balanced
    object that contains a ``facts`` key (answers often follow prose).
    """
    stripped = text.strip()
    if not stripped:
        return None, "empty response"
    loaded, error = _loads_object(stripped)
    if loaded is not None:
        return loaded, None
    fenced = _FENCE.findall(stripped)
    for block in reversed(fenced):
        loaded, _fence_error = _loads_object(block)
        if loaded is not None:
            return loaded, None
    candidates = _balanced_objects(stripped)
    with_facts = [item for item in candidates if isinstance(item, dict) and "facts" in item]
    pool = with_facts or [item for item in candidates if isinstance(item, dict)]
    if pool:
        return pool[-1], None
    return None, error or "no JSON object found"


def values_match(predicted: Any, gold: Any) -> bool:
    """Compare fact values. Numbers match within 1 percent. Booleans stay booleans."""
    if isinstance(gold, bool) or isinstance(predicted, bool):
        return predicted is gold or str(predicted).strip().casefold() == str(gold).strip().casefold()
    if isinstance(gold, (int, float)) and not isinstance(gold, bool):
        try:
            predicted_number = float(predicted)
        except (TypeError, ValueError):
            return False
        tolerance = max(1e-6, abs(float(gold)) * 0.01)
        return abs(predicted_number - float(gold)) <= tolerance
    return str(predicted).strip().casefold() == str(gold).strip().casefold()


def _match_facts(
    predicted: list[dict[str, Any]], gold: list[dict[str, Any]]
) -> tuple[float, float]:
    if not predicted and not gold:
        return 1.0, 1.0
    if not predicted or not gold:
        return 0.0, 0.0
    used: set[int] = set()
    provenance_hits = 0
    true_positives = 0
    for gold_fact in gold:
        for index, pred_fact in enumerate(predicted):
            if index in used:
                continue
            if not _fact_match(pred_fact, gold_fact):
                continue
            used.add(index)
            true_positives += 1
            if _provenance_match(pred_fact, gold_fact):
                provenance_hits += 1
            break
    precision = true_positives / len(predicted)
    recall = true_positives / len(gold)
    if precision + recall == 0:
        fact_f1 = 0.0
    else:
        fact_f1 = 2 * precision * recall / (precision + recall)
    provenance = provenance_hits / true_positives if true_positives else 0.0
    return fact_f1, provenance


def _fact_match(predicted: dict[str, Any], gold: dict[str, Any]) -> bool:
    if str(predicted.get("claim", "")).strip() != str(gold.get("claim", "")).strip():
        return False
    if not values_match(predicted.get("value"), gold.get("value")):
        return False
    pred_unit = predicted.get("unit")
    gold_unit = gold.get("unit")
    if pred_unit and gold_unit and str(pred_unit).strip().casefold() != str(gold_unit).strip().casefold():
        return False
    return True


def _provenance_match(predicted: dict[str, Any], gold: dict[str, Any]) -> bool:
    pred_source = predicted.get("source")
    gold_source = gold.get("source")
    if not isinstance(pred_source, dict) or not isinstance(gold_source, dict):
        return False
    return (
        pred_source.get("kind") == gold_source.get("kind")
        and pred_source.get("uri") == gold_source.get("uri")
        and pred_source.get("observed_at") == gold_source.get("observed_at")
    )


def _loads_object(text: str) -> tuple[dict[str, Any] | None, str | None]:
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError as exc:
        return None, str(exc)
    if isinstance(loaded, dict):
        return loaded, None
    return None, "JSON value is not an object"


def _balanced_objects(text: str) -> list[Any]:
    found: list[Any] = []
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
            continue
        if char == "{":
            if depth == 0:
                start = index
            depth += 1
        elif char == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                loaded, _error = _loads_object(text[start : index + 1])
                if loaded is not None:
                    found.append(loaded)
                start = None
    return found
