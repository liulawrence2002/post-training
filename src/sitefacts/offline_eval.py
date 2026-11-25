"""Score saved responses, or the canonical targets, without calling Tinker."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sitefacts.messages import canonical_answer
from sitefacts.records import SiteRecord
from sitefacts.reward import Score, score_response


def score_records(
    records: list[SiteRecord],
    responses: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Score ``responses`` keyed by example id.

    When ``responses`` is omitted, the gold answer is scored. That checks the
    scorer and the targets. It is not a model evaluation.
    """
    rows = []
    for record in records:
        if responses is None:
            text = canonical_answer(record)
            source = "canonical_target"
        else:
            if record.example_id not in responses:
                raise KeyError(f"No prediction for {record.example_id}")
            text = responses[record.example_id]
            source = "prediction"
        scored = score_response(
            text,
            site_id=record.site_id,
            observed_at=record.observed_at,
            facts=record.facts,
        )
        rows.append(_row(record, scored, source))
    return summarize(rows, source="canonical_target" if responses is None else "prediction")


def load_predictions(path: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if "example_id" not in raw or "response" not in raw:
            raise ValueError(f"{path}:{line_number} needs example_id and response")
        found[str(raw["example_id"])] = str(raw["response"])
    return found


def summarize(rows: list[dict[str, Any]], *, source: str) -> dict[str, Any]:
    count = len(rows) or 1
    return {
        "source": source,
        "n": len(rows),
        "mean_reward": sum(row["reward"] for row in rows) / count if rows else 0.0,
        "mean_fact_f1": sum(row["fact_f1"] for row in rows) / count if rows else 0.0,
        "schema_valid_rate": sum(row["schema_valid"] for row in rows) / count if rows else 0.0,
        "examples": rows,
    }


def _row(record: SiteRecord, scored: Score, source: str) -> dict[str, Any]:
    return {
        "example_id": record.example_id,
        "split": record.split,
        "source": source,
        "reward": scored.total,
        "fact_f1": scored.fact_f1,
        "provenance": scored.provenance,
        "schema_valid": scored.schema_valid,
        "site_id_match": scored.site_id_match,
        "errors": scored.errors,
    }
