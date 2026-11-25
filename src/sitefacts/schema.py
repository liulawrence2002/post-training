"""JSON schema for one site-level answer.

The object is the contract SFT targets, the RL reward, and eval all share.
It is intentionally small. The claim catalog, units, and time semantics are
not a customer schema yet — see the open questions in the README.

TODO(labels): replace ``analyst_note`` and ``site_context`` with the source
kinds a production pipeline can actually cite, once real evidence is wired up.
"""

from __future__ import annotations

import re
from typing import Any

SOURCE_KINDS = frozenset(
    {
        "imagery",
        "elevation",
        "weather",
        "ais",
        "adsb",
        "permits",
        "map",
        "analyst_note",
        "site_context",
    }
)

TOP_LEVEL_KEYS = frozenset({"site_id", "observed_at", "facts"})
FACT_KEYS = frozenset({"claim", "value", "unit", "confidence", "source"})
SOURCE_KEYS = frozenset({"kind", "uri", "observed_at"})

# Sample data and the reward use this exact UTC form so matches are stable.
ISO_ZULU = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


def validate_answer(payload: Any) -> list[str]:
    """Return a list of schema errors. An empty list means the payload is valid."""
    errors: list[str] = []
    if not isinstance(payload, dict):
        return ["answer must be a JSON object"]
    extra = sorted(set(payload) - TOP_LEVEL_KEYS)
    if extra:
        errors.append(f"unknown top-level keys: {extra}")
    site_id = payload.get("site_id")
    if not isinstance(site_id, str) or not site_id.strip():
        errors.append("site_id must be a non-empty string")
    observed = payload.get("observed_at")
    if not isinstance(observed, str) or ISO_ZULU.fullmatch(observed) is None:
        errors.append("observed_at must be a UTC timestamp like 2024-06-15T16:00:00Z")
    facts = payload.get("facts")
    if not isinstance(facts, list):
        errors.append("facts must be a list")
        return errors
    for index, fact in enumerate(facts):
        errors.extend(_validate_fact(index, fact))
    return errors


def _validate_fact(index: int, fact: Any) -> list[str]:
    prefix = f"facts[{index}]"
    if not isinstance(fact, dict):
        return [f"{prefix} must be an object"]
    errors: list[str] = []
    extra = sorted(set(fact) - FACT_KEYS)
    if extra:
        errors.append(f"{prefix} has unknown keys: {extra}")
    claim = fact.get("claim")
    if not isinstance(claim, str) or not claim.strip():
        errors.append(f"{prefix}.claim must be a non-empty string")
    if "value" not in fact or not _valid_value(fact.get("value")):
        errors.append(f"{prefix}.value must be a string, number, or boolean")
    if "unit" in fact and fact["unit"] is not None and not isinstance(fact["unit"], str):
        errors.append(f"{prefix}.unit must be a string or null")
    confidence = fact.get("confidence")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        errors.append(f"{prefix}.confidence must be a number in [0, 1]")
    elif not 0.0 <= float(confidence) <= 1.0:
        errors.append(f"{prefix}.confidence must be in [0, 1]")
    source = fact.get("source")
    if not isinstance(source, dict):
        errors.append(f"{prefix}.source must be an object")
        return errors
    extra_source = sorted(set(source) - SOURCE_KEYS)
    if extra_source:
        errors.append(f"{prefix}.source has unknown keys: {extra_source}")
    kind = source.get("kind")
    if kind not in SOURCE_KINDS:
        errors.append(f"{prefix}.source.kind must be one of {sorted(SOURCE_KINDS)}")
    uri = source.get("uri")
    if not isinstance(uri, str) or not uri.strip():
        errors.append(f"{prefix}.source.uri must be a non-empty string")
    source_time = source.get("observed_at")
    if not isinstance(source_time, str) or ISO_ZULU.fullmatch(source_time) is None:
        errors.append(f"{prefix}.source.observed_at must be a UTC timestamp")
    return errors


def _valid_value(value: Any) -> bool:
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value == value and value not in (float("inf"), float("-inf"))
    return isinstance(value, str) and bool(value.strip())
