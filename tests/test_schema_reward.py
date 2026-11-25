import json

import pytest

from sitefacts.messages import canonical_answer
from sitefacts.records import SiteRecord
from sitefacts.reward import extract_json_object, score_response
from sitefacts.sample_catalog import sample_rows
from sitefacts.schema import validate_answer


def _record() -> SiteRecord:
    return SiteRecord.from_dict(sample_rows()[0])


def test_canonical_answer_scores_one():
    record = _record()
    scored = score_response(
        canonical_answer(record),
        site_id=record.site_id,
        observed_at=record.observed_at,
        facts=record.facts,
    )
    assert scored.errors == []
    assert scored.total == pytest.approx(1.0)
    assert scored.schema_valid
    assert scored.fact_f1 == pytest.approx(1.0)
    assert scored.provenance == pytest.approx(1.0)


def test_invalid_json_and_schema_are_distinct():
    record = _record()
    empty = score_response(
        "I am not sure.",
        site_id=record.site_id,
        observed_at=record.observed_at,
        facts=record.facts,
    )
    assert empty.total == 0.0
    assert empty.json_parsed is False
    payload = record.answer_payload()
    del payload["facts"][0]["source"]
    broken = score_response(
        json.dumps(payload),
        site_id=record.site_id,
        observed_at=record.observed_at,
        facts=record.facts,
    )
    assert broken.json_parsed
    assert broken.schema_valid is False
    assert broken.total == pytest.approx(0.05)


def test_wrong_site_drops_fact_credit():
    record = _record()
    payload = record.answer_payload()
    payload["site_id"] = "some-other-site"
    scored = score_response(
        json.dumps(payload),
        site_id=record.site_id,
        observed_at=record.observed_at,
        facts=record.facts,
    )
    assert scored.schema_valid
    assert scored.site_id_match is False
    assert scored.fact_f1 == 0.0
    assert scored.total == pytest.approx(0.15)


def test_wrong_value_lowers_f1_and_fence_still_parses():
    record = _record()
    payload = record.answer_payload()
    payload["facts"][1]["value"] = "slate"
    scored = score_response(
        "Here is the object:\n```json\n" + json.dumps(payload) + "\n```\n",
        site_id=record.site_id,
        observed_at=record.observed_at,
        facts=record.facts,
    )
    assert scored.schema_valid
    assert scored.fact_f1 < 1.0
    assert scored.total < 1.0


def test_numeric_tolerance_and_schema_edges():
    record = SiteRecord.from_dict(sample_rows()[2])
    payload = record.answer_payload()
    for fact in payload["facts"]:
        if fact["claim"] == "vegetation_clearance_m":
            fact["value"] = 4.50
    scored = score_response(
        json.dumps(payload),
        site_id=record.site_id,
        observed_at=record.observed_at,
        facts=record.facts,
    )
    assert scored.total == pytest.approx(1.0)
    assert validate_answer({"site_id": "x", "observed_at": "2024-01-01T00:00:00Z", "facts": []}) == []
    errors = validate_answer(
        {
            "site_id": "x",
            "observed_at": "yesterday",
            "facts": [{"claim": "a", "value": True, "confidence": True, "source": {}}],
        }
    )
    assert errors


def test_extracts_last_facts_object_from_prose():
    payload, error = extract_json_object('thinking {"nope": 1} then {"facts": [], "site_id": "s"}')
    assert error is None
    assert payload is not None
    assert "facts" in payload
