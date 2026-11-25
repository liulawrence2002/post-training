"""Synthetic site episodes used to exercise the pipeline.

These are format examples, not inspections. The only fact a human can check
from the image file is ``placeholder_swatch_color``. Domain claims are copied
from the analyst notes in the prompt, or, for the port episode, from the
tool stub. Coordinates are a fake grid.

TODO(data): replace this catalog with licensed imagery and reviewed labels.
"""

from __future__ import annotations

from typing import Any

SWATCH_RGB: dict[str, tuple[int, int, int]] = {
    "brown": (139, 90, 43),
    "blue": (40, 90, 170),
    "green": (40, 130, 70),
    "gray": (128, 128, 128),
    "orange": (210, 120, 40),
    "red": (170, 50, 50),
}


def _source(kind: str, uri: str, observed_at: str) -> dict[str, str]:
    return {"kind": kind, "uri": uri, "observed_at": observed_at}


def _fact(
    claim: str,
    value: Any,
    unit: str | None,
    confidence: float,
    source: dict[str, str],
) -> dict[str, Any]:
    return {
        "claim": claim,
        "value": value,
        "unit": unit,
        "confidence": confidence,
        "source": source,
    }


def _episode(
    *,
    example_id: str,
    split: str,
    site_id: str,
    observed_at: str,
    asset_type: str,
    latitude: float,
    longitude: float,
    question: str,
    analyst_notes: list[str],
    swatch_color: str,
    context: dict[str, Any],
    tool_trace: list[dict[str, Any]],
    domain_facts: list[dict[str, Any]],
) -> dict[str, Any]:
    image = f"images/{example_id}.png"
    swatch = _fact(
        "placeholder_swatch_color",
        swatch_color,
        None,
        0.9,
        _source("imagery", f"sample://images/{example_id}.png", observed_at),
    )
    return {
        "example_id": example_id,
        "split": split,
        "label_status": "synthetic_sample",
        "site_id": site_id,
        "observed_at": observed_at,
        "asset_type": asset_type,
        "latitude": latitude,
        "longitude": longitude,
        "coordinate_status": "illustrative_not_surveyed",
        "question": question,
        "analyst_notes": analyst_notes,
        "image": image,
        "swatch_color": swatch_color,
        "context": context,
        "tool_trace": tool_trace,
        "facts": [swatch, *domain_facts],
        "notes": (
            "SYNTHETIC SAMPLE. Not a real parcel, inspection, or sensor reading. "
            "The PNG is a solid color swatch. Domain facts repeat the analyst notes "
            "or the tool stub. TODO(labels): replace before any training run you would trust."
        ),
    }


def sample_rows() -> list[dict[str, Any]]:
    roof_time = "2024-06-15T16:00:00Z"
    flood_time = "2024-07-02T18:30:00Z"
    veg_time = "2024-05-20T15:00:00Z"
    port_time = "2024-08-11T12:00:00Z"
    wui_time = "2024-09-01T17:00:00Z"
    yard_time = "2024-04-18T14:00:00Z"
    return [
        _episode(
            example_id="sample-roof-001",
            split="train",
            site_id="sample-site-roof-001",
            observed_at=roof_time,
            asset_type="single-family home",
            latitude=0.001,
            longitude=0.001,
            question=(
                "Report the roof claims supported by the analyst notes, "
                "and placeholder_swatch_color from the swatch."
            ),
            analyst_notes=[
                "Roof covering is asphalt_shingle.",
                "The covering is intact.",
            ],
            swatch_color="brown",
            context={},
            tool_trace=[],
            domain_facts=[
                _fact(
                    "roof_covering",
                    "asphalt_shingle",
                    None,
                    0.6,
                    _source("analyst_note", "sample://notes/sample-site-roof-001", roof_time),
                ),
                _fact(
                    "roof_covering_intact",
                    True,
                    None,
                    0.6,
                    _source("analyst_note", "sample://notes/sample-site-roof-001", roof_time),
                ),
            ],
        ),
        _episode(
            example_id="sample-flood-002",
            split="train",
            site_id="sample-site-flood-002",
            observed_at=flood_time,
            asset_type="warehouse yard",
            latitude=0.002,
            longitude=0.002,
            question=(
                "Report whether standing water is visible according to the notes, "
                "and placeholder_swatch_color from the swatch."
            ),
            analyst_notes=["Standing water is visible on the paved yard."],
            swatch_color="blue",
            context={},
            tool_trace=[],
            domain_facts=[
                _fact(
                    "standing_water_visible",
                    True,
                    None,
                    0.6,
                    _source("analyst_note", "sample://notes/sample-site-flood-002", flood_time),
                )
            ],
        ),
        _episode(
            example_id="sample-veg-003",
            split="train",
            site_id="sample-site-veg-003",
            observed_at=veg_time,
            asset_type="distribution corridor",
            latitude=0.003,
            longitude=0.003,
            question=(
                "Report vegetation clearance from the notes, "
                "and placeholder_swatch_color from the swatch."
            ),
            analyst_notes=["Vegetation is 4.5 m from the conductor."],
            swatch_color="green",
            context={},
            tool_trace=[],
            domain_facts=[
                _fact(
                    "vegetation_clearance_m",
                    4.5,
                    "m",
                    0.6,
                    _source("analyst_note", "sample://notes/sample-site-veg-003", veg_time),
                )
            ],
        ),
        _episode(
            example_id="sample-port-004",
            split="train",
            site_id="sample-site-port-004",
            observed_at=port_time,
            asset_type="port terminal",
            latitude=0.004,
            longitude=0.004,
            question=(
                "Report elevation_m and vessels_within_500m. "
                "Those values are not in the notes; call lookup_site_context. "
                "Also report placeholder_swatch_color from the swatch."
            ),
            analyst_notes=[
                "This is a port terminal asset. Elevation and the vessel count are not written here."
            ],
            swatch_color="gray",
            context={"elevation_m": 12.4, "vessels_within_500m": 2},
            tool_trace=[
                {
                    "name": "lookup_site_context",
                    "arguments": {
                        "fields": ["elevation_m", "vessels_within_500m"],
                        "site_id": "sample-site-port-004",
                    },
                }
            ],
            domain_facts=[
                _fact(
                    "elevation_m",
                    12.4,
                    "m",
                    0.8,
                    _source("site_context", "sample://context/sample-site-port-004", port_time),
                ),
                _fact(
                    "vessels_within_500m",
                    2,
                    None,
                    0.8,
                    _source("site_context", "sample://context/sample-site-port-004", port_time),
                ),
            ],
        ),
        _episode(
            example_id="sample-wui-005",
            split="test",
            site_id="sample-site-wui-005",
            observed_at=wui_time,
            asset_type="wildland-interface home",
            latitude=0.005,
            longitude=0.005,
            question=(
                "Report defensible space from the notes, "
                "and placeholder_swatch_color from the swatch."
            ),
            analyst_notes=["Defensible space is 8 m."],
            swatch_color="orange",
            context={},
            tool_trace=[],
            domain_facts=[
                _fact(
                    "defensible_space_m",
                    8,
                    "m",
                    0.6,
                    _source("analyst_note", "sample://notes/sample-site-wui-005", wui_time),
                )
            ],
        ),
        _episode(
            example_id="sample-yard-006",
            split="test",
            site_id="sample-site-yard-006",
            observed_at=yard_time,
            asset_type="distribution substation",
            latitude=0.006,
            longitude=0.006,
            question=(
                "Report whether the equipment yard is clear according to the notes, "
                "and placeholder_swatch_color from the swatch."
            ),
            analyst_notes=["The equipment yard is clear of stored materials."],
            swatch_color="red",
            context={},
            tool_trace=[],
            domain_facts=[
                _fact(
                    "equipment_yard_clear",
                    True,
                    None,
                    0.6,
                    _source("analyst_note", "sample://notes/sample-site-yard-006", yard_time),
                )
            ],
        ),
    ]
