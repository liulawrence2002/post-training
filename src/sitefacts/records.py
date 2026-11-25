"""Labeled site examples on disk.

Each JSONL row is one episode. ``label_status`` distinguishes synthetic
training rows from unlabeled public downloads. Training loads only
``synthetic_sample`` rows in the ``train`` and ``test`` splits.

TODO(data): real rows need a site key, an observation time, imagery bytes
that were actually sensed then, and labels a reviewer stands behind.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

SYNTHETIC_STATUS = "synthetic_sample"
UNLABELED_PUBLIC_STATUS = "unlabeled_public"
TRAIN_SPLITS = frozenset({"train", "test"})


@dataclass
class SiteRecord:
    example_id: str
    split: str
    label_status: str
    site_id: str
    observed_at: str
    asset_type: str
    latitude: float | None
    longitude: float | None
    coordinate_status: str
    question: str
    analyst_notes: list[str]
    image: str | None
    swatch_color: str | None
    context: dict[str, Any]
    tool_trace: list[dict[str, Any]]
    facts: list[dict[str, Any]]
    notes: str

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SiteRecord:
        required = (
            "example_id",
            "split",
            "label_status",
            "site_id",
            "observed_at",
            "asset_type",
            "question",
            "facts",
        )
        missing = [key for key in required if key not in raw]
        if missing:
            raise ValueError(f"Example is missing keys {missing}")
        return cls(
            example_id=str(raw["example_id"]),
            split=str(raw["split"]),
            label_status=str(raw["label_status"]),
            site_id=str(raw["site_id"]),
            observed_at=str(raw["observed_at"]),
            asset_type=str(raw["asset_type"]),
            latitude=_maybe_float(raw.get("latitude")),
            longitude=_maybe_float(raw.get("longitude")),
            coordinate_status=str(raw.get("coordinate_status", "illustrative_not_surveyed")),
            question=str(raw["question"]),
            analyst_notes=[str(note) for note in raw.get("analyst_notes", [])],
            image=raw.get("image"),
            swatch_color=raw.get("swatch_color"),
            context=dict(raw.get("context") or {}),
            tool_trace=list(raw.get("tool_trace") or []),
            facts=list(raw["facts"]),
            notes=str(raw.get("notes", "")),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "example_id": self.example_id,
            "split": self.split,
            "label_status": self.label_status,
            "site_id": self.site_id,
            "observed_at": self.observed_at,
            "asset_type": self.asset_type,
            "latitude": self.latitude,
            "longitude": self.longitude,
            "coordinate_status": self.coordinate_status,
            "question": self.question,
            "analyst_notes": list(self.analyst_notes),
            "image": self.image,
            "swatch_color": self.swatch_color,
            "context": self.context,
            "tool_trace": self.tool_trace,
            "facts": self.facts,
            "notes": self.notes,
        }

    def answer_payload(self) -> dict[str, Any]:
        """Ground-truth object the assistant is trained to emit.

        Facts are copied so a caller can edit a predicted answer without
        changing the label.
        """
        return {
            "site_id": self.site_id,
            "observed_at": self.observed_at,
            "facts": copy.deepcopy(self.facts),
        }

    def is_trainable(self) -> bool:
        return self.label_status == SYNTHETIC_STATUS and self.split in TRAIN_SPLITS


def _maybe_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    return float(value)


def load_jsonl(path: Path) -> list[SiteRecord]:
    records: list[SiteRecord] = []
    text = path.read_text(encoding="utf-8")
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.strip():
            continue
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}:{line_number} is not JSON") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number} must be an object")
        records.append(SiteRecord.from_dict(raw))
    return records


def write_jsonl(path: Path, records: list[SiteRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True) for record in records]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_records(records: list[SiteRecord], split: str) -> tuple[list[SiteRecord], list[SiteRecord]]:
    """Return ``(kept, skipped)`` for one split of trainable synthetic rows."""
    kept: list[SiteRecord] = []
    skipped: list[SiteRecord] = []
    for record in records:
        if record.is_trainable() and record.split == split:
            kept.append(record)
        else:
            skipped.append(record)
    return kept, skipped
