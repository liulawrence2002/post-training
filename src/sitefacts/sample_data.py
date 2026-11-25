"""Write the synthetic JSONL dataset and its color-swatch PNGs."""

from __future__ import annotations

import argparse
from pathlib import Path

from sitefacts.config import repo_root
from sitefacts.pngutil import write_solid_png
from sitefacts.records import SiteRecord, write_jsonl
from sitefacts.sample_catalog import SWATCH_RGB, sample_rows


def write_sample_dataset(root: Path | None = None) -> Path:
    """Regenerate ``data/sample`` under ``root`` (the repo root by default)."""
    destination = (root or repo_root()) / "data" / "sample"
    records = [SiteRecord.from_dict(row) for row in sample_rows()]
    for record in records:
        if record.swatch_color is None or record.image is None:
            raise ValueError(f"{record.example_id} is missing a swatch")
        write_solid_png(destination / record.image, SWATCH_RGB[record.swatch_color], size=32)
    jsonl = destination / "sites.jsonl"
    write_jsonl(jsonl, records)
    return jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Rewrite the synthetic sample dataset.")
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Directory that should contain data/sample. Defaults to the repo root.",
    )
    args = parser.parse_args()
    path = write_sample_dataset(args.root)
    print(f"Wrote synthetic sample data to {path}")
    print("These rows are format examples, not real inspections.")


if __name__ == "__main__":
    main()
