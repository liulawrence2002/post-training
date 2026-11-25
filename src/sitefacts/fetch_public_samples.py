"""Download a few free public imagery tiles and store them as unlabeled samples.

NASA GIBS MODIS corrected reflectance is the source. The tiles are real
public imagery and they have no site-fact labels. Training skips
``label_status=unlabeled_public``.

TODO(data): pick a product, a license review, a site geometry, and a label
procedure before these files are mixed into a training set. Confirm current
NASA usage terms before commercial use.
"""

from __future__ import annotations

import argparse
import json
import urllib.error
import urllib.request
from pathlib import Path

from sitefacts.config import repo_root
from sitefacts.records import UNLABELED_PUBLIC_STATUS

# MODIS true-color WMTS tiles. The matrix set is GoogleMapsCompatible_Level9;
# the following number is the zoom. Zoom 2 keeps each JPEG around 20 KB.
_TILES = (
    {
        "example_id": "public-modis-2014-04-09",
        "date": "2014-04-09",
        "url": (
            "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            "MODIS_Terra_CorrectedReflectance_TrueColor/default/2014-04-09/"
            "GoogleMapsCompatible_Level9/2/1/1.jpg"
        ),
    },
    {
        "example_id": "public-modis-2024-08-01",
        "date": "2024-08-01",
        "url": (
            "https://gibs.earthdata.nasa.gov/wmts/epsg3857/best/"
            "MODIS_Terra_CorrectedReflectance_TrueColor/default/2024-08-01/"
            "GoogleMapsCompatible_Level9/2/2/1.jpg"
        ),
    },
)


def unlabeled_record(example_id: str, date: str, url: str, image_name: str) -> dict:
    return {
        "example_id": example_id,
        "split": "unlabeled",
        "label_status": UNLABELED_PUBLIC_STATUS,
        "site_id": example_id,
        "observed_at": f"{date}T00:00:00Z",
        "asset_type": "unassigned public tile",
        "latitude": None,
        "longitude": None,
        "coordinate_status": "tile_index_not_a_parcel",
        "question": "Unlabeled public tile. Do not train on this row.",
        "analyst_notes": [],
        "image": image_name,
        "swatch_color": None,
        "context": {},
        "tool_trace": [],
        "facts": [],
        "source_url": url,
        "notes": (
            "UNLABELED PUBLIC SAMPLE from NASA GIBS MODIS corrected reflectance. "
            "No ground-truth facts. The training loader skips this label_status. "
            "TODO(labels): do not promote this row into train or test."
        ),
    }


def fetch_public_samples(destination: Path, limit: int = 2, timeout: float = 30.0) -> Path:
    """Download up to ``limit`` tiles into ``destination`` and write a JSONL index."""
    if limit < 1:
        raise ValueError("limit must be >= 1")
    destination.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in _TILES[:limit]:
        image_name = f"images/{spec['example_id']}.jpg"
        target = destination / image_name
        target.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(spec["url"], headers={"User-Agent": "sitefacts-sample-fetch/0.1"})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
        except urllib.error.URLError as exc:
            raise SystemExit(
                f"Could not download {spec['url']}: {exc}. "
                "The synthetic dataset under data/sample does not need this download."
            ) from exc
        if not payload.startswith(b"\xff\xd8"):
            raise SystemExit(f"Download from {spec['url']} was not a JPEG.")
        target.write_bytes(payload)
        rows.append(unlabeled_record(spec["example_id"], spec["date"], spec["url"], image_name))
    index = destination / "unlabeled.jsonl"
    index.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )
    return index


def main() -> None:
    parser = argparse.ArgumentParser(description="Download a few unlabeled public imagery tiles.")
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output directory. Default: data/public_samples (gitignored).",
    )
    parser.add_argument("--limit", type=int, default=2)
    args = parser.parse_args()
    destination = args.out or (repo_root() / "data" / "public_samples")
    index = fetch_public_samples(destination, limit=args.limit)
    print(f"Wrote unlabeled public samples to {index}")
    print("These tiles have no fact labels. Training skips label_status=unlabeled_public.")


if __name__ == "__main__":
    main()
