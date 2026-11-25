#!/usr/bin/env python3
"""Download a few unlabeled public imagery tiles. See sitefacts.fetch_public_samples."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from sitefacts.fetch_public_samples import main

if __name__ == "__main__":
    main()
