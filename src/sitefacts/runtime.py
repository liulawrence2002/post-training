"""Process helpers shared by the training and eval entry points."""

from __future__ import annotations

import os

from sitefacts.config import RunConfig


def require_tinker_api_key() -> None:
    """Fail before any network call when the key is missing.

    The Tinker SDK reads ``TINKER_API_KEY`` itself. This function does not
    return or print the value.
    """
    if not os.environ.get("TINKER_API_KEY", "").strip():
        raise SystemExit(
            "TINKER_API_KEY is unset. Create a key at "
            "https://tinker-console.thinkingmachines.ai and export it in the "
            "environment. Do not commit the key."
        )


def resolve_base_url(config: RunConfig) -> str | None:
    value = config.base_url or os.environ.get("TINKER_BASE_URL") or ""
    value = value.strip()
    return value or None
