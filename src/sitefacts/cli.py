"""Shared CLI flags for the SFT, RL, and eval entry points."""

from __future__ import annotations

import argparse
from typing import Any

from sitefacts.config import RunConfig, load_run_config


def add_run_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--config",
        default="configs/default.json",
        help="JSON config. Partial files override built-in defaults.",
    )
    parser.add_argument("--model", help="Tinker model id or short alias, such as inkling-2-small.")
    parser.add_argument("--effort", help="Thinking effort in [0, 1), or a preset name (high, xhigh, ...).")
    parser.add_argument("--data", help="JSONL dataset path.")
    parser.add_argument("--max-steps", type=int, help="Stop after this many optimizer steps.")
    parser.add_argument("--base-url", help="Optional Tinker service URL.")
    parser.add_argument(
        "--load-checkpoint",
        help="Tinker checkpoint to warm-start from. Recommended for RL after SFT.",
    )


def config_from_args(args: argparse.Namespace) -> RunConfig:
    overrides: dict[str, Any] = {}
    if args.model:
        overrides["model_name"] = args.model
    if args.effort is not None:
        overrides["effort"] = args.effort
    if args.data:
        overrides["data_path"] = args.data
    if args.max_steps is not None:
        overrides["max_steps"] = args.max_steps
    if args.base_url:
        overrides["base_url"] = args.base_url
    if args.load_checkpoint:
        overrides["load_checkpoint_path"] = args.load_checkpoint
    return load_run_config(args.config, overrides)
