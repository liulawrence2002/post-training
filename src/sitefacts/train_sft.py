"""Supervised warm-start on site-fact conversations.

Renders every example with an explicit thinking effort and the renderer
recommended for the selected Inkling model. Loss is on assistant messages,
including tool-call turns when the example has a tool trace.

    python -m sitefacts.train_sft --config configs/default.json
    python -m sitefacts.train_sft --model inkling-2-small
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import random
from datetime import datetime
from pathlib import Path

import chz
from tinker_cookbook import cli_utils
from tinker_cookbook.renderers import ToolCall, TrainOnWhat
from tinker_cookbook.supervised import train
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDataset, SupervisedDatasetBuilder

from sitefacts.cli import add_run_args, config_from_args
from sitefacts.config import RunConfig
from sitefacts.messages import SYSTEM_PROMPT, build_sft_messages
from sitefacts.pipeline import data_root_for, load_trainable
from sitefacts.records import SiteRecord
from sitefacts.rendering import make_renderer, sft_learning_rate
from sitefacts.runtime import require_tinker_api_key, resolve_base_url
from sitefacts.tools import TOOL_SPECS

logger = logging.getLogger(__name__)


def _coerce_tool_calls(messages: list[dict]) -> list[dict]:
    coerced = []
    for message in messages:
        copied = dict(message)
        calls = copied.get("tool_calls")
        if calls:
            copied["tool_calls"] = [
                call if not isinstance(call, dict) else ToolCall.model_validate(call) for call in calls
            ]
        coerced.append(copied)
    return coerced


def _with_renderer_prefix(renderer, messages: list[dict]) -> list[dict]:
    """Replace the hand-built prefix so the live renderer owns system + tools."""
    rest = []
    dropped = 0
    for message in messages:
        if dropped < 2 and message.get("role") in {"system", "tool_declare"}:
            dropped += 1
            continue
        rest.append(message)
    prefix = renderer.create_conversation_prefix_with_tools(TOOL_SPECS, SYSTEM_PROMPT)
    return list(prefix) + rest


class _ConversationDataset(SupervisedDataset):
    def __init__(self, records: list[SiteRecord], map_fn, batch_size: int) -> None:
        self.records = list(records)
        self.map_fn = map_fn
        self.batch_size = batch_size
        self.order = list(range(len(self.records)))
        leftover = len(self.records) % batch_size
        if leftover:
            logger.warning(
                "Dropping %s example(s) that do not fill a batch of %s.",
                leftover,
                batch_size,
            )

    def set_epoch(self, seed: int = 0) -> None:
        self.order = list(range(len(self.records)))
        random.Random(seed).shuffle(self.order)

    def get_batch(self, index: int):
        start = index * self.batch_size
        indexes = self.order[start : start + self.batch_size]
        return [self.map_fn(self.records[item]) for item in indexes]

    def __len__(self) -> int:
        return len(self.records) // self.batch_size


@chz.chz
class SiteFactsSftBuilder(SupervisedDatasetBuilder):
    data_path: str
    model_name: str
    effort: float
    batch_size: int
    max_length: int

    def __call__(self):
        renderer, renderer_name = make_renderer(self.model_name)
        logger.info(
            "SFT renderer %s for %s at effort %s",
            renderer_name,
            self.model_name,
            self.effort,
        )
        path = Path(self.data_path)
        root = data_root_for(path)

        def map_fn(record: SiteRecord):
            messages = build_sft_messages(record, root, embed_images=True)
            messages = _with_renderer_prefix(renderer, _coerce_tool_calls(messages))
            model_input, weights = renderer.build_supervised_example(
                messages,
                train_on_what=TrainOnWhat.ALL_ASSISTANT_MESSAGES,
                effort=self.effort,
            )
            return datum_from_model_input_weights(model_input, weights, self.max_length)

        train_records = load_trainable(path, "train")
        test_records = load_trainable(path, "test")
        train_ds = _ConversationDataset(train_records, map_fn, self.batch_size)
        test_ds = _ConversationDataset(test_records, map_fn, min(self.batch_size, len(test_records)))
        if len(train_ds) < 1 or len(test_ds) < 1:
            raise ValueError(
                "Need at least one full batch in train and test. "
                f"Got {len(train_records)} train and {len(test_records)} test rows "
                f"with batch_size {self.batch_size}."
            )
        return train_ds, test_ds


def build_train_config(config: RunConfig) -> train.Config:
    learning_rate, source = sft_learning_rate(config.model_name, config.sft_learning_rate)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = config.log_dir("sft", stamp)
    session = f"sitefacts-sft-{config.model_name.replace('/', '-').replace(':', '-')}"
    print(
        "\n".join(
            [
                "Sitefacts SFT",
                f"  model: {config.model_name}",
                f"  effort: {config.effort}",
                f"  learning_rate: {learning_rate:.6g} ({source})",
                f"  data: {config.data_file()}",
                f"  load_checkpoint: {config.load_checkpoint_path or '(base Inkling weights)'}",
                f"  log_path: {log_path}",
                "  TINKER_API_KEY: set",
            ]
        )
    )
    cli_utils.check_log_dir(str(log_path), behavior_if_exists="raise")
    return train.Config(
        log_path=str(log_path),
        model_name=config.model_name,
        recipe_name="sitefacts_sft",
        dataset_builder=SiteFactsSftBuilder(
            data_path=str(config.data_file()),
            model_name=config.model_name,
            effort=config.effort,
            batch_size=config.batch_size,
            max_length=config.max_length,
        ),
        learning_rate=learning_rate,
        lr_schedule="linear",
        num_epochs=config.num_epochs,
        lora_rank=config.lora_rank,
        save_every=config.save_every,
        eval_every=config.eval_every,
        max_steps=config.max_steps,
        load_checkpoint_path=config.load_checkpoint_path,
        base_url=resolve_base_url(config),
        session_name=session,
        wandb_project=os.environ.get("WANDB_PROJECT") or None,
        wandb_name=session,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="SFT an Inkling model on site-fact conversations.")
    add_run_args(parser)
    config = config_from_args(parser.parse_args())
    require_tinker_api_key()
    asyncio.run(train.main(build_train_config(config)))


if __name__ == "__main__":
    main()
