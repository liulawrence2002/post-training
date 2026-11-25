"""RL on the site-facts environment, ideally warm-started from SFT.

The learning rate defaults to ``0.1 * hyperparam_utils.get_lr(model)``. That is
the cookbook's conservative starting point. Sweep it together with effort on
the first steps, and watch ``optim/entropy`` plus groups whose rewards are all
equal. ``remove_constant_reward_groups`` is on so those groups contribute no
gradient.

    python -m sitefacts.train_rl --config configs/default.json \\
        --load-checkpoint tinker://...
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
from tinker_cookbook.rl import train
from tinker_cookbook.rl.types import RLDataset, RLDatasetBuilder

from sitefacts.cli import add_run_args, config_from_args
from sitefacts.config import RunConfig
from sitefacts.pipeline import data_root_for, load_trainable
from sitefacts.records import SiteRecord
from sitefacts.rendering import make_renderer, rl_learning_rate
from sitefacts.rl_env import SiteFactsGroupBuilder
from sitefacts.runtime import require_tinker_api_key, resolve_base_url

logger = logging.getLogger(__name__)


class SiteFactsRlDataset(RLDataset):
    def __init__(
        self,
        records: list[SiteRecord],
        renderer,
        *,
        effort: float,
        data_root: Path,
        batch_size: int,
        group_size: int,
        max_turns: int,
    ) -> None:
        self.records = list(records)
        self.renderer = renderer
        self.effort = effort
        self.data_root = data_root
        self.batch_size = batch_size
        self.group_size = group_size
        self.max_turns = max_turns
        self.order = list(range(len(self.records)))

    def set_epoch(self, seed: int = 0) -> None:
        self.order = list(range(len(self.records)))
        random.Random(seed).shuffle(self.order)

    def get_batch(self, index: int):
        start = index * self.batch_size
        indexes = self.order[start : start + self.batch_size]
        return [
            SiteFactsGroupBuilder(
                record=self.records[item],
                renderer=self.renderer,
                effort=self.effort,
                data_root=self.data_root,
                group_size=self.group_size,
                max_turns=self.max_turns,
            )
            for item in indexes
        ]

    def __len__(self) -> int:
        return len(self.records) // self.batch_size


@chz.chz
class SiteFactsRlBuilder(RLDatasetBuilder):
    data_path: str
    model_name: str
    effort: float
    batch_size: int
    group_size: int
    max_turns: int

    async def __call__(self):
        renderer, renderer_name = make_renderer(self.model_name)
        logger.info(
            "RL renderer %s for %s at effort %s",
            renderer_name,
            self.model_name,
            self.effort,
        )
        path = Path(self.data_path)
        root = data_root_for(path)
        train_records = load_trainable(path, "train")
        test_records = load_trainable(path, "test")
        train_ds = SiteFactsRlDataset(
            train_records,
            renderer,
            effort=self.effort,
            data_root=root,
            batch_size=self.batch_size,
            group_size=self.group_size,
            max_turns=self.max_turns,
        )
        if len(train_ds) < 1:
            raise ValueError(
                f"Need a full RL batch. Have {len(train_records)} train rows "
                f"and batch_size {self.batch_size}."
            )
        test_batch = min(self.batch_size, len(test_records))
        test_ds = SiteFactsRlDataset(
            test_records,
            renderer,
            effort=self.effort,
            data_root=root,
            batch_size=test_batch,
            group_size=self.group_size,
            max_turns=self.max_turns,
        )
        if len(test_ds) < 1:
            logger.warning("Test split does not fill a batch; RL will run without an eval dataset.")
            return train_ds, None
        return train_ds, test_ds


def build_train_config(config: RunConfig) -> train.Config:
    learning_rate, source = rl_learning_rate(
        config.model_name, config.rl_learning_rate, config.rl_lr_scale
    )
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_path = config.log_dir("rl", stamp)
    session = f"sitefacts-rl-{config.model_name.replace('/', '-').replace(':', '-')}"
    print(
        "\n".join(
            [
                "Sitefacts RL",
                f"  model: {config.model_name}",
                f"  effort: {config.effort}",
                f"  temperature: {config.temperature}",
                f"  learning_rate: {learning_rate:.6g} ({source})",
                f"  group_size: {config.group_size}",
                f"  max_tokens: {config.max_tokens}",
                f"  load_checkpoint: {config.load_checkpoint_path or '(base Inkling weights)'}",
                f"  log_path: {log_path}",
                "  TINKER_API_KEY: set",
            ]
        )
    )
    if config.load_checkpoint_path is None:
        print(
            "  note: RL is starting from the post-trained Inkling weights, "
            "without a sitefacts SFT warm-start. A short SFT run is the "
            "recommended first stage when the JSON shape is unfamiliar."
        )
    if config.group_size < 2:
        print("  note: group_size < 2 gives GRPO no within-group variance.")
    if config.temperature != 1.0:
        print("  note: Inkling recipes sample at temperature 1.0. Effort is the reasoning knob.")
    cli_utils.check_log_dir(str(log_path), behavior_if_exists="raise")
    return train.Config(
        log_path=str(log_path),
        model_name=config.model_name,
        recipe_name="sitefacts_rl",
        dataset_builder=SiteFactsRlBuilder(
            data_path=str(config.data_file()),
            model_name=config.model_name,
            effort=config.effort,
            batch_size=config.batch_size,
            group_size=config.group_size,
            max_turns=config.max_turns,
        ),
        learning_rate=learning_rate,
        lora_rank=config.lora_rank,
        max_tokens=config.max_tokens,
        temperature=config.temperature,
        save_every=config.save_every,
        eval_every=config.eval_every,
        num_epochs=config.num_epochs,
        max_steps=config.max_steps,
        remove_constant_reward_groups=config.remove_constant_reward_groups,
        load_checkpoint_path=config.load_checkpoint_path,
        base_url=resolve_base_url(config),
        session_name=session,
        wandb_project=os.environ.get("WANDB_PROJECT") or None,
        wandb_name=session,
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description="RL-train an Inkling model on site-fact rewards.")
    add_run_args(parser)
    config = config_from_args(parser.parse_args())
    require_tinker_api_key()
    asyncio.run(train.main(build_train_config(config)))


if __name__ == "__main__":
    main()
