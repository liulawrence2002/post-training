"""Run configuration, model ids, and thinking-effort presets.

Inkling models share the ``tml_v0`` renderer. The cookbook selects that
renderer from the model name, so this project stores a model id and never a
hardcoded renderer name. Learning rates are also per model; training scripts
ask ``hyperparam_utils.get_lr`` rather than copying a number between models.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

# Short names are convenience aliases. The value is the Tinker model id.
# Context lengths and prices: https://tinker-docs.thinkingmachines.ai/tinker/models/
MODEL_ALIASES: dict[str, str] = {
    "inkling-2-mini": "thinkingmachines/Inkling-2-Mini",
    "inkling-2-mini-free": "thinkingmachines/Inkling-2-Mini:peft:65536:free",
    "inkling-2-mini-256k": "thinkingmachines/Inkling-2-Mini:peft:262144",
    "inkling-2-small": "thinkingmachines/Inkling-2-Small",
    "inkling-2-small-256k": "thinkingmachines/Inkling-2-Small:peft:262144",
    "inkling": "thinkingmachines/Inkling",
    "inkling-256k": "thinkingmachines/Inkling:peft:262144",
    "inkling-small": "thinkingmachines/Inkling-Small",
    "inkling-small-256k": "thinkingmachines/Inkling-Small:peft:262144",
}

# Cheap paid tier for iteration. The free id is rate limited; it is a config
# switch, not the default, so a run does not silently hit the free-tier cap.
DEFAULT_MODEL = MODEL_ALIASES["inkling-2-mini"]

# Same scalars the OpenAI-compatible API uses for reasoning_effort presets.
# Any finite value in [0.0, 1.0) is valid. 1.0 is not.
EFFORT_PRESETS: dict[str, float] = {
    "none": 0.0,
    "minimal": 0.1,
    "low": 0.2,
    "medium": 0.7,
    "high": 0.9,
    "xhigh": 0.99,
}

# One effort for the whole dataset, and the same value at sample time.
# 0.9 is the renderer default and the high preset. It is a deliberate choice.
DEFAULT_EFFORT = EFFORT_PRESETS["high"]

# Cookbook RL guide: start near 0.1 * get_lr(model), then sweep higher too.
DEFAULT_RL_LR_SCALE = 0.1

# Inkling sampling scripts use temperature 1.0. Lowering it is not a stand-in
# for lowering effort.
DEFAULT_TEMPERATURE = 1.0


def repo_root() -> Path:
    """Directory that contains this checkout's pyproject.toml."""
    here = Path(__file__).resolve()
    for candidate in [Path.cwd(), *here.parents]:
        if (candidate / "pyproject.toml").is_file() and (candidate / "src" / "sitefacts").is_dir():
            return candidate
    return Path.cwd()


def resolve_repo_path(path: str | Path) -> Path:
    raw = Path(path)
    if raw.is_absolute():
        return raw
    return repo_root() / raw


def resolve_model_name(name: str) -> str:
    """Map a short alias or pass through a ``org/model`` Tinker id."""
    key = name.strip()
    if key in MODEL_ALIASES:
        return MODEL_ALIASES[key]
    if "/" not in key:
        known = ", ".join(sorted(MODEL_ALIASES))
        raise ValueError(f"Unknown model {name!r}. Use a Tinker id or one of: {known}")
    return key


def assert_inkling(model_name: str) -> str:
    """Return the base id and reject anything that is not an Inkling model."""
    base = model_name.split(":", 1)[0]
    if not base.startswith("thinkingmachines/Inkling"):
        raise ValueError(
            f"This project post-trains Inkling models via tml-renderers. Got {model_name!r}."
        )
    return base


def resolve_effort(value: float | str) -> float:
    """Accept a preset name or a finite float in ``[0.0, 1.0)``."""
    if isinstance(value, str):
        key = value.strip().lower()
        if key in EFFORT_PRESETS:
            return EFFORT_PRESETS[key]
        try:
            value = float(key)
        except ValueError as exc:
            known = ", ".join(EFFORT_PRESETS)
            raise ValueError(f"Unknown effort {value!r}. Use a float in [0, 1) or one of: {known}") from exc
    effort = float(value)
    if effort != effort or effort in (float("inf"), float("-inf")) or not 0.0 <= effort < 1.0:
        raise ValueError(f"Thinking effort must be finite and in [0, 1), got {effort!r}")
    return effort


@dataclass
class RunConfig:
    """Knobs shared by SFT, RL, and eval.

    A JSON config file is a partial override of these defaults. ``null``
    learning rates mean "ask hyperparam_utils.get_lr at launch".
    """

    model_name: str = DEFAULT_MODEL
    effort: float = DEFAULT_EFFORT
    lora_rank: int = 32
    batch_size: int = 2
    group_size: int = 4
    max_length: int = 16384
    # High effort produces longer traces. Raise this toward 16384+ if eval
    # reports malformed or truncated responses. Effort and max_tokens are
    # independent knobs.
    max_tokens: int = 8192
    num_epochs: int = 1
    max_steps: int | None = None
    sft_learning_rate: float | None = None
    rl_learning_rate: float | None = None
    rl_lr_scale: float = DEFAULT_RL_LR_SCALE
    temperature: float = DEFAULT_TEMPERATURE
    max_turns: int = 4
    remove_constant_reward_groups: bool = True
    eval_every: int = 0
    save_every: int = 1
    data_path: str = "data/sample/sites.jsonl"
    log_root: str = "runs"
    base_url: str | None = None
    load_checkpoint_path: str | None = None

    def resolved(self) -> RunConfig:
        self.model_name = resolve_model_name(self.model_name)
        assert_inkling(self.model_name)
        self.effort = resolve_effort(self.effort)
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.group_size < 1:
            raise ValueError("group_size must be >= 1")
        if self.max_turns < 1:
            raise ValueError("max_turns must be >= 1")
        if self.rl_lr_scale <= 0:
            raise ValueError("rl_lr_scale must be positive")
        if self.lora_rank < 1:
            raise ValueError("lora_rank must be >= 1")
        if self.max_length < 1 or self.max_tokens < 1:
            raise ValueError("max_length and max_tokens must be >= 1")
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        return self

    def data_file(self) -> Path:
        return resolve_repo_path(self.data_path)

    def log_dir(self, stage: str, stamp: str) -> Path:
        slug = self.model_name.replace("/", "-").replace(":", "-")
        return resolve_repo_path(self.log_root) / stage / slug / stamp


def load_run_config(path: str | Path | None, overrides: dict | None = None) -> RunConfig:
    """Load defaults, then a JSON override file, then CLI overrides."""
    data = asdict(RunConfig())
    if path is not None:
        file_path = Path(path)
        if not file_path.is_file():
            file_path = resolve_repo_path(path)
        loaded = json.loads(file_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"Config {file_path} must be a JSON object")
        unknown = sorted(set(loaded) - set(data))
        if unknown:
            raise ValueError(f"Unknown config keys in {file_path}: {unknown}")
        data.update(loaded)
    if overrides:
        unknown = sorted(set(overrides) - set(data))
        if unknown:
            raise ValueError(f"Unknown config overrides: {unknown}")
        for key, value in overrides.items():
            if value is not None:
                data[key] = value
    return RunConfig(**data).resolved()
