"""Construct the Inkling renderer the cookbook recommends for a model name.

Never hardcode ``tml_v0``, and never call ``tokenizer.encode`` on a prompt.
The tokenizer has no chat template; raw encoding is out of distribution.
The renderer inserts the thinking-effort message when ``effort`` is passed
to ``build_generation_prompt`` / ``build_supervised_example``.
"""

from __future__ import annotations

from typing import Any

from sitefacts.config import assert_inkling


def make_renderer(model_name: str) -> tuple[Any, str]:
    """Return ``(renderer, renderer_name)`` for an Inkling model.

    ``renderer_name`` comes from ``model_info.get_recommended_renderer_name``.
    Imports of tinker-cookbook happen here so unit tests can import the rest
    of the package without the SDK.
    """
    assert_inkling(model_name)
    from tinker_cookbook import model_info
    from tinker_cookbook.renderers import get_renderer
    from tinker_cookbook.tokenizer_utils import get_tokenizer

    attributes = model_info.get_model_attributes(model_name)
    if not attributes.is_vl:
        raise ValueError(f"{model_name} does not accept images. Sitefacts episodes attach a tile.")
    renderer_name = model_info.get_recommended_renderer_name(model_name)
    tokenizer = get_tokenizer(model_name)
    return get_renderer(renderer_name, tokenizer), renderer_name


def sft_learning_rate(model_name: str, override: float | None) -> tuple[float, str]:
    """Return ``(learning_rate, source)``.

    The source string is logged. It is not a secret. ``get_lr`` is an AdamW
    LoRA starting point for this model, not a tuned optimum.
    """
    if override is not None:
        return float(override), "config.sft_learning_rate"
    from tinker_cookbook import hyperparam_utils

    return float(hyperparam_utils.get_lr(model_name)), "hyperparam_utils.get_lr"


def rl_learning_rate(model_name: str, override: float | None, scale: float) -> tuple[float, str]:
    """Conservative RL start: ``scale * get_lr``, default scale 0.1.

    Sweep nearby values, including higher ones, on the first RL steps.
    """
    if override is not None:
        return float(override), "config.rl_learning_rate"
    from tinker_cookbook import hyperparam_utils

    base = float(hyperparam_utils.get_lr(model_name))
    return scale * base, f"{scale} * hyperparam_utils.get_lr"
