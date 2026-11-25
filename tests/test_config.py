from pathlib import Path

import pytest

from sitefacts.config import (
    DEFAULT_EFFORT,
    DEFAULT_MODEL,
    assert_inkling,
    load_run_config,
    repo_root,
    resolve_effort,
    resolve_model_name,
)


def test_default_model_is_the_cheap_mini():
    assert DEFAULT_MODEL == "thinkingmachines/Inkling-2-Mini"
    config = load_run_config(repo_root() / "configs/default.json")
    assert config.model_name == DEFAULT_MODEL
    assert config.effort == DEFAULT_EFFORT == 0.9
    assert config.temperature == 1.0
    assert config.rl_lr_scale == 0.1
    assert config.remove_constant_reward_groups is True


def test_aliases_switch_models_without_a_renderer_name():
    assert resolve_model_name("inkling-2-small") == "thinkingmachines/Inkling-2-Small"
    assert resolve_model_name("inkling-2-mini-free") == "thinkingmachines/Inkling-2-Mini:peft:65536:free"
    assert resolve_model_name("thinkingmachines/Inkling") == "thinkingmachines/Inkling"
    small = load_run_config(repo_root() / "configs/inkling-2-small.json")
    assert small.model_name == "thinkingmachines/Inkling-2-Small"
    assert small.effort == 0.9
    free = load_run_config(repo_root() / "configs/inkling-2-mini-free.json")
    assert free.model_name.endswith(":free")
    assert assert_inkling(free.model_name) == "thinkingmachines/Inkling-2-Mini"


def test_effort_presets_and_bounds():
    assert resolve_effort("high") == 0.9
    assert resolve_effort("xhigh") == 0.99
    assert resolve_effort(0) == 0.0
    with pytest.raises(ValueError):
        resolve_effort(1)
    with pytest.raises(ValueError):
        resolve_effort(-0.1)
    with pytest.raises(ValueError):
        resolve_effort("medium-high")


def test_rejects_non_inkling_and_unknown_keys(tmp_path: Path):
    with pytest.raises(ValueError):
        load_run_config(None, {"model_name": "Qwen/Qwen3-8B"})
    bad = tmp_path / "bad.json"
    bad.write_text('{"renderer_name": "tml_v0"}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="renderer_name"):
        load_run_config(bad)


def test_cli_override_replaces_file_model():
    config = load_run_config(
        repo_root() / "configs/default.json",
        {"model_name": "inkling", "effort": "low"},
    )
    assert config.model_name == "thinkingmachines/Inkling"
    assert config.effort == 0.2
