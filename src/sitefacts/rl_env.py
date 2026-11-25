"""Single-site RL environment.

Each episode shows one site, optionally runs ``lookup_site_context``, and
scores the final JSON answer with :func:`sitefacts.reward.score_response`.

Thinking effort is passed on every ``build_generation_prompt`` call. The
cookbook's ``build_agent_tool_env`` helper does not take an effort argument,
so this environment owns the tool loop and keeps effort explicit. Inkling's
default rollout settings (agentic preset, ``MinViableGroup``) still apply to
the training loop because this project does not override them.

A malformed parse ends the episode with reward 0. Eval retries that case once;
RL does not, so a truncated sample is not treated as a successful answer.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Sequence

import tinker
from tinker_cookbook.completers import StopCondition
from tinker_cookbook.renderers import ParseTermination, get_text_content
from tinker_cookbook.rl.types import (
    Action,
    ActionExtra,
    Env,
    EnvGroupBuilder,
    Metrics,
    Observation,
    StepResult,
    Trajectory,
)

from sitefacts.messages import build_prompt_messages, tool_for_record
from sitefacts.records import SiteRecord
from sitefacts.reward import score_response
from sitefacts.tools import execute_tool_call, tool_result_message


class SiteFactsEnv(Env):
    def __init__(
        self,
        record: SiteRecord,
        renderer: Any,
        effort: float,
        data_root: Any,
        max_turns: int,
    ) -> None:
        self.record = record
        self.renderer = renderer
        self.effort = effort
        self.max_turns = max_turns
        self.turn = 0
        self.messages = copy.deepcopy(
            build_prompt_messages(record, data_root, embed_images=True)
        )
        self.tool = tool_for_record(record)

    def _prompt(self) -> Observation:
        # effort is required. Omitting it still defaults to 0.9 inside
        # TmlV0Renderer, which hides the choice this run actually made.
        return self.renderer.build_generation_prompt(self.messages, effort=self.effort)

    def _stop(self) -> StopCondition:
        return self.renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Observation, StopCondition]:
        return self._prompt(), self._stop()

    async def step(self, action: Action, *, extra: ActionExtra | None = None) -> StepResult:
        self.turn += 1
        message, termination = self.renderer.parse_response(action)
        self.messages.append(message)
        if termination == ParseTermination.MALFORMED or not getattr(termination, "is_clean", False):
            return self._finish(0.0, parse_error=True, detail=f"parse termination {termination}")
        tool_calls = list(message.get("tool_calls") or [])
        unparsed = list(message.get("unparsed_tool_calls") or [])
        if unparsed and not tool_calls:
            return self._finish(0.0, parse_error=True, detail="unparsed tool call")
        if tool_calls and self.turn < self.max_turns:
            self._append_tool_results(tool_calls)
            return StepResult(
                reward=0.0,
                episode_done=False,
                next_observation=self._prompt(),
                next_stop_condition=self._stop(),
                metrics={"parse_error": 0.0, "tool_turn": 1.0},
            )
        text = get_text_content(message)
        scored = score_response(
            text,
            site_id=self.record.site_id,
            observed_at=self.record.observed_at,
            facts=self.record.facts,
        )
        metrics = scored.metrics()
        metrics["parse_error"] = 0.0
        metrics["tool_turn"] = 0.0
        return StepResult(
            reward=scored.total,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self._stop(),
            metrics=metrics,
            logs={"example_id": self.record.example_id, "errors": "; ".join(scored.errors)},
        )

    def _append_tool_results(self, tool_calls: list[Any]) -> None:
        for call in tool_calls:
            try:
                call_id, name, result = execute_tool_call(self.tool, call)
            except ValueError as exc:
                call_id = str(getattr(call, "id", "") or "")
                name = str(getattr(getattr(call, "function", None), "name", "") or "unknown")
                result = {"ok": False, "error": "bad_arguments", "message": str(exc)}
            self.messages.append(tool_result_message(call_id, name, result))

    def _finish(self, reward: float, *, parse_error: bool, detail: str) -> StepResult:
        return StepResult(
            reward=reward,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=self._stop(),
            metrics={
                "parse_error": float(parse_error),
                "schema_valid": 0.0,
                "fact_f1": 0.0,
                "provenance": 0.0,
                "correct": 0.0,
            },
            logs={"example_id": self.record.example_id, "errors": detail},
        )


@dataclass
class SiteFactsGroupBuilder(EnvGroupBuilder):
    """One site, repeated ``group_size`` times so GRPO can center rewards."""

    record: SiteRecord
    renderer: Any
    effort: float
    data_root: Any
    group_size: int
    max_turns: int

    async def make_envs(self) -> Sequence[Env]:
        return [
            SiteFactsEnv(
                self.record,
                self.renderer,
                self.effort,
                self.data_root,
                self.max_turns,
            )
            for _ in range(self.group_size)
        ]

    async def compute_group_rewards(
        self, trajectory_group: list[Trajectory], env_group: Sequence[Env]
    ) -> list[tuple[float, Metrics]]:
        return [(0.0, {}) for _ in trajectory_group]

    def logging_tags(self) -> list[str]:
        return ["sitefacts", self.record.split]
