"""Evaluate site-fact answers.

Offline mode scores a predictions file, or the canonical targets, and does
not need ``TINKER_API_KEY``. Online mode samples the test split at the
configured effort and temperature (default 1.0) and records the effort next
to the scores.

A malformed parse is retried once with a corrective user message that includes
the termination. RL does not retry, so treat online eval as slightly more
forgiving on truncated samples.

    python -m sitefacts.evaluate --offline
    python -m sitefacts.evaluate --offline --predictions preds.jsonl
    python -m sitefacts.evaluate --checkpoint tinker://...
"""

from __future__ import annotations

import argparse
import asyncio
import json
from pathlib import Path
from typing import Any

from sitefacts.cli import add_run_args, config_from_args
from sitefacts.config import RunConfig
from sitefacts.messages import build_prompt_messages, tool_for_record
from sitefacts.offline_eval import load_predictions, score_records, summarize
from sitefacts.pipeline import data_root_for, load_trainable
from sitefacts.records import SiteRecord
from sitefacts.reward import score_response
from sitefacts.runtime import require_tinker_api_key, resolve_base_url
from sitefacts.tools import execute_tool_call, tool_result_message


def run_offline(config: RunConfig, predictions: Path | None) -> dict[str, Any]:
    records = load_trainable(config.data_file(), "train") + load_trainable(config.data_file(), "test")
    if predictions is None:
        report = score_records(records)
        report["note"] = (
            "Scored the canonical targets. This checks the scorer and the sample "
            "labels. It is not a model evaluation."
        )
    else:
        report = score_records(records, load_predictions(predictions))
        report["note"] = "Scored saved responses. No model was sampled."
    report["effort"] = None
    report["model_name"] = config.model_name
    return report


async def run_online(config: RunConfig, checkpoint: str | None) -> dict[str, Any]:
    import tinker
    from tinker_cookbook.renderers import ParseTermination, get_text_content

    from sitefacts.rendering import make_renderer

    require_tinker_api_key()
    renderer, renderer_name = make_renderer(config.model_name)
    service = tinker.ServiceClient(base_url=resolve_base_url(config))
    if checkpoint:
        created = service.create_sampling_client(model_path=checkpoint)
    else:
        created = service.create_sampling_client(base_model=config.model_name)
    sampling_client = await created if hasattr(created, "__await__") else created.result()
    records = load_trainable(config.data_file(), "test")
    root = data_root_for(config.data_file())
    rows = []
    for record in records:
        text, termination, turns = await _rollout(
            sampling_client,
            renderer,
            record,
            root,
            config,
            ParseTermination,
            get_text_content,
        )
        scored = score_response(
            text,
            site_id=record.site_id,
            observed_at=record.observed_at,
            facts=record.facts,
        )
        rows.append(
            {
                "example_id": record.example_id,
                "split": record.split,
                "source": "sample",
                "reward": scored.total,
                "fact_f1": scored.fact_f1,
                "provenance": scored.provenance,
                "schema_valid": scored.schema_valid,
                "site_id_match": scored.site_id_match,
                "termination": termination,
                "turns": turns,
                "response": text,
                "errors": scored.errors,
            }
        )
    report = summarize(rows, source="sample")
    report.update(
        {
            "model_name": config.model_name,
            "renderer": renderer_name,
            "effort": config.effort,
            "temperature": config.temperature,
            "max_tokens": config.max_tokens,
            "checkpoint": checkpoint,
            "note": "Effort is part of the result. A score without it is not reproducible.",
        }
    )
    return report


async def _rollout(sampling_client, renderer, record: SiteRecord, root, config, parse_termination, get_text_content):
    messages = build_prompt_messages(record, root, embed_images=True)
    tool = tool_for_record(record)
    retried_malformed = False
    termination_value = "not_sampled"
    text = ""
    turns = 0
    while turns < config.max_turns:
        turns += 1
        message, termination = await _sample(sampling_client, renderer, messages, config)
        messages.append(message)
        termination_value = getattr(termination, "value", str(termination))
        text = get_text_content(message)
        malformed = termination == parse_termination.MALFORMED or not getattr(termination, "is_clean", False)
        if malformed and not retried_malformed:
            retried_malformed = True
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "The previous reply did not parse. "
                        f"Termination: {termination_value}. "
                        "Return one JSON object for this site and no other text."
                    ),
                }
            )
            continue
        tool_calls = list(message.get("tool_calls") or [])
        if tool_calls and turns < config.max_turns and not malformed:
            for call in tool_calls:
                try:
                    call_id, name, result = execute_tool_call(tool, call)
                except ValueError as exc:
                    call_id, name = "", "unknown"
                    result = {"ok": False, "error": "bad_arguments", "message": str(exc)}
                messages.append(tool_result_message(call_id, name, result))
            continue
        break
    return text, termination_value, turns


async def _sample(sampling_client, renderer, messages, config: RunConfig):
    import tinker

    prompt = renderer.build_generation_prompt(messages, effort=config.effort)
    response = await sampling_client.sample(
        prompt=prompt,
        num_samples=1,
        sampling_params=tinker.SamplingParams(
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            stop=renderer.get_stop_sequences(),
        ),
    )
    return renderer.parse_response(response.sequences[0].tokens)


def _print_report(report: dict[str, Any]) -> None:
    print(
        f"n={report['n']} mean_reward={report['mean_reward']:.3f} "
        f"fact_f1={report['mean_fact_f1']:.3f} "
        f"schema_valid={report['schema_valid_rate']:.3f} "
        f"effort={report.get('effort')}"
    )
    print(report.get("note", ""))
    for row in report["examples"]:
        errors = "; ".join(row.get("errors") or [])
        print(
            f"  {row['example_id']}: reward={row['reward']:.3f} "
            f"f1={row['fact_f1']:.3f} schema={row['schema_valid']} {errors}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Score site-fact answers online or offline.")
    add_run_args(parser)
    parser.add_argument("--offline", action="store_true", help="Do not call the API.")
    parser.add_argument("--predictions", type=Path, help="JSONL with example_id and response.")
    parser.add_argument("--checkpoint", help="Tinker sampling checkpoint. Defaults to the base model.")
    parser.add_argument("--output", type=Path, help="Where to write the JSON report.")
    args = parser.parse_args()
    config = config_from_args(args)
    if args.offline:
        report = run_offline(config, args.predictions)
    else:
        checkpoint = args.checkpoint or config.load_checkpoint_path
        report = asyncio.run(run_online(config, checkpoint))
    _print_report(report)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
