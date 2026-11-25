"""Build chat messages for site-fact episodes.

The renderer, not this module, inserts the thinking-effort system message.
Do not add a second one. Prompts are rendered later with an explicit ``effort``
argument so train tokens and sample tokens match.

Image parts use a local path or a ``data:`` URI. Remote URLs are not used;
tml-renderers will not fetch them.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Any

from sitefacts.records import SiteRecord
from sitefacts.tools import (
    TOOL_SPECS,
    SiteContextTool,
    execute_tool_call,
    openai_tool_declarations,
    tool_result_message,
)

# Confidence numbers are a sample-set convention so the SFT target is stated
# in the prompt. They are not a calibrated probability.
SYSTEM_PROMPT = """You answer questions about one property or infrastructure site for risk review.
Use the user message, the attached image, and the lookup_site_context tool.
Every fact must cite a source the episode actually gave you.

Return one JSON object and no surrounding prose:
{
  "site_id": string,
  "observed_at": "YYYY-MM-DDTHH:MM:SSZ",
  "facts": [
    {
      "claim": string,
      "value": string | number | boolean,
      "unit": string | null,
      "confidence": number,
      "source": {"kind": string, "uri": string, "observed_at": "YYYY-MM-DDTHH:MM:SSZ"}
    }
  ]
}

Allowed source.kind values: imagery, elevation, weather, ais, adsb, permits, map, analyst_note, site_context.
On this sample set, use confidence 0.9 for placeholder_swatch_color, 0.8 for a value returned by lookup_site_context, and 0.6 for a value taken from analyst notes.
If the image is a synthetic color swatch, placeholder_swatch_color is the only visual claim you may make.
Do not add your own effort-control system message.
"""


def canonical_answer(record: SiteRecord) -> str:
    return json.dumps(record.answer_payload(), indent=2, ensure_ascii=False) + "\n"


def tool_for_record(record: SiteRecord) -> SiteContextTool:
    return SiteContextTool(
        record.site_id,
        record.context,
        source_uri=f"sample://context/{record.site_id}",
        observed_at=record.observed_at,
    )


def build_user_text(record: SiteRecord) -> str:
    notes = "\n".join(f"- {note}" for note in record.analyst_notes) or "- (none)"
    image_line = (
        "A 32 by 32 synthetic color swatch is attached. "
        "You may report placeholder_swatch_color from that swatch. "
        "Do not invent other visual claims from it."
        if record.image
        else "No image is attached."
    )
    return (
        f"Site: {record.site_id}\n"
        f"Asset type: {record.asset_type}\n"
        f"Illustrative coordinates (not a survey): {record.latitude}, {record.longitude}\n"
        f"Coordinate status: {record.coordinate_status}\n"
        f"As of: {record.observed_at}\n"
        f"Question: {record.question}\n\n"
        f"Analyst notes (sample data, not a real inspection):\n{notes}\n\n"
        f"{image_line}\n"
    )


def image_part(record: SiteRecord, data_root: Path, *, embed: bool) -> dict[str, Any] | None:
    if not record.image:
        return None
    path = data_root / record.image
    if embed:
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {"type": "image", "image": f"data:image/png;base64,{encoded}"}
    return {"type": "image", "image": str(path)}


def build_user_message(record: SiteRecord, data_root: Path, *, embed_images: bool) -> dict[str, Any]:
    parts: list[dict[str, Any]] = []
    image = image_part(record, data_root, embed=embed_images)
    if image is not None:
        parts.append(image)
    parts.append({"type": "text", "text": build_user_text(record)})
    return {"role": "user", "content": parts}


def conversation_prefix() -> list[dict[str, Any]]:
    """System prompt plus the tool declaration.

    Training replaces this pair with ``renderer.create_conversation_prefix_with_tools``
    so the served renderer owns the exact prefix. The strings here match that
    helper for the built-in tool spec.
    """
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "tool_declare", "content": openai_tool_declarations(TOOL_SPECS)},
    ]


def build_prompt_messages(
    record: SiteRecord,
    data_root: Path,
    *,
    embed_images: bool = False,
) -> list[dict[str, Any]]:
    """Messages up to the user turn. Sampling and RL start from this list."""
    return [*conversation_prefix(), build_user_message(record, data_root, embed_images=embed_images)]


def build_sft_messages(
    record: SiteRecord,
    data_root: Path,
    *,
    embed_images: bool = False,
) -> list[dict[str, Any]]:
    """Full supervised conversation, including the gold answer.

    Tool traces call the same stub RL uses, so the tool message is the real
    stub result rather than a hand-written paraphrase.
    """
    messages = build_prompt_messages(record, data_root, embed_images=embed_images)
    tool = tool_for_record(record)
    for index, step in enumerate(record.tool_trace, start=1):
        arguments = step["arguments"]
        call = {
            "id": step.get("id") or f"call_{index}",
            "type": "function",
            "function": {
                "name": step["name"],
                "arguments": json.dumps(arguments, sort_keys=True, separators=(",", ":")),
            },
        }
        messages.append(
            {
                "role": "assistant",
                "content": "I need structured context before answering.",
                "tool_calls": [call],
            }
        )
        call_id, name, result = execute_tool_call(tool, call)
        messages.append(tool_result_message(call_id, name, result))
    messages.append({"role": "assistant", "content": canonical_answer(record)})
    return messages


def visible_prompt_text(messages: list[dict[str, Any]]) -> str:
    """Text the model is shown before the first assistant turn, for tests."""
    chunks: list[str] = []
    for message in messages:
        if message["role"] == "assistant":
            break
        content = message.get("content", "")
        if isinstance(content, str):
            chunks.append(content)
        elif isinstance(content, list):
            for part in content:
                if isinstance(part, dict) and part.get("type") == "text":
                    chunks.append(str(part.get("text", "")))
    return "\n".join(chunks)
