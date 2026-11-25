"""Geo tools the model can call during SFT traces and RL rollouts.

``lookup_site_context`` is a stub. It returns fields stored on the episode.
It does not call elevation, weather, AIS, ADS-B, permit, or map services.

TODO(tools): replace the stub with live providers. Keep the same argument
and result shape so rendered traces stay valid. Download imagery in the data
pipeline; the Inkling renderer refuses remote image URLs.
"""

from __future__ import annotations

import json
from typing import Any

LOOKUP_SITE_CONTEXT_SPEC: dict[str, Any] = {
    "name": "lookup_site_context",
    "description": (
        "Look up structured context for one site: elevation, weather, vessel "
        "counts, flight counts, permits, or map attributes bundled with the "
        "episode. Sample episodes return only the fields stored on that "
        "episode. Live providers are not wired up."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "site_id": {"type": "string", "description": "Site the question is about."},
            "fields": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Context keys to return. Omit to return every key.",
            },
        },
        "required": ["site_id"],
    },
}

TOOL_SPECS: list[dict[str, Any]] = [LOOKUP_SITE_CONTEXT_SPEC]


def openai_tool_declarations(specs: list[dict[str, Any]] | None = None) -> str:
    """JSON the cookbook writes into a ``tool_declare`` message.

    Matches ``tinker_cookbook.third_party.openai_compat.tool_specs_to_openai_tools``
    so a hand-built prefix and ``renderer.create_conversation_prefix_with_tools``
    describe the same tools.
    """
    payload = [
        {
            "type": "function",
            "function": {
                "name": spec["name"],
                "description": spec["description"],
                "parameters": spec["parameters"],
            },
        }
        for spec in (specs if specs is not None else TOOL_SPECS)
    ]
    return json.dumps(payload, separators=(",", ":"))


class SiteContextTool:
    """Episode-local context lookup."""

    name = LOOKUP_SITE_CONTEXT_SPEC["name"]

    def __init__(
        self,
        site_id: str,
        context: dict[str, Any],
        *,
        source_uri: str,
        observed_at: str,
    ) -> None:
        self.site_id = site_id
        self.context = dict(context)
        self.source_uri = source_uri
        self.observed_at = observed_at

    def run(self, arguments: dict[str, Any]) -> dict[str, Any]:
        requested_site = arguments.get("site_id")
        if requested_site != self.site_id:
            return {
                "ok": False,
                "error": "unknown_site",
                "site_id": requested_site,
                "message": "This episode only has context for the site in the question.",
            }
        fields = arguments.get("fields")
        if fields is None:
            selected = dict(self.context)
            unknown: list[str] = []
        else:
            if not isinstance(fields, list) or not all(isinstance(item, str) for item in fields):
                return {"ok": False, "error": "fields must be a list of strings"}
            unknown = [name for name in fields if name not in self.context]
            selected = {name: self.context[name] for name in fields if name in self.context}
        return {
            "ok": True,
            "site_id": self.site_id,
            "observed_at": self.observed_at,
            "source_uri": self.source_uri,
            "fields": selected,
            "unknown_fields": unknown,
            "available_fields": sorted(self.context),
        }


def execute_tool_call(tool: SiteContextTool, call: Any) -> tuple[str, str, dict[str, Any]]:
    """Run one parsed or serialized tool call.

    Returns ``(call_id, tool_name, result_dict)``.
    """
    name, arguments, call_id = _coerce_call(call)
    if name != tool.name:
        result = {
            "ok": False,
            "error": "unknown_tool",
            "name": name,
            "message": f"Only {tool.name} is available in this skeleton.",
        }
        return call_id, name, result
    try:
        result = tool.run(arguments)
    except Exception as exc:  # noqa: BLE001 — surface tool failures as tool results
        result = {"ok": False, "error": "tool_exception", "message": str(exc)}
    return call_id, name, result


def tool_result_message(call_id: str, name: str, result: dict[str, Any]) -> dict[str, Any]:
    return {
        "role": "tool",
        "name": name,
        "tool_call_id": call_id,
        "content": json.dumps(result, ensure_ascii=False, sort_keys=True),
    }


def _coerce_call(call: Any) -> tuple[str, dict[str, Any], str]:
    if isinstance(call, dict):
        function = call.get("function") or {}
        name = function.get("name") or call.get("name") or ""
        raw_args = function.get("arguments", call.get("arguments", "{}"))
        call_id = str(call.get("id") or "")
    else:
        function = call.function
        name = function.name
        raw_args = function.arguments
        call_id = str(call.id or "")
    if isinstance(raw_args, str):
        try:
            arguments = json.loads(raw_args) if raw_args else {}
        except json.JSONDecodeError as exc:
            raise ValueError(f"tool arguments are not JSON: {exc}") from exc
    elif isinstance(raw_args, dict):
        arguments = raw_args
    else:
        raise ValueError("tool arguments must be a JSON object")
    if not isinstance(arguments, dict):
        raise ValueError("tool arguments must be a JSON object")
    return str(name), arguments, call_id
