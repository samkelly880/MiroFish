"""
Parse and validate OASIS agent actions from model JSON.

OASIS/CAMEL normally uses OpenAI tool/function calling. CLI providers do not
emit native tool_calls, so MiroFish asks the model for a JSON action object and
converts a validated payload into an OpenAI-compatible tool_calls completion.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional, Tuple


class ActionValidationError(ValueError):
    """Raised when a model action payload is missing or invalid."""


def tool_name(tool: Dict[str, Any]) -> str:
    fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
    return str((fn or {}).get("name") or tool.get("name") or "").strip()


def allowed_action_names(tools: List[Dict[str, Any]]) -> List[str]:
    names = []
    for tool in tools or []:
        name = tool_name(tool)
        if name:
            names.append(name)
    return names


def build_action_json_schema(tools: List[Dict[str, Any]]) -> Dict[str, Any]:
    """JSON Schema forcing a single action selection among OASIS tools."""
    names = allowed_action_names(tools)
    return {
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "enum": names or ["DO_NOTHING"],
                "description": "OASIS action / tool name to invoke",
            },
            "arguments": {
                "type": "object",
                "description": "Arguments for the selected action",
                "additionalProperties": True,
            },
        },
        "required": ["name", "arguments"],
        "additionalProperties": False,
    }


def build_action_instruction(tools: List[Dict[str, Any]]) -> str:
    """Prompt appendix describing available OASIS actions."""
    lines = [
        "You are selecting the next OASIS social-media agent action.",
        "Choose exactly one action from the catalog below.",
        "Return ONLY a JSON object: {\"name\": \"<ACTION>\", \"arguments\": {..}}.",
        "If no meaningful action applies, use DO_NOTHING with empty arguments.",
        "",
        "Available actions:",
    ]
    for tool in tools or []:
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        name = tool_name(tool)
        if not name:
            continue
        desc = (fn or {}).get("description") or tool.get("description") or ""
        params = (fn or {}).get("parameters") or tool.get("parameters") or {}
        lines.append(f"- {name}: {desc}".rstrip())
        if params:
            lines.append(
                f"  parameters schema: {json.dumps(params, ensure_ascii=False)}"
            )
    return "\n".join(lines)


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned).strip()
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        value, _ = json.JSONDecoder().raw_decode(cleaned)
    if not isinstance(value, dict):
        raise ActionValidationError("Action payload must be a JSON object")
    return value


def parse_action_payload(
    payload: Any,
    tools: List[Dict[str, Any]],
) -> Tuple[str, Dict[str, Any]]:
    """
    Validate an action payload against the tool catalog.

    Accepts either a dict or a JSON string. Returns (name, arguments).
    """
    if isinstance(payload, str):
        data = _extract_json_object(payload)
    elif isinstance(payload, dict):
        data = payload
    else:
        raise ActionValidationError("Action payload must be a dict or JSON string")

    # Accept a few common shapes
    name = data.get("name") or data.get("action") or data.get("tool")
    arguments = data.get("arguments")
    if arguments is None:
        arguments = data.get("args") or data.get("parameters") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError as exc:
            raise ActionValidationError(
                "Action arguments must be a JSON object"
            ) from exc
    if not isinstance(arguments, dict):
        raise ActionValidationError("Action arguments must be an object")
    if not name or not isinstance(name, str):
        raise ActionValidationError("Action name is required")

    name = name.strip()
    allowed = set(allowed_action_names(tools))
    if allowed and name not in allowed:
        # Soft-fallback: DO_NOTHING if present, else fail
        if "DO_NOTHING" in allowed:
            return "DO_NOTHING", {}
        raise ActionValidationError(
            f"Unknown action {name!r}; allowed: {sorted(allowed)}"
        )

    # Light required-property check when the tool schema lists required fields
    for tool in tools or []:
        if tool_name(tool) != name:
            continue
        fn = tool.get("function") if isinstance(tool.get("function"), dict) else tool
        params = (fn or {}).get("parameters") or {}
        required = params.get("required") if isinstance(params, dict) else None
        if isinstance(required, list):
            missing = [key for key in required if key not in arguments]
            if missing:
                raise ActionValidationError(
                    f"Action {name} missing required arguments: {missing}"
                )
        break

    return name, arguments


def action_to_tool_calls_message(
    name: str,
    arguments: Dict[str, Any],
    *,
    call_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build an OpenAI-style assistant message containing tool_calls."""
    import uuid

    tool_call_id = call_id or f"call_{uuid.uuid4().hex[:24]}"
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {
                "id": tool_call_id,
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            }
        ],
    }
