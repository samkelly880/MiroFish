"""Generate structured verdict.json from a simulation report."""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from ..utils.llm_client import LLMClient

VERDICT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "prediction": {
            "type": "string",
            "description": "One-sentence prediction (max ~100 words)",
        },
        "confidence": {
            "type": "number",
            "minimum": 0,
            "maximum": 1,
            "description": "Confidence in the simulation evidence from 0.0 to 1.0",
        },
        "key_dynamics": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Important social dynamics observed",
        },
        "signals": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Evidence signals supporting the prediction",
        },
        "insufficient_data": {
            "type": "boolean",
            "description": "True when the report lacks enough evidence",
        },
    },
    "required": [
        "prediction",
        "confidence",
        "key_dynamics",
        "signals",
        "insufficient_data",
    ],
    "additionalProperties": False,
}


def generate_verdict(
    report_markdown: str,
    requirement: str,
    *,
    llm: Optional[LLMClient] = None,
) -> Dict[str, Any]:
    """Extract a machine-readable verdict from report markdown."""
    text = (report_markdown or "").strip()
    if len(text) < 40:
        return {
            "prediction": "Insufficient report content to form a prediction.",
            "confidence": 0.0,
            "key_dynamics": [],
            "signals": [],
            "insufficient_data": True,
        }

    client = llm or LLMClient()
    messages = [
        {
            "role": "system",
            "content": (
                "You extract a structured verdict from a MiroFish simulation report. "
                "Return JSON with exactly the schema fields. "
                "If evidence is weak, set insufficient_data=true and lower confidence."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Simulation requirement:\n{requirement}\n\n"
                f"Report markdown:\n{text[:50000]}\n\n"
                "Return JSON with keys: prediction, confidence, key_dynamics, "
                "signals, insufficient_data."
            ),
        },
    ]
    result = client.chat_json(
        messages=messages,
        temperature=0.2,
        max_attempts=2,
        json_schema=VERDICT_SCHEMA,
    )
    return normalize_verdict(result)


def normalize_verdict(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Clamp/normalize verdict fields for stable agent consumption."""
    prediction = str(payload.get("prediction") or "").strip()
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    def _string_list(value: Any) -> list:
        if not isinstance(value, list):
            return []
        return [str(item).strip() for item in value if str(item).strip()]

    insufficient = bool(payload.get("insufficient_data"))
    if not prediction:
        insufficient = True
        prediction = "Insufficient data to form a prediction."
        confidence = 0.0

    return {
        "prediction": prediction,
        "confidence": confidence,
        "key_dynamics": _string_list(payload.get("key_dynamics")),
        "signals": _string_list(payload.get("signals")),
        "insufficient_data": insufficient,
    }


def verdict_to_json(payload: Dict[str, Any]) -> str:
    return json.dumps(normalize_verdict(payload), ensure_ascii=False, indent=2)
