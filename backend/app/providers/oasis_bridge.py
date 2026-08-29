"""
OASIS/CAMEL model bridge with proper action handling for CLI providers.

Unlike amadad's CLIModel (which ignored tool schemas), this bridge:
1. Detects OpenAI tool schemas passed by OASIS
2. Asks the LLM for a JSON action via chat_json + schema
3. Validates the action name/arguments
4. Returns a ChatCompletion with real tool_calls so OASIS continues to work
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import uuid
from typing import Any, Dict, List, Optional

from .action_parser import (
    ActionValidationError,
    action_to_tool_calls_message,
    build_action_instruction,
    build_action_json_schema,
    parse_action_payload,
)
from .factory import create_llm_client, get_provider_name

logger = logging.getLogger(__name__)


def _estimate_tokens(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, str):
        return max(1, math.ceil(len(value) / 4)) if value else 0
    if isinstance(value, list):
        return sum(_estimate_tokens(item) for item in value)
    if isinstance(value, dict):
        return _estimate_tokens(json.dumps(value, ensure_ascii=False))
    return _estimate_tokens(str(value))


def _normalize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    normalized: List[Dict[str, str]] = []
    for message in messages:
        role = str(message.get("role") or "user")
        content = message.get("content")
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text") or ""))
                else:
                    parts.append(str(item))
            content = "\n".join(parts)
        elif content is None:
            content = ""
        else:
            content = str(content)
        normalized.append({"role": role, "content": content})
    return normalized


def build_chat_completion(
    *,
    model: str,
    messages: List[Dict[str, Any]],
    content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> Any:
    """Build a ChatCompletion-like object (dict validated by pydantic if available)."""
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    finish_reason = "stop"
    if tool_calls:
        message["content"] = None
        message["tool_calls"] = tool_calls
        finish_reason = "tool_calls"

    prompt_tokens = sum(_estimate_tokens(m.get("content")) for m in messages)
    completion_tokens = _estimate_tokens(content) + _estimate_tokens(tool_calls)

    payload = {
        "id": f"chatcmpl-mirofish-{uuid.uuid4().hex[:24]}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        },
    }

    try:
        from openai.types.chat.chat_completion import ChatCompletion

        return ChatCompletion.model_validate(payload)
    except Exception:  # noqa: BLE001 - dict is fine for tests / light usage
        return payload


class OasisLLMBridge:
    """
    Provider-agnostic completion helper used by OASIS runner scripts.

    For grok-cli (and any non-native tool provider), tool requests are converted
    into validated JSON actions. For openai-compatible HTTP, callers may still
    use CAMEL's native OpenAI path; this bridge remains available for tests.
    """

    def __init__(self, provider_name: Optional[str] = None, **provider_kwargs: Any):
        self.provider_name = provider_name or get_provider_name()
        self.llm = create_llm_client(provider=self.provider_name, **provider_kwargs)
        self.model_name = getattr(self.llm, "model", None) or self.provider_name

    def complete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
    ) -> Any:
        normalized = _normalize_messages(messages)
        if tools:
            return self._complete_with_tools(
                normalized,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        text = self.llm.chat(
            messages=normalized,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return build_chat_completion(
            model=self.model_name,
            messages=normalized,
            content=text,
        )

    def _complete_with_tools(
        self,
        messages: List[Dict[str, str]],
        *,
        tools: List[Dict[str, Any]],
        temperature: float,
        max_tokens: Optional[int],
    ) -> Any:
        instruction = build_action_instruction(tools)
        schema = build_action_json_schema(tools)
        action_messages = list(messages) + [
            {"role": "system", "content": instruction}
        ]

        try:
            payload = self.llm.chat_json(
                messages=action_messages,
                temperature=min(temperature, 0.4),
                max_tokens=max_tokens,
                max_attempts=2,
                json_schema=schema,
            )
            name, arguments = parse_action_payload(payload, tools)
        except (ActionValidationError, Exception) as exc:
            logger.warning(
                "OASIS action parse failed (%s); falling back to DO_NOTHING",
                exc,
            )
            from .action_parser import allowed_action_names

            allowed = allowed_action_names(tools)
            if "DO_NOTHING" in allowed:
                name, arguments = "DO_NOTHING", {}
            else:
                raise

        tool_message = action_to_tool_calls_message(name, arguments)
        return build_chat_completion(
            model=self.model_name,
            messages=messages,
            tool_calls=tool_message["tool_calls"],
        )

    async def acomplete(
        self,
        messages: List[Dict[str, Any]],
        tools: Optional[List[Dict[str, Any]]] = None,
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
    ) -> Any:
        return await asyncio.to_thread(
            self.complete,
            messages,
            tools,
            temperature,
            max_tokens,
        )


def create_oasis_model(config: Optional[Dict[str, Any]] = None, use_boost: bool = False):
    """
    Create a CAMEL-compatible model for OASIS simulations.

    - grok-cli: returns a CLI-backed OpenAIModel subclass with tool→JSON bridge
    - openai-compatible: returns CAMEL ModelFactory OpenAI platform model
    """
    del use_boost  # boost remains available via env in runner scripts for API mode
    config = config or {}
    provider = (
        config.get("llm_provider")
        or get_provider_name()
    )

    if provider == "openai-compatible":
        import os
        from camel.types import ModelPlatformType
        from camel.models import ModelFactory
        from ..config import Config

        llm_api_key = os.environ.get("LLM_API_KEY") or Config.LLM_API_KEY
        llm_base_url = os.environ.get("LLM_BASE_URL") or Config.LLM_BASE_URL
        llm_model = (
            config.get("llm_model")
            or os.environ.get("LLM_MODEL_NAME")
            or Config.LLM_MODEL_NAME
        )
        if not llm_api_key:
            raise ValueError(
                "LLM_API_KEY is required for openai-compatible OASIS runs"
            )
        os.environ["OPENAI_API_KEY"] = llm_api_key
        if llm_base_url:
            os.environ["OPENAI_API_BASE_URL"] = llm_base_url
        return ModelFactory.create(
            model_platform=ModelPlatformType.OPENAI,
            model_type=llm_model,
        )

    # grok-cli path: subclass OpenAIModel when camel is available
    try:
        from camel.models.openai_model import OpenAIModel
    except ImportError as exc:  # pragma: no cover - runtime OASIS dependency
        raise ImportError(
            "camel-ai is required for OASIS simulations with grok-cli"
        ) from exc

    bridge = OasisLLMBridge(provider_name="grok-cli")

    class GrokCLIOasisModel(OpenAIModel):
        def __init__(self) -> None:
            super().__init__(
                model_type=bridge.model_name or "grok-cli",
                model_config_dict={},
                api_key="grok-cli-bridge",
                url=None,
                timeout=None,
                max_retries=2,
            )
            self._bridge = bridge

        def _request_chat_completion(
            self,
            messages: List[Dict[str, Any]],
            tools: List[Dict[str, Any]] | None = None,
        ):
            temperature = float(
                (self.model_config_dict or {}).get("temperature", 0.7) or 0.7
            )
            max_tokens = (self.model_config_dict or {}).get("max_tokens", 4096)
            return self._bridge.complete(
                messages=messages,
                tools=tools,
                temperature=temperature,
                max_tokens=max_tokens,
            )

        async def _arequest_chat_completion(
            self,
            messages: List[Dict[str, Any]],
            tools: List[Dict[str, Any]] | None = None,
        ):
            return await asyncio.to_thread(
                self._request_chat_completion, messages, tools
            )

    logger.info("OASIS model: provider=grok-cli mode=action-bridge")
    return GrokCLIOasisModel()
