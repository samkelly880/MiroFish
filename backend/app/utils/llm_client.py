"""
LLM client wrapper.

Primary provider: grok-cli (local Grok Build CLI; no API key required).
Secondary provider: openai-compatible HTTP APIs.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from ..providers.factory import create_llm_client, get_provider_name, resolve_provider_name
from ..providers.openai_compat import LLMResponseError

__all__ = ["LLMClient", "LLMResponseError"]


class LLMClient:
    """
    Backward-compatible LLM client used throughout MiroFish services.

    Delegates to the configured provider (default: grok-cli).
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ):
        self.provider_name = (
            resolve_provider_name(provider)
            if provider is not None
            else get_provider_name()
        )
        kwargs: Dict[str, Any] = {}
        if model is not None:
            kwargs["model"] = model
        if self.provider_name == "openai-compatible":
            if api_key is not None:
                kwargs["api_key"] = api_key
            if base_url is not None:
                kwargs["base_url"] = base_url
        self._provider = create_llm_client(provider=self.provider_name, **kwargs)
        # Compatibility attributes used by older code / logging
        self.api_key = api_key
        self.base_url = base_url
        self.model = model or getattr(self._provider, "model", None)

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        response_format: Optional[Dict] = None,
    ) -> str:
        return self._provider.chat(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=response_format,
        )

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = 4096,
        max_attempts: int = 1,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return self._provider.chat_json(
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            max_attempts=max_attempts,
            json_schema=json_schema,
        )
