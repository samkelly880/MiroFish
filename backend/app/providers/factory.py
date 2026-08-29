"""Resolve and construct LLM providers from configuration."""

from __future__ import annotations

import os
from typing import Any, Optional

from .grok_cli import GrokCLIProvider
from .openai_compat import OpenAICompatProvider

# Aliases accepted in LLM_PROVIDER / --provider
_PROVIDER_ALIASES = {
    "grok-cli": "grok-cli",
    "grok": "grok-cli",
    "grok_cli": "grok-cli",
    "openai-compatible": "openai-compatible",
    "openai": "openai-compatible",
    "openai_compatible": "openai-compatible",
    "api": "openai-compatible",
    "http": "openai-compatible",
}

DEFAULT_PROVIDER = "grok-cli"


def resolve_provider_name(name: Optional[str] = None) -> str:
    raw = (name or os.environ.get("LLM_PROVIDER") or DEFAULT_PROVIDER).strip().lower()
    if raw not in _PROVIDER_ALIASES:
        raise ValueError(
            f"Unsupported LLM_PROVIDER={raw!r}. "
            f"Use one of: {', '.join(sorted(set(_PROVIDER_ALIASES.values())))}"
        )
    return _PROVIDER_ALIASES[raw]


def get_provider_name() -> str:
    return resolve_provider_name()


def create_llm_client(
    provider: Optional[str] = None,
    **kwargs: Any,
):
    """
    Create the configured LLM provider instance.

    Returns a provider object with chat() / chat_json() methods.
    """
    resolved = resolve_provider_name(provider)
    if resolved == "grok-cli":
        return GrokCLIProvider(**{
            k: v for k, v in kwargs.items()
            if k in {"binary", "model", "timeout_seconds", "max_turns", "disallowed_tools", "cwd"}
            and v is not None
        })
    return OpenAICompatProvider(**{
        k: v for k, v in kwargs.items()
        if k in {"api_key", "base_url", "model"} and v is not None
    })
