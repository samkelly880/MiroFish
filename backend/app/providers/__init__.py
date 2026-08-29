"""
LLM provider abstractions for MiroFish.

Primary provider: grok-cli (local Grok Build CLI; no API key required).
Secondary provider: openai-compatible HTTP APIs (optional).
"""

from .factory import create_llm_client, get_provider_name, resolve_provider_name
from .base import LLMProvider

__all__ = [
    "LLMProvider",
    "create_llm_client",
    "get_provider_name",
    "resolve_provider_name",
]
