"""Provider protocol shared by Grok CLI and OpenAI-compatible backends."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class LLMProvider(Protocol):
    """Minimal chat interface used by MiroFish services."""

    name: str

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Return assistant text for a chat-style message list."""

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = 4096,
        max_attempts: int = 1,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Return a parsed JSON object."""
