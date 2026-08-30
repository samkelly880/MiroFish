"""
Grok Build CLI provider (primary LLM backend).

Uses `grok --prompt-file ...` so large prompts are never passed on argv.
Does not require XAI_API_KEY when the local CLI is already authenticated
(`grok login` / cached token).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Coding-agent tools that must not run for MiroFish prompt completions.
_DEFAULT_DISALLOWED_TOOLS = ",".join([
    "Bash",
    "Edit",
    "Write",
    "Read",
    "Glob",
    "Grep",
    "WebSearch",
    "WebFetch",
    "Agent",
    "Task",
    "NotebookEdit",
    "Skill",
])


class GrokCLIError(RuntimeError):
    """Raised when the Grok CLI invocation fails."""


def grok_binary_available(binary: Optional[str] = None) -> bool:
    """Return True when the grok executable is on PATH (or an explicit path)."""
    path = binary or os.environ.get("GROK_CLI_BIN", "grok")
    if os.path.isabs(path) or os.sep in path:
        return os.path.isfile(path) and os.access(path, os.X_OK)
    return shutil.which(path) is not None


def _clean_chat_text(content: str) -> str:
    cleaned = re.sub(r"<think>[\s\S]*?</think>", "", content).strip()
    cleaned = cleaned.lstrip("\ufeff")
    cleaned = re.sub(r"^```(?:json)?\s*\n?", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\n?```\s*$", "", cleaned)
    return cleaned.strip()


def _messages_to_prompt(messages: List[Dict[str, str]]) -> str:
    """Flatten OpenAI-style messages into a single headless prompt."""
    parts: List[str] = []
    for msg in messages:
        role = (msg.get("role") or "user").strip().lower()
        content = msg.get("content") or ""
        if role == "system":
            parts.append(f"[System]\n{content}")
        elif role == "assistant":
            parts.append(f"[Assistant]\n{content}")
        else:
            parts.append(f"[User]\n{content}")
    parts.append(
        "\n[Instruction]\nRespond as the Assistant to the latest User message. "
        "Follow any System instructions carefully."
    )
    return "\n\n".join(parts)


def _scrub_env(base: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """
    Build a subprocess environment.

    Keep auth-related vars the Grok CLI may need, but drop MiroFish LLM API
    keys so they are not inherited into the agent process unnecessarily.
    """
    env = dict(base or os.environ)
    for key in (
        "LLM_API_KEY",
        "LLM_BOOST_API_KEY",
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
    ):
        env.pop(key, None)
    # Prefer local CLI auth; do not force API-key mode unless the user set it.
    # XAI_API_KEY is left intact if present (optional fallback for headless CI).
    env.setdefault("NO_COLOR", "1")
    return env


class GrokCLIProvider:
    """LLM provider backed by the local `grok` CLI."""

    name = "grok-cli"

    def __init__(
        self,
        binary: Optional[str] = None,
        model: Optional[str] = None,
        timeout_seconds: Optional[int] = None,
        max_turns: Optional[int] = None,
        disallowed_tools: Optional[str] = None,
        cwd: Optional[str] = None,
    ):
        self.binary = binary or os.environ.get("GROK_CLI_BIN", "grok")
        self.model = model or os.environ.get("GROK_CLI_MODEL") or os.environ.get(
            "LLM_MODEL_NAME"
        )
        self.timeout_seconds = timeout_seconds or int(
            os.environ.get("GROK_CLI_TIMEOUT_SECONDS", "300")
        )
        # ReportAgent ReACT prompts can require multiple headless turns even with
        # coding tools disabled. Default 1 was too low and surfaced as
        # "max turns reached" during section generation.
        if max_turns is None:
            max_turns = int(os.environ.get("GROK_CLI_MAX_TURNS", "16"))
        self.max_turns = max(1, max_turns)
        self.disallowed_tools = (
            disallowed_tools
            if disallowed_tools is not None
            else os.environ.get("GROK_CLI_DISALLOWED_TOOLS", _DEFAULT_DISALLOWED_TOOLS)
        )
        # Isolated cwd avoids pulling the MiroFish repo into Grok's workspace context.
        self.cwd = cwd  # resolved per-call if None

        if not grok_binary_available(self.binary):
            raise GrokCLIError(
                f"Grok CLI binary not found: {self.binary!r}. "
                "Install from https://x.ai/cli/install.sh and run `grok login`."
            )

    def _run(
        self,
        prompt: str,
        *,
        json_schema: Optional[Dict[str, Any]] = None,
        output_format: str = "json",
    ) -> Dict[str, Any]:
        with tempfile.TemporaryDirectory(prefix="mirofish-grok-") as tmp:
            prompt_path = os.path.join(tmp, "prompt.txt")
            with open(prompt_path, "w", encoding="utf-8") as handle:
                handle.write(prompt)

            work_cwd = self.cwd or tmp
            cmd = [
                self.binary,
                "--no-auto-update",
                "--prompt-file",
                prompt_path,
                "--output-format",
                output_format,
                "--max-turns",
                str(self.max_turns),
                "--cwd",
                work_cwd,
            ]
            if self.disallowed_tools:
                cmd.extend(["--disallowed-tools", self.disallowed_tools])
            # Prefer an explicit empty allowlist when supported; combine with deny list.
            cmd.extend(["--tools", ""])
            if self.model:
                cmd.extend(["-m", self.model])
            if json_schema is not None:
                cmd.extend(["--json-schema", json.dumps(json_schema, ensure_ascii=False)])

            try:
                completed = subprocess.run(
                    cmd,
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_seconds,
                    env=_scrub_env(),
                    shell=False,
                )
            except subprocess.TimeoutExpired as exc:
                raise GrokCLIError(
                    f"Grok CLI timed out after {self.timeout_seconds}s"
                ) from exc

            stdout = (completed.stdout or "").strip()
            stderr = (completed.stderr or "").strip()
            if completed.returncode != 0:
                detail = stderr or stdout or f"exit {completed.returncode}"
                raise GrokCLIError(f"Grok CLI failed: {detail}")

            if output_format == "plain":
                return {"text": stdout}

            if not stdout:
                raise GrokCLIError("Grok CLI returned empty stdout")

            # CLI may emit progress on stderr; stdout should be one JSON object.
            try:
                return json.loads(stdout)
            except json.JSONDecodeError:
                # Some versions may wrap multiple lines; take the last JSON object.
                for line in reversed(stdout.splitlines()):
                    line = line.strip()
                    if not line.startswith("{"):
                        continue
                    try:
                        return json.loads(line)
                    except json.JSONDecodeError:
                        continue
                raise GrokCLIError(
                    "Grok CLI returned non-JSON stdout in json mode"
                )

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: Optional[int] = 4096,
        response_format: Optional[Dict[str, Any]] = None,
    ) -> str:
        del temperature, max_tokens  # CLI does not expose these knobs reliably
        prompt = _messages_to_prompt(messages)
        if response_format and response_format.get("type") == "json_object":
            prompt += (
                "\n\n[Output format]\nReturn a single JSON object only. "
                "Do not wrap it in markdown fences."
            )
        payload = self._run(prompt, output_format="json")
        text = payload.get("text")
        if text is None and isinstance(payload.get("structuredOutput"), dict):
            return json.dumps(payload["structuredOutput"], ensure_ascii=False)
        if not isinstance(text, str):
            raise GrokCLIError("Grok CLI JSON response missing text field")
        return _clean_chat_text(text)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.3,
        max_tokens: Optional[int] = 4096,
        max_attempts: int = 1,
        json_schema: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        del temperature, max_tokens
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")

        schema = json_schema or {
            "type": "object",
            "additionalProperties": True,
        }
        last_error: Optional[Exception] = None
        for attempt in range(1, max_attempts + 1):
            prompt = _messages_to_prompt(messages)
            prompt += (
                "\n\n[Output format]\nReturn a single JSON object matching the "
                "requested schema. Do not include markdown fences."
            )
            try:
                payload = self._run(prompt, json_schema=schema, output_format="json")
                structured = payload.get("structuredOutput")
                if isinstance(structured, dict):
                    return structured
                text = _clean_chat_text(str(payload.get("text") or ""))
                if not text:
                    raise GrokCLIError("Grok CLI returned empty JSON content")
                value = json.loads(text)
                if not isinstance(value, dict):
                    # Accept raw_decode for trailing prose
                    value, _ = json.JSONDecoder().raw_decode(text)
                if not isinstance(value, dict):
                    raise GrokCLIError("Grok CLI JSON must be a top-level object")
                return value
            except Exception as exc:  # noqa: BLE001 - retry then raise
                last_error = exc
                logger.warning(
                    "Grok CLI chat_json attempt %s/%s failed: %s",
                    attempt,
                    max_attempts,
                    exc,
                )
        assert last_error is not None
        raise last_error
