import json
import os
from pathlib import Path

import pytest

from app.providers.grok_cli import GrokCLIError, GrokCLIProvider


class DummyCompleted:
    def __init__(self, stdout="", stderr="", returncode=0):
        self.stdout = stdout
        self.stderr = stderr
        self.returncode = returncode


def test_grok_cli_uses_prompt_file_not_argv(monkeypatch, tmp_path):
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        assert "--prompt-file" in cmd
        prompt_file = cmd[cmd.index("--prompt-file") + 1]
        assert Path(prompt_file).is_file()
        # Large prompt must not appear as a bare -p argument value
        assert "-p" not in cmd
        assert kwargs.get("shell") is False
        env = kwargs.get("env") or {}
        assert "LLM_API_KEY" not in env
        return DummyCompleted(
            stdout=json.dumps({"text": "hello world", "stopReason": "end_turn"})
        )

    monkeypatch.setattr("app.providers.grok_cli.subprocess.run", fake_run)
    monkeypatch.setattr("app.providers.grok_cli.grok_binary_available", lambda binary=None: True)

    provider = GrokCLIProvider(binary="grok", cwd=str(tmp_path))
    text = provider.chat([{"role": "user", "content": "Say hello"}])
    assert text == "hello world"
    assert calls


def test_grok_cli_chat_json_prefers_structured_output(monkeypatch, tmp_path):
    def fake_run(cmd, **kwargs):
        assert "--json-schema" in cmd
        return DummyCompleted(
            stdout=json.dumps(
                {
                    "text": "{\"ignored\": true}",
                    "structuredOutput": {"prediction": "ok", "confidence": 0.5},
                }
            )
        )

    monkeypatch.setattr("app.providers.grok_cli.subprocess.run", fake_run)
    monkeypatch.setattr("app.providers.grok_cli.grok_binary_available", lambda binary=None: True)
    provider = GrokCLIProvider(binary="grok", cwd=str(tmp_path))
    result = provider.chat_json([{"role": "user", "content": "json please"}])
    assert result["prediction"] == "ok"


def test_grok_cli_nonzero_exit(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "app.providers.grok_cli.subprocess.run",
        lambda *a, **k: DummyCompleted(stdout="", stderr="boom", returncode=2),
    )
    monkeypatch.setattr("app.providers.grok_cli.grok_binary_available", lambda binary=None: True)
    provider = GrokCLIProvider(binary="grok", cwd=str(tmp_path))
    with pytest.raises(GrokCLIError, match="boom"):
        provider.chat([{"role": "user", "content": "x"}])


def test_factory_defaults_to_grok(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "grok-cli")
    monkeypatch.setattr("app.providers.grok_cli.grok_binary_available", lambda binary=None: True)
    from app.providers.factory import create_llm_client, get_provider_name

    assert get_provider_name() == "grok-cli"
    client = create_llm_client()
    assert client.name == "grok-cli"
