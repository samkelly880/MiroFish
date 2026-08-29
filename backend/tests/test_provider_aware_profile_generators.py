"""Provider-aware construction of simulation-path LLM generators."""

import pytest

from app.providers.grok_cli import GrokCLIProvider
from app.providers.openai_compat import OpenAICompatProvider
from app.services.oasis_profile_generator import OasisProfileGenerator
from app.services.simulation_config_generator import SimulationConfigGenerator
from app.utils.llm_client import LLMClient


class RecordingLLM:
    """Minimal LLMClient stand-in for profile generation."""

    def __init__(self):
        self.calls = []
        self.model = "recording-model"
        self.provider_name = "fake"

    def chat_json(
        self,
        messages,
        temperature=0.3,
        max_tokens=None,
        max_attempts=1,
        json_schema=None,
    ):
        self.calls.append(
            {
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            }
        )
        return {
            "bio": "A transit advocate.",
            "persona": "Cares about bus lanes and congestion fees.",
            "gender": "female",
            "age": 34,
            "mbti": "ENFJ",
            "country": "China",
            "profession": "Activist",
            "interested_topics": ["transit", "climate"],
        }


@pytest.fixture
def grok_cli_env(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "grok-cli")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    monkeypatch.setattr(
        "app.providers.grok_cli.grok_binary_available", lambda binary=None: True
    )


def test_oasis_profile_generator_accepts_grok_cli_without_api_key(grok_cli_env):
    gen = OasisProfileGenerator(graph_id="g1")
    assert isinstance(gen.llm_client, LLMClient)
    assert gen.llm_client.provider_name == "grok-cli"
    assert isinstance(gen.llm_client._provider, GrokCLIProvider)
    assert not gen.api_key


def test_simulation_config_generator_accepts_grok_cli_without_api_key(grok_cli_env):
    gen = SimulationConfigGenerator()
    assert isinstance(gen.llm_client, LLMClient)
    assert gen.llm_client.provider_name == "grok-cli"
    assert not gen.api_key


def test_oasis_profile_generator_openai_compatible_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        OasisProfileGenerator(graph_id="g1")


def test_simulation_config_generator_openai_compatible_requires_api_key(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.delenv("LLM_API_KEY", raising=False)
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        SimulationConfigGenerator()


def test_oasis_profile_generator_uses_injected_llm_client():
    recording = RecordingLLM()
    gen = object.__new__(OasisProfileGenerator)
    gen.llm_client = recording
    gen.INDIVIDUAL_ENTITY_TYPES = OasisProfileGenerator.INDIVIDUAL_ENTITY_TYPES
    gen.GROUP_ENTITY_TYPES = OasisProfileGenerator.GROUP_ENTITY_TYPES

    result = OasisProfileGenerator._generate_profile_with_llm(
        gen,
        entity_name="Maya Chen",
        entity_type="Activist",
        entity_summary="Supports congestion fees.",
        entity_attributes={},
        context="Downtown vote next week.",
    )
    assert result["bio"] == "A transit advocate."
    assert result["persona"].startswith("Cares about")
    assert recording.calls, "expected llm_client.chat_json to be used"
    assert recording.calls[0]["messages"][0]["role"] == "system"
    assert recording.calls[0]["messages"][1]["role"] == "user"
    assert "Maya Chen" in recording.calls[0]["messages"][1]["content"]


def test_simulation_config_generator_uses_injected_llm_client():
    recording = RecordingLLM()
    gen = object.__new__(SimulationConfigGenerator)
    gen.llm_client = recording
    result = SimulationConfigGenerator._call_llm_with_retry(
        gen, prompt="Return JSON", system_prompt="You are a config generator."
    )
    assert result["bio"] == "A transit advocate."
    assert recording.calls
    assert recording.calls[0]["messages"][0]["content"] == "You are a config generator."


def test_llm_client_defaults_to_grok_cli_without_api_key(grok_cli_env):
    client = LLMClient()
    assert client.provider_name == "grok-cli"
    assert isinstance(client._provider, GrokCLIProvider)


def test_llm_client_openai_compatible_uses_api_config(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai-compatible")
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    monkeypatch.setenv("LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setenv("LLM_MODEL_NAME", "test-model")
    from app import config as config_mod

    monkeypatch.setattr(config_mod.Config, "LLM_API_KEY", "test-key")
    monkeypatch.setattr(config_mod.Config, "LLM_BASE_URL", "https://example.test/v1")
    monkeypatch.setattr(config_mod.Config, "LLM_MODEL_NAME", "test-model")
    monkeypatch.setattr(config_mod.Config, "LLM_PROVIDER", "openai-compatible")

    client = LLMClient()
    assert client.provider_name == "openai-compatible"
    assert isinstance(client._provider, OpenAICompatProvider)
    assert client._provider.api_key == "test-key"
    assert client._provider.base_url == "https://example.test/v1"
