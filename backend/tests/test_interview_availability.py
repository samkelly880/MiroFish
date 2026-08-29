"""Skip interview_agents when OASIS env is closed; keep it when alive."""

from types import SimpleNamespace

import pytest

from app.services.report_agent import ReportAgent
from app.services.zep_tools import InterviewResult, ZepToolsService


class RecordingZepTools(ZepToolsService):
    def __init__(self):
        # Bypass normal init (no Zep/LLM needed for these unit tests)
        self.interview_calls = []
        self.insight_calls = []

    def interview_agents(self, **kwargs):
        self.interview_calls.append(kwargs)
        result = InterviewResult(
            interview_topic=kwargs.get("interview_requirement", ""),
            interview_questions=[],
        )
        result.summary = "live interview ok"
        return result

    def insight_forge(self, **kwargs):
        self.insight_calls.append(kwargs)
        return SimpleNamespace(to_text=lambda: "insight ok")

    def panorama_search(self, **kwargs):
        return SimpleNamespace(to_text=lambda: "panorama ok")

    def quick_search(self, **kwargs):
        return SimpleNamespace(to_text=lambda: "quick ok")


@pytest.fixture
def agent_closed(monkeypatch):
    tools = RecordingZepTools()
    agent = ReportAgent(
        graph_id="graph_x",
        simulation_id="sim_closed",
        simulation_requirement="test requirement",
        llm_client=SimpleNamespace(chat=lambda **k: "Final Answer: body"),
        zep_tools=tools,
    )
    monkeypatch.setattr(
        "app.services.simulation_runner.SimulationRunner.check_env_alive",
        classmethod(lambda cls, simulation_id: False),
    )
    # Reset cache after monkeypatch
    agent._interviews_available_cached = None
    return agent, tools


@pytest.fixture
def agent_live(monkeypatch):
    tools = RecordingZepTools()
    agent = ReportAgent(
        graph_id="graph_x",
        simulation_id="sim_live",
        simulation_requirement="test requirement",
        llm_client=SimpleNamespace(chat=lambda **k: "Final Answer: body"),
        zep_tools=tools,
    )
    monkeypatch.setattr(
        "app.services.simulation_runner.SimulationRunner.check_env_alive",
        classmethod(lambda cls, simulation_id: True),
    )
    agent._interviews_available_cached = None
    return agent, tools


def test_closed_env_hides_interview_from_available_tools(agent_closed):
    agent, _ = agent_closed
    assert agent.interviews_available() is False
    assert "interview_agents" not in agent._available_tool_names()
    assert "insight_forge" in agent._available_tool_names()
    desc = agent._get_tools_description()
    assert "interview_agents" not in desc
    assert "insight_forge" in desc
    advice = agent._get_tool_usage_advice()
    assert "unavailable" in advice
    assert "interview_agents:" not in advice.split("unavailable")[0]


def test_live_env_keeps_interview_tool(agent_live):
    agent, _ = agent_live
    assert agent.interviews_available() is True
    assert "interview_agents" in agent._available_tool_names()
    assert "interview_agents" in agent._get_tools_description()
    assert "interview_agents:" in agent._get_tool_usage_advice()


def test_closed_env_execute_tool_does_not_call_interview(agent_closed):
    agent, tools = agent_closed
    result = agent._execute_tool(
        "interview_agents",
        {"interview_topic": "views on the fee"},
    )
    assert "unavailable" in result.lower() or "not running" in result.lower()
    assert tools.interview_calls == []


def test_live_env_execute_tool_calls_interview(agent_live):
    agent, tools = agent_live
    result = agent._execute_tool(
        "interview_agents",
        {"interview_topic": "views on the fee"},
    )
    assert "live interview ok" in result
    assert len(tools.interview_calls) == 1


def test_closed_env_rejects_interview_in_tool_parse(agent_closed):
    agent, _ = agent_closed
    assert agent._is_valid_tool_call(
        {"name": "interview_agents", "parameters": {"interview_topic": "x"}}
    ) is False
    assert agent._is_valid_tool_call(
        {"name": "quick_search", "parameters": {"query": "x"}}
    ) is True


def test_zep_tools_interview_skips_llm_when_env_closed(monkeypatch):
    """Defense in depth: interview_agents returns before LLM selection."""
    calls = {"select": 0}

    monkeypatch.setattr(
        "app.services.simulation_runner.SimulationRunner.check_env_alive",
        classmethod(lambda cls, simulation_id: False),
    )

    tools = object.__new__(ZepToolsService)

    def boom_select(*a, **k):
        calls["select"] += 1
        raise AssertionError("should not select agents")

    tools._load_agent_profiles = lambda simulation_id: [{"user_id": 0}]
    tools._select_agents_for_interview = boom_select

    result = ZepToolsService.interview_agents(
        tools,
        simulation_id="sim_x",
        interview_requirement="topic",
    )
    assert "not running" in result.summary.lower() or "skipped" in result.summary.lower()
    assert result.interviews == []
    assert calls["select"] == 0


def test_section_prompt_omits_interview_when_closed(agent_closed):
    agent, _ = agent_closed
    from app.services.report_agent import ReportOutline, ReportSection

    outline = ReportOutline(title="T", summary="S", sections=[ReportSection(title="Sec")])
    # Build the same prompt fragments used by ReACT
    tools_desc = agent._get_tools_description()
    advice = agent._get_tool_usage_advice()
    assert "interview_agents" not in tools_desc
    assert "interview_agents unavailable" in advice or "not running" in advice
