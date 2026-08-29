import pytest

from app.providers.action_parser import (
    ActionValidationError,
    action_to_tool_calls_message,
    allowed_action_names,
    build_action_instruction,
    build_action_json_schema,
    parse_action_payload,
)

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "CREATE_POST",
            "description": "Create a post",
            "parameters": {
                "type": "object",
                "properties": {"content": {"type": "string"}},
                "required": ["content"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "DO_NOTHING",
            "description": "Skip this turn",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]


def test_allowed_action_names():
    assert allowed_action_names(TOOLS) == ["CREATE_POST", "DO_NOTHING"]


def test_parse_valid_action():
    name, args = parse_action_payload(
        {"name": "CREATE_POST", "arguments": {"content": "hello"}},
        TOOLS,
    )
    assert name == "CREATE_POST"
    assert args == {"content": "hello"}


def test_parse_action_from_json_string():
    name, args = parse_action_payload(
        '{"name":"DO_NOTHING","arguments":{}}',
        TOOLS,
    )
    assert name == "DO_NOTHING"
    assert args == {}


def test_unknown_action_falls_back_to_do_nothing():
    name, args = parse_action_payload({"name": "HACK", "arguments": {}}, TOOLS)
    assert name == "DO_NOTHING"
    assert args == {}


def test_missing_required_argument():
    with pytest.raises(ActionValidationError, match="missing required"):
        parse_action_payload({"name": "CREATE_POST", "arguments": {}}, TOOLS)


def test_action_to_tool_calls_message():
    message = action_to_tool_calls_message("CREATE_POST", {"content": "x"})
    assert message["role"] == "assistant"
    assert message["content"] is None
    assert message["tool_calls"][0]["function"]["name"] == "CREATE_POST"
    assert '"content": "x"' in message["tool_calls"][0]["function"]["arguments"]


def test_schema_and_instruction_include_actions():
    schema = build_action_json_schema(TOOLS)
    assert schema["properties"]["name"]["enum"] == ["CREATE_POST", "DO_NOTHING"]
    text = build_action_instruction(TOOLS)
    assert "CREATE_POST" in text
    assert "DO_NOTHING" in text
