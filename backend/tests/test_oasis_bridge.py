from app.providers.oasis_bridge import OasisLLMBridge, build_chat_completion


class FakeLLM:
    def __init__(self, payload):
        self.payload = payload
        self.model = "fake"
        self.calls = []

    def chat(self, messages, temperature=0.7, max_tokens=None, response_format=None):
        self.calls.append(("chat", messages))
        return "plain text"

    def chat_json(self, messages, temperature=0.3, max_tokens=None, max_attempts=1, json_schema=None):
        self.calls.append(("chat_json", messages, json_schema))
        return self.payload


def test_bridge_without_tools_returns_text():
    bridge = OasisLLMBridge.__new__(OasisLLMBridge)
    bridge.provider_name = "fake"
    bridge.llm = FakeLLM({})
    bridge.model_name = "fake"
    completion = bridge.complete([{"role": "user", "content": "hi"}])
    choice = completion.choices[0] if hasattr(completion, "choices") else completion["choices"][0]
    message = choice.message if hasattr(choice, "message") else choice["message"]
    content = message.content if hasattr(message, "content") else message["content"]
    assert content == "plain text"


def test_bridge_with_tools_emits_tool_calls():
    bridge = OasisLLMBridge.__new__(OasisLLMBridge)
    bridge.provider_name = "fake"
    bridge.llm = FakeLLM({"name": "CREATE_POST", "arguments": {"content": "hello"}})
    bridge.model_name = "fake"
    tools = [
        {
            "type": "function",
            "function": {
                "name": "CREATE_POST",
                "parameters": {
                    "type": "object",
                    "properties": {"content": {"type": "string"}},
                    "required": ["content"],
                },
            },
        },
        {
            "type": "function",
            "function": {"name": "DO_NOTHING", "parameters": {"type": "object", "properties": {}}},
        },
    ]
    completion = bridge.complete(
        [{"role": "user", "content": "act"}],
        tools=tools,
    )
    choice = completion.choices[0] if hasattr(completion, "choices") else completion["choices"][0]
    message = choice.message if hasattr(choice, "message") else choice["message"]
    tool_calls = message.tool_calls if hasattr(message, "tool_calls") else message["tool_calls"]
    assert tool_calls
    fn = tool_calls[0].function if hasattr(tool_calls[0], "function") else tool_calls[0]["function"]
    name = fn.name if hasattr(fn, "name") else fn["name"]
    assert name == "CREATE_POST"
    # Ensure action instruction was injected
    assert bridge.llm.calls[0][0] == "chat_json"
    assert any("CREATE_POST" in m["content"] for m in bridge.llm.calls[0][1] if m["role"] == "system")


def test_bridge_invalid_action_falls_back_to_do_nothing():
    bridge = OasisLLMBridge.__new__(OasisLLMBridge)
    bridge.provider_name = "fake"
    bridge.llm = FakeLLM({"name": "NOT_A_TOOL", "arguments": {}})
    bridge.model_name = "fake"
    tools = [
        {"type": "function", "function": {"name": "DO_NOTHING", "parameters": {"type": "object"}}},
        {"type": "function", "function": {"name": "LIKE_POST", "parameters": {"type": "object"}}},
    ]
    completion = bridge.complete([{"role": "user", "content": "x"}], tools=tools)
    choice = completion.choices[0] if hasattr(completion, "choices") else completion["choices"][0]
    message = choice.message if hasattr(choice, "message") else choice["message"]
    tool_calls = message.tool_calls if hasattr(message, "tool_calls") else message["tool_calls"]
    fn = tool_calls[0].function if hasattr(tool_calls[0], "function") else tool_calls[0]["function"]
    name = fn.name if hasattr(fn, "name") else fn["name"]
    assert name == "DO_NOTHING"


def test_build_chat_completion_tool_calls_finish_reason():
    completion = build_chat_completion(
        model="m",
        messages=[],
        tool_calls=[{"id": "1", "type": "function", "function": {"name": "DO_NOTHING", "arguments": "{}"}}],
    )
    choice = completion.choices[0] if hasattr(completion, "choices") else completion["choices"][0]
    finish = choice.finish_reason if hasattr(choice, "finish_reason") else choice["finish_reason"]
    assert finish == "tool_calls"
