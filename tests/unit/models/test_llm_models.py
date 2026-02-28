"""
Unit tests for curio_agent_sdk.models.llm

Covers: Message, ToolCall, ContentBlock, ToolSchema,
        TokenUsage, LLMRequest, LLMResponse, LLMStreamChunk
"""

import pytest

from curio_agent_sdk.models.llm import (
    ContentBlock,
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
    TokenUsage,
    ToolCall,
    ToolSchema,
)


# ===================================================================
# ToolCall
# ===================================================================


class TestToolCall:
    def test_creation(self):
        tc = ToolCall(id="call_1", name="search", arguments={"q": "hello"})
        assert tc.id == "call_1"
        assert tc.name == "search"
        assert tc.arguments == {"q": "hello"}

    def test_with_complex_args(self):
        args = {"filters": [{"field": "age", "op": ">", "value": 30}], "nested": {"a": 1}}
        tc = ToolCall(id="c2", name="query", arguments=args)
        assert tc.arguments["filters"][0]["field"] == "age"
        assert tc.arguments["nested"]["a"] == 1

    def test_empty_arguments(self):
        tc = ToolCall(id="c3", name="noop", arguments={})
        assert tc.arguments == {}


# ===================================================================
# ContentBlock
# ===================================================================


class TestContentBlock:
    def test_text_block(self):
        block = ContentBlock(type="text", text="hello world")
        assert block.type == "text"
        assert block.text == "hello world"

    def test_image_url_block(self):
        block = ContentBlock(type="image_url", image_url="https://example.com/img.png")
        assert block.type == "image_url"
        assert block.image_url == "https://example.com/img.png"

    def test_tool_use_block(self):
        tc = ToolCall(id="t1", name="calc", arguments={"x": 1})
        block = ContentBlock(type="tool_use", tool_call=tc)
        assert block.tool_call is not None
        assert block.tool_call.name == "calc"

    def test_tool_result_block(self):
        block = ContentBlock(type="tool_result", tool_call_id="t1")
        assert block.tool_call_id == "t1"

    def test_defaults(self):
        block = ContentBlock(type="text")
        assert block.text is None
        assert block.image_url is None
        assert block.tool_call is None
        assert block.tool_call_id is None


# ===================================================================
# Message
# ===================================================================


class TestMessage:
    def test_creation_user(self):
        msg = Message(role="user", content="hello")
        assert msg.role == "user"
        assert msg.content == "hello"

    def test_creation_assistant(self):
        msg = Message(role="assistant", content="hi there")
        assert msg.role == "assistant"
        assert msg.content == "hi there"

    def test_creation_system(self):
        msg = Message(role="system", content="You are helpful.")
        assert msg.role == "system"

    def test_creation_tool(self):
        msg = Message(role="tool", content="result", tool_call_id="call_1")
        assert msg.role == "tool"
        assert msg.tool_call_id == "call_1"

    def test_with_tool_calls(self):
        tc = ToolCall(id="c1", name="search", arguments={"q": "test"})
        msg = Message(role="assistant", content="Let me search.", tool_calls=[tc])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1
        assert msg.tool_calls[0].name == "search"

    def test_with_content_blocks(self):
        blocks = [
            ContentBlock(type="text", text="first"),
            ContentBlock(type="text", text="second"),
        ]
        msg = Message(role="assistant", content=blocks)
        assert isinstance(msg.content, list)
        assert len(msg.content) == 2

    def test_defaults(self):
        msg = Message(role="user")
        assert msg.content is None
        assert msg.tool_calls is None
        assert msg.tool_call_id is None
        assert msg.name is None

    def test_with_name(self):
        msg = Message(role="user", content="hi", name="alice")
        assert msg.name == "alice"

    # ---- text property ----

    def test_text_property_string_content(self):
        msg = Message(role="user", content="hello")
        assert msg.text == "hello"

    def test_text_property_none_content(self):
        msg = Message(role="user", content=None)
        assert msg.text == ""

    def test_text_property_content_blocks(self):
        blocks = [
            ContentBlock(type="text", text="part1"),
            ContentBlock(type="image_url", image_url="http://x.com/i.png"),
            ContentBlock(type="text", text="part2"),
        ]
        msg = Message(role="assistant", content=blocks)
        assert msg.text == "part1\npart2"

    def test_text_property_empty_blocks(self):
        msg = Message(role="assistant", content=[])
        assert msg.text == ""

    def test_text_property_blocks_with_none_text(self):
        blocks = [ContentBlock(type="text", text=None)]
        msg = Message(role="assistant", content=blocks)
        assert msg.text == ""

    # ---- static factory methods ----

    def test_system_factory(self):
        msg = Message.system("sys prompt")
        assert msg.role == "system"
        assert msg.content == "sys prompt"

    def test_user_factory(self):
        msg = Message.user("question")
        assert msg.role == "user"
        assert msg.content == "question"

    def test_assistant_factory(self):
        msg = Message.assistant("answer")
        assert msg.role == "assistant"
        assert msg.content == "answer"
        assert msg.tool_calls is None

    def test_assistant_factory_with_tool_calls(self):
        tc = ToolCall(id="c1", name="fn", arguments={})
        msg = Message.assistant("text", tool_calls=[tc])
        assert msg.tool_calls is not None
        assert len(msg.tool_calls) == 1

    def test_tool_result_factory(self):
        msg = Message.tool_result("call_1", "42")
        assert msg.role == "tool"
        assert msg.content == "42"
        assert msg.tool_call_id == "call_1"
        assert msg.name is None

    def test_tool_result_factory_with_name(self):
        msg = Message.tool_result("call_1", "42", name="calculator")
        assert msg.name == "calculator"


# ===================================================================
# ToolSchema
# ===================================================================


class TestToolSchema:
    @pytest.fixture
    def schema(self):
        return ToolSchema(
            name="search",
            description="Search the web",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                },
                "required": ["query"],
            },
        )

    def test_creation(self, schema):
        assert schema.name == "search"
        assert schema.description == "Search the web"
        assert "query" in schema.parameters["properties"]

    def test_to_openai_format(self, schema):
        fmt = schema.to_openai_format()
        assert fmt["type"] == "function"
        assert fmt["function"]["name"] == "search"
        assert fmt["function"]["description"] == "Search the web"
        assert fmt["function"]["parameters"] == schema.parameters

    def test_to_anthropic_format(self, schema):
        fmt = schema.to_anthropic_format()
        assert fmt["name"] == "search"
        assert fmt["description"] == "Search the web"
        assert fmt["input_schema"] == schema.parameters

    def test_minimal_schema(self):
        s = ToolSchema(name="ping", description="Ping", parameters={"type": "object"})
        assert s.to_openai_format()["function"]["name"] == "ping"
        assert s.to_anthropic_format()["name"] == "ping"


# ===================================================================
# TokenUsage
# ===================================================================


class TestTokenUsage:
    def test_defaults(self):
        usage = TokenUsage()
        assert usage.input_tokens == 0
        assert usage.output_tokens == 0
        assert usage.cache_read_tokens == 0
        assert usage.cache_write_tokens == 0

    def test_total_tokens(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        assert usage.total_tokens == 150

    def test_with_cache(self):
        usage = TokenUsage(
            input_tokens=100,
            output_tokens=50,
            cache_read_tokens=20,
            cache_write_tokens=10,
        )
        assert usage.cache_read_tokens == 20
        assert usage.cache_write_tokens == 10
        assert usage.total_tokens == 150  # total is input + output only

    def test_zero_total(self):
        usage = TokenUsage()
        assert usage.total_tokens == 0


# ===================================================================
# LLMRequest
# ===================================================================


class TestLLMRequest:
    def test_minimal(self):
        req = LLMRequest(messages=[Message.user("hi")])
        assert len(req.messages) == 1
        assert req.tools is None
        assert req.tool_choice is None
        assert req.max_tokens == 4096
        assert req.temperature == 0.7
        assert req.stream is False
        assert req.response_format is None
        assert req.stop is None
        assert req.model is None
        assert req.provider is None
        assert req.tier is None
        assert req.metadata == {}
        assert req.prompt_cache is False
        assert req.prompt_cache_key is None

    def test_full(self):
        schema = ToolSchema(name="t", description="d", parameters={"type": "object"})
        req = LLMRequest(
            messages=[Message.system("sys"), Message.user("q")],
            tools=[schema],
            tool_choice="auto",
            max_tokens=2048,
            temperature=0.0,
            stream=True,
            response_format={"type": "json_object"},
            stop=["END"],
            model="gpt-4o",
            provider="openai",
            tier="tier1",
            metadata={"session": "abc"},
            prompt_cache=True,
            prompt_cache_key="key_1",
        )
        assert len(req.messages) == 2
        assert len(req.tools) == 1
        assert req.tool_choice == "auto"
        assert req.max_tokens == 2048
        assert req.temperature == 0.0
        assert req.stream is True
        assert req.response_format == {"type": "json_object"}
        assert req.stop == ["END"]
        assert req.model == "gpt-4o"
        assert req.provider == "openai"
        assert req.tier == "tier1"
        assert req.metadata["session"] == "abc"
        assert req.prompt_cache is True
        assert req.prompt_cache_key == "key_1"

    def test_defaults(self):
        req = LLMRequest(messages=[])
        assert req.messages == []
        assert req.metadata == {}

    def test_with_response_format(self):
        req = LLMRequest(
            messages=[Message.user("q")],
            response_format={"type": "json_object"},
        )
        assert req.response_format["type"] == "json_object"

    def test_with_metadata(self):
        req = LLMRequest(
            messages=[Message.user("q")],
            metadata={"user_id": "u1", "trace": True},
        )
        assert req.metadata["user_id"] == "u1"
        assert req.metadata["trace"] is True


# ===================================================================
# LLMResponse
# ===================================================================


class TestLLMResponse:
    def test_creation(self):
        resp = LLMResponse(
            message=Message.assistant("answer"),
            usage=TokenUsage(input_tokens=10, output_tokens=5),
            model="gpt-4o",
            provider="openai",
            finish_reason="stop",
        )
        assert resp.model == "gpt-4o"
        assert resp.provider == "openai"
        assert resp.finish_reason == "stop"
        assert resp.latency_ms == 0
        assert resp.raw_response is None
        assert resp.error is None

    def test_error_response(self):
        resp = LLMResponse(
            message=Message.assistant(""),
            usage=TokenUsage(),
            model="gpt-4o",
            provider="openai",
            finish_reason="error",
            error="Server error",
        )
        assert resp.error == "Server error"
        assert resp.finish_reason == "error"

    def test_content_property(self):
        resp = LLMResponse(
            message=Message.assistant("hello"),
            usage=TokenUsage(),
            model="m",
            provider="p",
            finish_reason="stop",
        )
        assert resp.content == "hello"

    def test_content_property_empty(self):
        resp = LLMResponse(
            message=Message.assistant(""),
            usage=TokenUsage(),
            model="m",
            provider="p",
            finish_reason="stop",
        )
        assert resp.content == ""

    def test_tool_calls_property(self):
        tc = ToolCall(id="c1", name="fn", arguments={"a": 1})
        resp = LLMResponse(
            message=Message.assistant("", tool_calls=[tc]),
            usage=TokenUsage(),
            model="m",
            provider="p",
            finish_reason="tool_use",
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "fn"

    def test_tool_calls_property_empty(self):
        resp = LLMResponse(
            message=Message.assistant("hi"),
            usage=TokenUsage(),
            model="m",
            provider="p",
            finish_reason="stop",
        )
        assert resp.tool_calls == []

    def test_has_tool_calls(self):
        tc = ToolCall(id="c1", name="fn", arguments={})
        resp = LLMResponse(
            message=Message.assistant("", tool_calls=[tc]),
            usage=TokenUsage(),
            model="m",
            provider="p",
            finish_reason="tool_use",
        )
        assert resp.has_tool_calls is True

    def test_has_tool_calls_false(self):
        resp = LLMResponse(
            message=Message.assistant("hi"),
            usage=TokenUsage(),
            model="m",
            provider="p",
            finish_reason="stop",
        )
        assert resp.has_tool_calls is False

    def test_finish_reasons(self):
        for reason in ("stop", "tool_use", "length", "error"):
            resp = LLMResponse(
                message=Message.assistant(""),
                usage=TokenUsage(),
                model="m",
                provider="p",
                finish_reason=reason,
            )
            assert resp.finish_reason == reason

    def test_with_latency(self):
        resp = LLMResponse(
            message=Message.assistant(""),
            usage=TokenUsage(),
            model="m",
            provider="p",
            finish_reason="stop",
            latency_ms=250,
        )
        assert resp.latency_ms == 250

    def test_with_raw_response(self):
        raw = {"id": "chatcmpl-xxx", "choices": []}
        resp = LLMResponse(
            message=Message.assistant(""),
            usage=TokenUsage(),
            model="m",
            provider="p",
            finish_reason="stop",
            raw_response=raw,
        )
        assert resp.raw_response == raw


# ===================================================================
# LLMStreamChunk
# ===================================================================


class TestLLMStreamChunk:
    def test_text_delta(self):
        chunk = LLMStreamChunk(type="text_delta", text="Hello")
        assert chunk.type == "text_delta"
        assert chunk.text == "Hello"

    def test_tool_call_start(self):
        tc = ToolCall(id="c1", name="fn", arguments={})
        chunk = LLMStreamChunk(type="tool_call_start", tool_call=tc)
        assert chunk.tool_call is not None
        assert chunk.tool_call.name == "fn"

    def test_tool_call_delta(self):
        chunk = LLMStreamChunk(
            type="tool_call_delta",
            tool_call_id="c1",
            argument_delta='{"key": "val',
        )
        assert chunk.argument_delta == '{"key": "val'

    def test_tool_call_end(self):
        chunk = LLMStreamChunk(type="tool_call_end", tool_call_id="c1")
        assert chunk.type == "tool_call_end"

    def test_usage_chunk(self):
        usage = TokenUsage(input_tokens=10, output_tokens=5)
        chunk = LLMStreamChunk(type="usage", usage=usage)
        assert chunk.usage is not None
        assert chunk.usage.total_tokens == 15

    def test_done_chunk(self):
        chunk = LLMStreamChunk(type="done", finish_reason="stop")
        assert chunk.type == "done"
        assert chunk.finish_reason == "stop"

    def test_defaults(self):
        chunk = LLMStreamChunk(type="text_delta")
        assert chunk.text is None
        assert chunk.tool_call is None
        assert chunk.tool_call_id is None
        assert chunk.argument_delta is None
        assert chunk.usage is None
        assert chunk.finish_reason is None
