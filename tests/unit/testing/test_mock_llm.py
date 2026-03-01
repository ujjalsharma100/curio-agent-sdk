"""
Unit tests for MockLLM and helpers (Phase 16 — Testing Utilities).
"""

import pytest

from curio_agent_sdk.models.llm import LLMRequest, Message
from curio_agent_sdk.testing.mock_llm import MockLLM, text_response, tool_call_response


# ---------------------------------------------------------------------------
# MockLLM creation and responses
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_mock_llm_creation():
    """Empty MockLLM can be created."""
    mock = MockLLM()
    assert mock.call_count == 0
    assert mock.calls == []


@pytest.mark.unit
def test_add_text_response():
    """Add and retrieve text response."""
    mock = MockLLM()
    mock.add_text_response("Hello, world!")
    assert len(mock._responses) == 1
    resp = mock._responses[0]
    assert resp.message.text == "Hello, world!"
    assert resp.finish_reason == "stop"
    assert resp.model == "mock-model"


@pytest.mark.unit
def test_add_tool_call_response():
    """Add tool call response."""
    mock = MockLLM()
    mock.add_tool_call_response("search", {"q": "test"})
    assert len(mock._responses) == 1
    resp = mock._responses[0]
    assert resp.finish_reason == "tool_use"
    assert len(resp.message.tool_calls) == 1
    assert resp.message.tool_calls[0].name == "search"
    assert resp.message.tool_calls[0].arguments == {"q": "test"}


@pytest.mark.unit
def test_add_multiple_responses():
    """Queue multiple responses."""
    mock = MockLLM()
    mock.add_text_response("First")
    mock.add_tool_call_response("tool_a", {})
    mock.add_text_response("Second")
    assert len(mock._responses) == 3


@pytest.mark.unit
@pytest.mark.asyncio
async def test_response_ordering():
    """FIFO ordering of responses."""
    mock = MockLLM()
    mock.add_text_response("A")
    mock.add_text_response("B")
    mock.add_text_response("C")
    req = LLMRequest(messages=[Message.user("Hi")])
    r1 = await mock.call(req)
    r2 = await mock.call(req)
    r3 = await mock.call(req)
    assert r1.message.text == "A"
    assert r2.message.text == "B"
    assert r3.message.text == "C"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_call_count():
    """Call count tracking."""
    mock = MockLLM()
    mock.add_text_response("X")
    mock.add_text_response("Y")
    req = LLMRequest(messages=[Message.user("Hi")])
    await mock.call(req)
    await mock.call(req)
    assert mock.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_calls_history():
    """Request history is recorded."""
    mock = MockLLM()
    mock.add_text_response("Ok")
    req1 = LLMRequest(messages=[Message.user("One")])
    req2 = LLMRequest(messages=[Message.user("Two")])
    await mock.call(req1)
    await mock.call(req2)
    assert len(mock.calls) == 2
    assert mock.calls[0].messages[0].content == "One"
    assert mock.calls[1].messages[0].content == "Two"


@pytest.mark.unit
def test_text_response_helper():
    """text_response() function creates valid LLMResponse."""
    resp = text_response("Hello!", model="custom")
    assert resp.message.text == "Hello!"
    assert resp.model == "custom"
    assert resp.finish_reason == "stop"


@pytest.mark.unit
def test_tool_call_response_helper():
    """tool_call_response() function creates valid LLMResponse."""
    resp = tool_call_response("greet", {"name": "Alice"})
    assert resp.finish_reason == "tool_use"
    assert len(resp.message.tool_calls) == 1
    assert resp.message.tool_calls[0].name == "greet"
    assert resp.message.tool_calls[0].arguments == {"name": "Alice"}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_exhausted_responses():
    """When no more queued responses, call returns default text response."""
    mock = MockLLM()
    mock.add_text_response("Only one")
    req = LLMRequest(messages=[Message.user("Hi")])
    r1 = await mock.call(req)
    assert r1.message.text == "Only one"
    r2 = await mock.call(req)
    assert r2.message.text == "I'm done."
    assert r2.finish_reason == "stop"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mock_llm_stream():
    """stream() yields text_delta chunks then done."""
    mock = MockLLM()
    mock.add_text_response("Hi")
    req = LLMRequest(messages=[Message.user("Go")])
    chunks = [c async for c in mock.stream(req)]
    assert len(chunks) >= 2
    text_chunks = [c for c in chunks if getattr(c, "type", None) == "text_delta"]
    assert any(getattr(c, "text", "") for c in text_chunks)
    assert chunks[-1].type == "done"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_mock_llm_request_messages():
    """request_messages returns message lists from recorded calls."""
    mock = MockLLM()
    mock.add_text_response("Ok")
    await mock.call(LLMRequest(messages=[Message.user("A"), Message.system("S")]))
    await mock.call(LLMRequest(messages=[Message.user("B")]))
    assert len(mock.request_messages) == 2
    assert len(mock.request_messages[0]) == 2
    assert mock.request_messages[1][0].content == "B"


@pytest.mark.unit
def test_add_tool_call_response_with_tool_call_list():
    """add_tool_call_response accepts list of ToolCall (legacy form)."""
    from curio_agent_sdk.models.llm import ToolCall
    mock = MockLLM()
    tc = ToolCall(id="id1", name="foo", arguments={"x": 1})
    mock.add_tool_call_response([tc])
    assert len(mock._responses) == 1
    assert mock._responses[0].message.tool_calls[0].id == "id1"
    assert mock._responses[0].message.tool_calls[0].name == "foo"
