"""
Unit tests for Record/Replay utilities (Phase 16 — Testing Utilities).
"""

import json
import tempfile
from pathlib import Path

import pytest

from curio_agent_sdk.models.llm import LLMRequest, LLMResponse, Message, TokenUsage
from curio_agent_sdk.testing.mock_llm import text_response
from curio_agent_sdk.testing.replay import (
    Recording,
    RecordingMiddleware,
    ReplayLLMClient,
    LLMCallRecord,
    ToolCallRecord,
)


# ---------------------------------------------------------------------------
# RecordingMiddleware
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_recording_middleware():
    """Records LLM calls."""
    middleware = RecordingMiddleware()
    req = LLMRequest(messages=[Message.user("Hello")])
    resp = text_response("Hi there")
    out = await middleware.after_llm_call(req, resp)
    assert out == resp
    assert len(middleware.recording.llm_calls) == 1
    assert middleware.recording.llm_calls[0].response.get("message", {}).get("text") == "Hi there"


@pytest.mark.unit
def test_recording_save():
    """Save to file."""
    rec = Recording()
    rec.llm_calls.append(
        LLMCallRecord(request={"messages": []}, response={"message": {"text": "saved"}})
    )
    middleware = RecordingMiddleware()
    middleware.recording = rec
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        middleware.save(path)
        data = json.loads(Path(path).read_text())
        assert "llm_calls" in data
        assert len(data["llm_calls"]) == 1
        assert data["llm_calls"][0]["response"]["message"]["text"] == "saved"
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.unit
def test_replay_client_from_file():
    """Load recording from file."""
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        data = {
            "llm_calls": [
                {
                    "request": {"messages": []},
                    "response": {
                        "message": {"role": "assistant", "content": "Replied", "tool_calls": []},
                        "usage": {"input_tokens": 0, "output_tokens": 0},
                        "model": "mock",
                        "provider": "mock",
                        "finish_reason": "stop",
                    },
                }
            ],
            "tool_calls": [],
        }
        Path(path).write_text(json.dumps(data))
        client = ReplayLLMClient.from_file(path)
        assert len(client._responses) == 1
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_replay_client_call():
    """Replay matches recording."""
    responses = [text_response("First"), text_response("Second")]
    client = ReplayLLMClient(responses)
    req = LLMRequest(messages=[Message.user("Hi")])
    r1 = await client.call(req)
    r2 = await client.call(req)
    assert r1.message.text == "First"
    assert r2.message.text == "Second"
    assert client.call_count == 2


@pytest.mark.unit
@pytest.mark.asyncio
async def test_roundtrip():
    """Record → save → load → replay."""
    middleware = RecordingMiddleware()
    req = LLMRequest(messages=[Message.user("Roundtrip")])
    resp = text_response("Roundtrip reply")
    await middleware.after_llm_call(req, resp)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        middleware.save(path)
        client = ReplayLLMClient.from_file(path)
        out = await client.call(LLMRequest(messages=[Message.user("Any")]))
        assert "Roundtrip reply" in (out.message.text or "")
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.unit
@pytest.mark.asyncio
async def test_recording_recorded_output():
    """recorded_output returns last assistant text from recording."""
    middleware = RecordingMiddleware()
    req = LLMRequest(messages=[Message.user("Hi")])
    await middleware.after_llm_call(req, text_response("Last reply"))
    assert middleware.recorded_output == "Last reply"
    await middleware.after_llm_call(req, text_response("New last"))
    assert middleware.recorded_output == "New last"
