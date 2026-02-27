"""
Record/replay testing utilities for Curio Agent SDK.

Provides:
- RecordingMiddleware: middleware that records LLM requests/responses and tool calls.
- ReplayLLMClient: LLM client that replays recorded responses deterministically.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Dict, List

from curio_agent_sdk.middleware.base import Middleware
from curio_agent_sdk.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
    TokenUsage,
    ToolCall,
)


@dataclass
class LLMCallRecord:
    """Single LLM call record with request and response payloads."""

    request: Dict[str, Any]
    response: Dict[str, Any]


@dataclass
class ToolCallRecord:
    """Single tool call record."""

    name: str
    args: Dict[str, Any]
    result: Any | None = None
    error: str | None = None


@dataclass
class Recording:
    """Complete recording of an agent run."""

    llm_calls: List[LLMCallRecord] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "llm_calls": [r.__dict__ for r in self.llm_calls],
            "tool_calls": [r.__dict__ for r in self.tool_calls],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Recording:
        llm_calls = [
            LLMCallRecord(request=rc["request"], response=rc["response"])
            for rc in data.get("llm_calls", [])
        ]
        tool_calls = [
            ToolCallRecord(
                name=tc["name"],
                args=tc.get("args", {}),
                result=tc.get("result"),
                error=tc.get("error"),
            )
            for tc in data.get("tool_calls", [])
        ]
        return cls(llm_calls=llm_calls, tool_calls=tool_calls)


class RecordingMiddleware(Middleware):
    """
    Middleware that records LLM requests/responses and tool calls.

    Intended for golden-file and regression-style testing:

        recorder = RecordingMiddleware()
        agent = Agent.builder().middleware(recorder).model("anthropic:claude-sonnet-4-6").build()
        await agent.arun("Analyze this code")
        recorder.save("tests/fixtures/analyze_code.json")
    """

    def __init__(self) -> None:
        self.recording = Recording()

    # ---- LLM hooks -------------------------------------------------

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        # We snapshot the request in after_llm_call where we also have the response.
        return request

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        req_payload = _request_to_dict(request)
        resp_payload = _response_to_dict(response)
        self.recording.llm_calls.append(LLMCallRecord(request=req_payload, response=resp_payload))
        return response

    # ---- Tool hooks ------------------------------------------------

    async def before_tool_call(self, tool_name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        # We record full tool call (including result) in after_tool_call.
        return tool_name, args

    async def after_tool_call(self, tool_name: str, args: dict[str, Any], result: Any) -> Any:
        self.recording.tool_calls.append(
            ToolCallRecord(name=tool_name, args=dict(args), result=result, error=None)
        )
        return result

    # ---- Persistence helpers ---------------------------------------

    def save(self, path: str | Path) -> None:
        """Save the current recording to a JSON file."""
        Path(path).write_text(json.dumps(self.recording.to_dict(), indent=2, default=str))

    @property
    def recorded_output(self) -> str:
        """
        Best-effort last assistant output captured in the recording.

        Useful in tests for comparing replayed runs:
            assert result.output == recorder.recorded_output
        """
        for call in reversed(self.recording.llm_calls):
            msg = call["response"]["message"] if isinstance(call, dict) else call.response.get("message")
            # When using LLMCallRecord, response is a dict
        # Fallback: derive from structured payloads
        for rc in reversed(self.recording.llm_calls):
            text = rc.response.get("message", {}).get("text")
            if text:
                return text
        return ""


class ReplayLLMClient:
    """
    LLM client that replays recorded responses deterministically.

    Use with recordings produced by RecordingMiddleware:

        replay_client = ReplayLLMClient.from_file("tests/fixtures/analyze_code.json")
        harness = AgentTestHarness(agent)
        harness.set_llm(replay_client)
        result = await harness.run("Analyze this code")
    """

    def __init__(self, responses: List[LLMResponse]) -> None:
        self._responses = list(responses)
        self._index = 0
        self.call_count: int = 0

    @classmethod
    def from_file(cls, path: str | Path) -> ReplayLLMClient:
        data = json.loads(Path(path).read_text())
        recording = Recording.from_dict(data)
        responses: List[LLMResponse] = [
            _response_from_dict(rc.response) for rc in recording.llm_calls
        ]
        return cls(responses)

    async def call(
        self,
        request: LLMRequest,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> LLMResponse:
        """Return the next pre-recorded response."""
        if self._index >= len(self._responses):
            raise RuntimeError("ReplayLLMClient exhausted: no more recorded responses.")
        resp = self._responses[self._index]
        self._index += 1
        self.call_count += 1
        return resp

    async def stream(
        self,
        request: LLMRequest,
        run_id: str | None = None,
        agent_id: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        """
        Stream chunks derived from the recorded full response.
        """
        response = await self.call(request, run_id=run_id, agent_id=agent_id)
        text = response.message.text
        if text:
            for i in range(0, len(text), 10):
                yield LLMStreamChunk(type="text_delta", text=text[i : i + 10])
        yield LLMStreamChunk(
            type="done",
            finish_reason=response.finish_reason,
            usage=response.usage,
        )


# -------------------------------------------------------------------
# Serialization helpers
# -------------------------------------------------------------------


def _tool_call_to_dict(tc: ToolCall) -> Dict[str, Any]:
    return {"id": tc.id, "name": tc.name, "arguments": tc.arguments}


def _message_to_dict(msg: Message) -> Dict[str, Any]:
    return {
        "role": msg.role,
        "content": msg.content,
        "tool_calls": [_tool_call_to_dict(tc) for tc in (msg.tool_calls or [])],
        "tool_call_id": msg.tool_call_id,
        "name": msg.name,
        "text": msg.text,
    }


def _usage_to_dict(usage: TokenUsage) -> Dict[str, Any]:
    return {
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "cache_read_tokens": usage.cache_read_tokens,
        "cache_write_tokens": usage.cache_write_tokens,
    }


def _request_to_dict(req: LLMRequest) -> Dict[str, Any]:
    return {
        "messages": [_message_to_dict(m) for m in req.messages],
        "model": req.model,
        "provider": req.provider,
        "tier": req.tier,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
        "tool_choice": req.tool_choice,
        "metadata": req.metadata,
    }


def _response_to_dict(resp: LLMResponse) -> Dict[str, Any]:
    return {
        "message": _message_to_dict(resp.message),
        "usage": _usage_to_dict(resp.usage),
        "model": resp.model,
        "provider": resp.provider,
        "finish_reason": resp.finish_reason,
        "latency_ms": resp.latency_ms,
        "error": resp.error,
    }


def _tool_call_from_dict(data: Dict[str, Any]) -> ToolCall:
    return ToolCall(
        id=data["id"],
        name=data["name"],
        arguments=data.get("arguments", {}),
    )


def _message_from_dict(data: Dict[str, Any]) -> Message:
    tool_calls = [ _tool_call_from_dict(tc) for tc in data.get("tool_calls", []) ]
    return Message(
        role=data["role"],
        content=data.get("content"),
        tool_calls=tool_calls or None,
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
    )


def _usage_from_dict(data: Dict[str, Any]) -> TokenUsage:
    return TokenUsage(
        input_tokens=data.get("input_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
        cache_read_tokens=data.get("cache_read_tokens", 0),
        cache_write_tokens=data.get("cache_write_tokens", 0),
        output_tokens=data.get("output_tokens", 0),
    )


def _response_from_dict(data: Dict[str, Any]) -> LLMResponse:
    message = _message_from_dict(data["message"])
    usage = _usage_from_dict(data["usage"])
    return LLMResponse(
        message=message,
        usage=usage,
        model=data.get("model", "") or "",
        provider=data.get("provider", "") or "",
        finish_reason=data.get("finish_reason", "stop"),
        latency_ms=data.get("latency_ms", 0),
        error=data.get("error"),
    )

