"""
OpenAI provider with async, native tool calling, and streaming support.

Supports GPT-4o, GPT-4o-mini, o1, o3, and any OpenAI-compatible API.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

import httpx

from curio_agent_sdk.llm.providers.base import LLMProvider
from curio_agent_sdk.models.llm import (
    LLMRequest,
    LLMResponse,
    LLMStreamChunk,
    Message,
    ToolCall,
    TokenUsage,
)
from curio_agent_sdk.models.exceptions import (
    LLMRateLimitError,
    LLMAuthenticationError,
    LLMProviderError,
    LLMTimeoutError,
)

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError, APITimeoutError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


class OpenAIProvider(LLMProvider):
    """
    OpenAI provider with full support for:
    - Native tool/function calling
    - Streaming responses
    - Async operations
    - Per-request API key (thread-safe)
    - OpenAI-compatible APIs (vLLM, Together, etc.)
    """

    provider_name = "openai"

    def __init__(self) -> None:
        # Shared HTTP client for connection pooling across OpenAI requests.
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        """
        Lazily create a shared httpx.AsyncClient with connection pooling.

        This client is reused across all OpenAI requests made by this provider
        instance, while API keys/base URLs remain per-request on the AsyncOpenAI client.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._http_client

    def _get_client(self, api_key: str | None = None, base_url: str | None = None) -> AsyncOpenAI:
        """Create a client for this specific request using a pooled HTTP client."""
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package not installed. Install with: pip install openai")
        kwargs = {"http_client": self._get_http_client()}
        if api_key:
            kwargs["api_key"] = api_key
        if base_url:
            kwargs["base_url"] = base_url
        return AsyncOpenAI(**kwargs)

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        """Convert our Message objects to OpenAI's format."""
        messages = []
        for msg in request.messages:
            m: dict = {"role": msg.role}

            if msg.role == "tool":
                m["content"] = msg.content if isinstance(msg.content, str) else ""
                m["tool_call_id"] = msg.tool_call_id or ""
                if msg.name:
                    m["name"] = msg.name
            elif msg.role == "assistant" and msg.tool_calls:
                m["content"] = msg.content if isinstance(msg.content, str) else ""
                m["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments) if isinstance(tc.arguments, dict) else tc.arguments,
                        },
                    }
                    for tc in msg.tool_calls
                ]
            else:
                m["content"] = msg.content if isinstance(msg.content, str) else ""

            messages.append(m)
        return messages

    def _build_tools(self, request: LLMRequest) -> list[dict] | None:
        """Convert our ToolSchema objects to OpenAI's format."""
        if not request.tools:
            return None
        return [t.to_openai_format() for t in request.tools]

    def _parse_response(self, response, provider: str, model: str, latency_ms: int) -> LLMResponse:
        """Parse OpenAI response into our LLMResponse."""
        choice = response.choices[0]
        msg = choice.message

        # Parse tool calls
        tool_calls = None
        if msg.tool_calls:
            tool_calls = []
            for tc in msg.tool_calls:
                try:
                    args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                except json.JSONDecodeError:
                    args = {"_raw": tc.function.arguments}
                tool_calls.append(ToolCall(
                    id=tc.id,
                    name=tc.function.name,
                    arguments=args,
                ))

        # Determine finish reason
        finish_reason = choice.finish_reason or "stop"
        if finish_reason == "tool_calls":
            finish_reason = "tool_use"

        # Parse usage
        usage = TokenUsage()
        if response.usage:
            usage = TokenUsage(
                input_tokens=response.usage.prompt_tokens or 0,
                output_tokens=response.usage.completion_tokens or 0,
            )

        return LLMResponse(
            message=Message(
                role="assistant",
                content=msg.content or "",
                tool_calls=tool_calls,
            ),
            usage=usage,
            model=response.model or model,
            provider=provider,
            finish_reason=finish_reason,
            latency_ms=latency_ms,
            raw_response=response,
        )

    async def call(
        self,
        request: LLMRequest,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> LLMResponse:
        client = self._get_client(api_key, base_url)
        model = request.model or "gpt-4o-mini"
        start = time.monotonic()

        try:
            params: dict = {
                "model": model,
                "messages": self._build_messages(request),
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }

            tools = self._build_tools(request)
            if tools:
                params["tools"] = tools
                if request.tool_choice:
                    if isinstance(request.tool_choice, str):
                        params["tool_choice"] = request.tool_choice
                    elif isinstance(request.tool_choice, dict) and "name" in request.tool_choice:
                        params["tool_choice"] = {
                            "type": "function",
                            "function": {"name": request.tool_choice["name"]},
                        }

            if request.response_format:
                params["response_format"] = request.response_format
            if request.stop:
                params["stop"] = request.stop

            response = await client.chat.completions.create(**params)
            latency_ms = int((time.monotonic() - start) * 1000)

            return self._parse_response(response, self.provider_name, model, latency_ms)

        except RateLimitError as e:
            raise LLMRateLimitError(self.provider_name, model) from e
        except AuthenticationError as e:
            raise LLMAuthenticationError(str(e), self.provider_name, model) from e
        except APITimeoutError as e:
            raise LLMTimeoutError(str(e), self.provider_name, model) from e
        except APIError as e:
            status = getattr(e, "status_code", None)
            raise LLMProviderError(str(e), self.provider_name, model, status) from e

    async def stream(
        self,
        request: LLMRequest,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        client = self._get_client(api_key, base_url)
        model = request.model or "gpt-4o-mini"

        try:
            params: dict = {
                "model": model,
                "messages": self._build_messages(request),
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True,
                "stream_options": {"include_usage": True},
            }

            tools = self._build_tools(request)
            if tools:
                params["tools"] = tools
                if request.tool_choice:
                    if isinstance(request.tool_choice, str):
                        params["tool_choice"] = request.tool_choice

            if request.stop:
                params["stop"] = request.stop

            stream = await client.chat.completions.create(**params)

            current_tool_calls: dict[int, dict] = {}

            async for chunk in stream:
                if not chunk.choices and chunk.usage:
                    yield LLMStreamChunk(
                        type="usage",
                        usage=TokenUsage(
                            input_tokens=chunk.usage.prompt_tokens or 0,
                            output_tokens=chunk.usage.completion_tokens or 0,
                        ),
                    )
                    continue

                if not chunk.choices:
                    continue

                delta = chunk.choices[0].delta
                finish = chunk.choices[0].finish_reason

                # Text content
                if delta.content:
                    yield LLMStreamChunk(type="text_delta", text=delta.content)

                # Tool calls
                if delta.tool_calls:
                    for tc_delta in delta.tool_calls:
                        idx = tc_delta.index
                        if idx not in current_tool_calls:
                            current_tool_calls[idx] = {
                                "id": tc_delta.id or "",
                                "name": tc_delta.function.name if tc_delta.function and tc_delta.function.name else "",
                                "arguments": "",
                            }
                            if current_tool_calls[idx]["name"]:
                                yield LLMStreamChunk(
                                    type="tool_call_start",
                                    tool_call=ToolCall(
                                        id=current_tool_calls[idx]["id"],
                                        name=current_tool_calls[idx]["name"],
                                        arguments={},
                                    ),
                                )
                        if tc_delta.function and tc_delta.function.arguments:
                            current_tool_calls[idx]["arguments"] += tc_delta.function.arguments
                            yield LLMStreamChunk(
                                type="tool_call_delta",
                                tool_call_id=current_tool_calls[idx]["id"],
                                argument_delta=tc_delta.function.arguments,
                            )

                if finish:
                    # Emit tool_call_end for completed tool calls
                    for tc_data in current_tool_calls.values():
                        try:
                            args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {"_raw": tc_data["arguments"]}
                        yield LLMStreamChunk(
                            type="tool_call_end",
                            tool_call=ToolCall(
                                id=tc_data["id"],
                                name=tc_data["name"],
                                arguments=args,
                            ),
                        )

                    yield LLMStreamChunk(
                        type="done",
                        finish_reason="tool_use" if finish == "tool_calls" else finish,
                    )

        except RateLimitError as e:
            raise LLMRateLimitError(self.provider_name, model) from e
        except AuthenticationError as e:
            raise LLMAuthenticationError(str(e), self.provider_name, model) from e
        except APIError as e:
            raise LLMProviderError(str(e), self.provider_name, model) from e

    async def shutdown(self) -> None:
        """
        Close the shared HTTP client to release pooled connections.
        """
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None
