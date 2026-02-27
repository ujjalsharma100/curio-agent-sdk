"""
Anthropic provider with async, native tool_use, and streaming support.

Supports Claude 4.x, 3.5, 3 model families.
"""

from __future__ import annotations

import json
import logging
import time
from typing import AsyncIterator

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
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False


class AnthropicProvider(LLMProvider):
    """
    Anthropic provider with full support for:
    - Native tool_use
    - Streaming responses
    - Async operations
    - Per-request API key (thread-safe)
    - System prompt as top-level parameter (Anthropic's convention)
    """

    provider_name = "anthropic"

    def _get_client(self, api_key: str | None = None) -> anthropic.AsyncAnthropic:
        if not ANTHROPIC_AVAILABLE:
            raise ImportError("anthropic package not installed. Install with: pip install anthropic")
        kwargs = {}
        if api_key:
            kwargs["api_key"] = api_key
        return anthropic.AsyncAnthropic(**kwargs)

    def _build_params(self, request: LLMRequest, model: str) -> dict:
        """Build Anthropic API params from LLMRequest."""
        # Separate system message from conversation messages
        system_content = ""
        messages = []

        for msg in request.messages:
            if msg.role == "system":
                system_content = msg.content if isinstance(msg.content, str) else ""
            elif msg.role == "assistant":
                content: list | str
                if msg.tool_calls:
                    # Build content blocks with text + tool_use
                    blocks = []
                    text = msg.content if isinstance(msg.content, str) else ""
                    if text:
                        blocks.append({"type": "text", "text": text})
                    for tc in msg.tool_calls:
                        blocks.append({
                            "type": "tool_use",
                            "id": tc.id,
                            "name": tc.name,
                            "input": tc.arguments,
                        })
                    content = blocks
                else:
                    content = msg.content if isinstance(msg.content, str) else ""
                messages.append({"role": "assistant", "content": content})
            elif msg.role == "tool":
                # Anthropic tool results are user messages with tool_result content blocks
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": msg.tool_call_id or "",
                        "content": msg.content if isinstance(msg.content, str) else "",
                    }],
                })
            else:  # user
                messages.append({
                    "role": msg.role,
                    "content": msg.content if isinstance(msg.content, str) else "",
                })

        params: dict = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        if system_content:
            params["system"] = system_content

        # Tools
        if request.tools:
            params["tools"] = [t.to_anthropic_format() for t in request.tools]
            if request.tool_choice:
                if request.tool_choice == "auto":
                    params["tool_choice"] = {"type": "auto"}
                elif request.tool_choice == "required":
                    params["tool_choice"] = {"type": "any"}
                elif request.tool_choice == "none":
                    pass  # Don't send tools
                elif isinstance(request.tool_choice, dict) and "name" in request.tool_choice:
                    params["tool_choice"] = {
                        "type": "tool",
                        "name": request.tool_choice["name"],
                    }

        if request.stop:
            params["stop_sequences"] = request.stop

        # Structured output (JSON schema or json_object)
        if request.response_format:
            rf = request.response_format
            if rf.get("type") == "json_schema" and "json_schema" in rf:
                # OpenAI-style -> Anthropic output_config
                js = rf["json_schema"]
                params["output_config"] = {
                    "format": {
                        "type": "json_schema",
                        "schema": js.get("schema", js),
                    }
                }
            elif rf.get("type") == "json_object":
                params["output_config"] = {"format": {"type": "json_object"}}

        return params

    def _parse_response(self, response, model: str, latency_ms: int) -> LLMResponse:
        """Parse Anthropic response into LLMResponse."""
        text_parts = []
        tool_calls = []

        for block in response.content:
            if block.type == "text":
                text_parts.append(block.text)
            elif block.type == "tool_use":
                tool_calls.append(ToolCall(
                    id=block.id,
                    name=block.name,
                    arguments=block.input if isinstance(block.input, dict) else {},
                ))

        finish_reason = response.stop_reason or "stop"
        if finish_reason == "tool_use":
            finish_reason = "tool_use"
        elif finish_reason == "end_turn":
            finish_reason = "stop"

        usage = TokenUsage(
            input_tokens=response.usage.input_tokens,
            output_tokens=response.usage.output_tokens,
            cache_read_tokens=getattr(response.usage, "cache_read_input_tokens", 0) or 0,
            cache_write_tokens=getattr(response.usage, "cache_creation_input_tokens", 0) or 0,
        )

        return LLMResponse(
            message=Message(
                role="assistant",
                content="\n".join(text_parts) if text_parts else "",
                tool_calls=tool_calls if tool_calls else None,
            ),
            usage=usage,
            model=response.model or model,
            provider=self.provider_name,
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
        client = self._get_client(api_key)
        model = request.model or "claude-sonnet-4-6"
        start = time.monotonic()

        try:
            params = self._build_params(request, model)
            response = await client.messages.create(**params)
            latency_ms = int((time.monotonic() - start) * 1000)
            return self._parse_response(response, model, latency_ms)

        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(self.provider_name, model) from e
        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(str(e), self.provider_name, model) from e
        except anthropic.APITimeoutError as e:
            raise LLMTimeoutError(str(e), self.provider_name, model) from e
        except anthropic.APIError as e:
            status = getattr(e, "status_code", None)
            raise LLMProviderError(str(e), self.provider_name, model, status) from e

    async def stream(
        self,
        request: LLMRequest,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> AsyncIterator[LLMStreamChunk]:
        client = self._get_client(api_key)
        model = request.model or "claude-sonnet-4-6"

        try:
            params = self._build_params(request, model)

            async with client.messages.stream(**params) as stream:
                current_tool: dict | None = None

                async for event in stream:
                    if event.type == "content_block_start":
                        block = event.content_block
                        if block.type == "tool_use":
                            current_tool = {"id": block.id, "name": block.name, "arguments": ""}
                            yield LLMStreamChunk(
                                type="tool_call_start",
                                tool_call=ToolCall(id=block.id, name=block.name, arguments={}),
                            )

                    elif event.type == "content_block_delta":
                        delta = event.delta
                        if delta.type == "text_delta":
                            yield LLMStreamChunk(type="text_delta", text=delta.text)
                        elif delta.type == "input_json_delta" and current_tool:
                            current_tool["arguments"] += delta.partial_json
                            yield LLMStreamChunk(
                                type="tool_call_delta",
                                tool_call_id=current_tool["id"],
                                argument_delta=delta.partial_json,
                            )

                    elif event.type == "content_block_stop":
                        if current_tool:
                            try:
                                args = json.loads(current_tool["arguments"]) if current_tool["arguments"] else {}
                            except json.JSONDecodeError:
                                args = {"_raw": current_tool["arguments"]}
                            yield LLMStreamChunk(
                                type="tool_call_end",
                                tool_call=ToolCall(
                                    id=current_tool["id"],
                                    name=current_tool["name"],
                                    arguments=args,
                                ),
                            )
                            current_tool = None

                    elif event.type == "message_delta":
                        stop_reason = getattr(event.delta, "stop_reason", None)
                        output_tokens = getattr(getattr(event, "usage", None), "output_tokens", 0)
                        if stop_reason:
                            fr = "tool_use" if stop_reason == "tool_use" else "stop"
                            yield LLMStreamChunk(
                                type="done",
                                finish_reason=fr,
                                usage=TokenUsage(output_tokens=output_tokens or 0),
                            )

                    elif event.type == "message_start":
                        msg_usage = getattr(event.message, "usage", None)
                        if msg_usage:
                            yield LLMStreamChunk(
                                type="usage",
                                usage=TokenUsage(
                                    input_tokens=msg_usage.input_tokens or 0,
                                    cache_read_tokens=getattr(msg_usage, "cache_read_input_tokens", 0) or 0,
                                    cache_write_tokens=getattr(msg_usage, "cache_creation_input_tokens", 0) or 0,
                                ),
                            )

        except anthropic.RateLimitError as e:
            raise LLMRateLimitError(self.provider_name, model) from e
        except anthropic.AuthenticationError as e:
            raise LLMAuthenticationError(str(e), self.provider_name, model) from e
        except anthropic.APIError as e:
            raise LLMProviderError(str(e), self.provider_name, model) from e
