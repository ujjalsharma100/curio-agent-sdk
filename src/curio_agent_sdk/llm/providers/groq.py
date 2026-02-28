"""
Groq provider with async and native tool calling support.

Groq uses OpenAI-compatible API, so this leverages the openai async client.
"""

from __future__ import annotations

import io
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
)

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI, APIError, RateLimitError, AuthenticationError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class GroqProvider(LLMProvider):
    """
    Groq provider using OpenAI-compatible async client.

    Groq's API is OpenAI-compatible, so we use the openai async client
    pointed at Groq's base URL. This gives us native tool calling,
    streaming, and async support for free.
    """

    provider_name = "groq"

    def __init__(self) -> None:
        # Shared HTTP client for connection pooling across Groq requests.
        self._http_client: httpx.AsyncClient | None = None

    def _get_http_client(self) -> httpx.AsyncClient:
        """
        Lazily create a shared httpx.AsyncClient with connection pooling.

        This client is reused across all Groq requests made by this provider
        instance, while API keys/base URLs remain per-request on the AsyncOpenAI client.
        """
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
        return self._http_client

    def _get_client(self, api_key: str | None = None, base_url: str | None = None) -> AsyncOpenAI:
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required for Groq. Install with: pip install openai")
        return AsyncOpenAI(
            api_key=api_key or "",
            base_url=base_url or GROQ_BASE_URL,
            http_client=self._get_http_client(),
        )

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        """Convert messages to OpenAI format (Groq-compatible)."""
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

    def _build_prompt_cache_key(self, request: LLMRequest) -> str:
        """
        Build a stable cache key for Groq prompt caching based on system prompts and tools.

        Groq's OpenAI-compatible API supports prompt caching; we pass a prompt_cache_key
        to improve routing and cache hit rates when many requests share the same prefix.
        """
        system_text = [m.text for m in request.messages if m.role == "system"]
        tools_schema = []
        if request.tools:
            for t in request.tools:
                tools_schema.append(
                    {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.parameters,
                    }
                )
        payload = json.dumps(
            {
                "system": system_text,
                "tools": tools_schema,
            },
            sort_keys=True,
            default=str,
        )
        import hashlib

        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    async def call(
        self,
        request: LLMRequest,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> LLMResponse:
        client = self._get_client(api_key, base_url)
        model = request.model or "llama-3.1-8b-instant"
        start = time.monotonic()

        try:
            params: dict = {
                "model": model,
                "messages": self._build_messages(request),
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
            }

            if request.tools:
                params["tools"] = [t.to_openai_format() for t in request.tools]
                if request.tool_choice:
                    if isinstance(request.tool_choice, str):
                        params["tool_choice"] = request.tool_choice

            if request.response_format:
                params["response_format"] = request.response_format
            if request.stop:
                params["stop"] = request.stop

            # Prompt caching: Groq's OpenAI-compatible API supports prompt caching via
            # prompt_cache_key, similar to OpenAI.
            cache_key = getattr(request, "prompt_cache_key", None)
            if getattr(request, "prompt_cache", False):
                if not cache_key:
                    cache_key = self._build_prompt_cache_key(request)
            if cache_key:
                params["prompt_cache_key"] = cache_key

            response = await client.chat.completions.create(**params)
            latency_ms = int((time.monotonic() - start) * 1000)

            choice = response.choices[0]
            msg = choice.message

            tool_calls = None
            if msg.tool_calls:
                tool_calls = []
                for tc in msg.tool_calls:
                    try:
                        args = json.loads(tc.function.arguments) if tc.function.arguments else {}
                    except json.JSONDecodeError:
                        args = {"_raw": tc.function.arguments}
                    tool_calls.append(ToolCall(id=tc.id, name=tc.function.name, arguments=args))

            finish_reason = choice.finish_reason or "stop"
            if finish_reason == "tool_calls":
                finish_reason = "tool_use"

            usage = TokenUsage()
            if response.usage:
                usage = TokenUsage(
                    input_tokens=response.usage.prompt_tokens or 0,
                    output_tokens=response.usage.completion_tokens or 0,
                )

            return LLMResponse(
                message=Message(role="assistant", content=msg.content or "", tool_calls=tool_calls),
                usage=usage,
                model=response.model or model,
                provider=self.provider_name,
                finish_reason=finish_reason,
                latency_ms=latency_ms,
                raw_response=response,
            )

        except RateLimitError as e:
            raise LLMRateLimitError(self.provider_name, model) from e
        except AuthenticationError as e:
            raise LLMAuthenticationError(str(e), self.provider_name, model) from e
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
        model = request.model or "llama-3.1-8b-instant"

        try:
            params: dict = {
                "model": model,
                "messages": self._build_messages(request),
                "max_tokens": request.max_tokens,
                "temperature": request.temperature,
                "stream": True,
            }

            if request.tools:
                params["tools"] = [t.to_openai_format() for t in request.tools]
            if request.stop:
                params["stop"] = request.stop

            # Prompt caching for streaming responses as well
            cache_key = getattr(request, "prompt_cache_key", None)
            if getattr(request, "prompt_cache", False):
                if not cache_key:
                    cache_key = self._build_prompt_cache_key(request)
            if cache_key:
                params["prompt_cache_key"] = cache_key

            stream = await client.chat.completions.create(**params)
            current_tool_calls: dict[int, dict] = {}

            async for chunk in stream:
                if not chunk.choices:
                    if chunk.usage:
                        yield LLMStreamChunk(
                            type="usage",
                            usage=TokenUsage(
                                input_tokens=chunk.usage.prompt_tokens or 0,
                                output_tokens=chunk.usage.completion_tokens or 0,
                            ),
                        )
                    continue

                delta = chunk.choices[0].delta
                finish = chunk.choices[0].finish_reason

                if delta.content:
                    yield LLMStreamChunk(type="text_delta", text=delta.content)

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
                    for tc_data in current_tool_calls.values():
                        try:
                            args = json.loads(tc_data["arguments"]) if tc_data["arguments"] else {}
                        except json.JSONDecodeError:
                            args = {"_raw": tc_data["arguments"]}
                        yield LLMStreamChunk(
                            type="tool_call_end",
                            tool_call=ToolCall(id=tc_data["id"], name=tc_data["name"], arguments=args),
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


class GroqBatchClient:
    """
    Helper for the Groq Batch API (async, offline batch processing).

    Groq exposes an OpenAI-compatible batch endpoint at /openai/v1/batches.
    This helper mirrors OpenAIBatchClient but targets the Groq base URL.
    """

    def __init__(self, api_key: str | None = None) -> None:
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required for Groq. Install with: pip install openai")
        self._http_client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=10, max_keepalive_connections=5),
        )
        self._client = AsyncOpenAI(
            api_key=api_key or "",
            base_url=GROQ_BASE_URL,
            http_client=self._http_client,
        )

    async def create_chat_batch(
        self,
        requests: list[dict],
        completion_window: str = "24h",
    ):
        """
        Create a Groq chat completions batch job.

        `requests` should be a list of dicts with fields:
        - custom_id
        - method (e.g., "POST")
        - url (e.g., "/v1/chat/completions")
        - body (chat.completions payload)
        """
        lines = [json.dumps(r, separators=(",", ":")) for r in requests]
        data = "\n".join(lines).encode("utf-8")

        input_file = await self._client.files.create(
            file=("batch.jsonl", io.BytesIO(data), "application/jsonl"),
            purpose="batch",
        )

        batch = await self._client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/chat/completions",
            completion_window=completion_window,
        )
        return batch

    async def retrieve_batch(self, batch_id: str):
        """Retrieve metadata for a previously created batch job."""
        return await self._client.batches.retrieve(batch_id)

    async def iter_batch_results(self, batch_id: str):
        """
        Stream parsed JSON lines from a completed batch's output file.

        Yields each result line as a Python dict.
        """
        batch = await self._client.batches.retrieve(batch_id)
        output_file_id = getattr(batch, "output_file_id", None)
        if not output_file_id:
            return

        stream = await self._client.files.content(output_file_id)
        async for chunk in stream:
            text = chunk.decode("utf-8") if isinstance(chunk, (bytes, bytearray)) else str(chunk)
            for line in text.splitlines():
                if not line.strip():
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue

    async def shutdown(self) -> None:
        """Close underlying HTTP client."""
        await self._http_client.aclose()
