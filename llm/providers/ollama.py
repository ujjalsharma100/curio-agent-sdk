"""
Ollama provider for local model inference.

Uses the OpenAI-compatible API that Ollama exposes.
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
from curio_agent_sdk.exceptions import LLMProviderError

logger = logging.getLogger(__name__)

try:
    from openai import AsyncOpenAI, APIError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False


OLLAMA_DEFAULT_URL = "http://localhost:11434/v1"


class OllamaProvider(LLMProvider):
    """
    Ollama provider using OpenAI-compatible API.

    Ollama exposes an OpenAI-compatible endpoint at /v1, so we use the
    openai async client pointed at the local Ollama server.
    """

    provider_name = "ollama"

    def _get_client(self, base_url: str | None = None) -> AsyncOpenAI:
        if not OPENAI_AVAILABLE:
            raise ImportError("openai package required for Ollama. Install with: pip install openai")
        return AsyncOpenAI(
            api_key="ollama",  # Ollama doesn't need a real key
            base_url=base_url or OLLAMA_DEFAULT_URL,
        )

    def _build_messages(self, request: LLMRequest) -> list[dict]:
        messages = []
        for msg in request.messages:
            m: dict = {"role": msg.role}
            if msg.role == "tool":
                m["content"] = msg.content if isinstance(msg.content, str) else ""
                m["tool_call_id"] = msg.tool_call_id or ""
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

    async def call(
        self,
        request: LLMRequest,
        api_key: str | None = None,
        base_url: str | None = None,
    ) -> LLMResponse:
        client = self._get_client(base_url)
        model = request.model or "llama3.1:8b"
        start = time.monotonic()

        try:
            params: dict = {
                "model": model,
                "messages": self._build_messages(request),
                "temperature": request.temperature,
            }

            # Ollama may not support max_tokens for all models, pass as best effort
            if request.max_tokens:
                params["max_tokens"] = request.max_tokens

            if request.tools:
                params["tools"] = [t.to_openai_format() for t in request.tools]

            if request.stop:
                params["stop"] = request.stop

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

        except APIError as e:
            status = getattr(e, "status_code", None)
            raise LLMProviderError(str(e), self.provider_name, model, status) from e
        except Exception as e:
            raise LLMProviderError(str(e), self.provider_name, model) from e
