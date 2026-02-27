"""
Token counting for message lists by provider.

Uses tiktoken for OpenAI/Groq, Anthropic's counting API when available,
and an approximate (chars/4) fallback for other providers.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from curio_agent_sdk.models.llm import Message, ToolSchema

logger = logging.getLogger(__name__)

# Default chars-per-token for approximate counting (English text)
APPROXIMATE_CHARS_PER_TOKEN = 4

# Module-level cache for tiktoken encoders to avoid repeated lookup cost
_TIKTOKEN_ENCODERS: dict[str, Any] = {}


def _get_model_name(model: str) -> str:
    """Extract model name from 'provider:model' or return as-is."""
    if ":" in model:
        return model.split(":", 1)[1].strip()
    return model.strip()


def _infer_provider(model: str) -> str:
    """
    Infer provider from model string.
    Handles 'provider:model' format or model name heuristics.
    """
    if ":" in model:
        return model.split(":", 1)[0].strip().lower()
    name = model.lower()
    if name.startswith("gpt-") or name.startswith("o1") or "openai" in name:
        return "openai"
    if name.startswith("claude"):
        return "anthropic"
    if "llama" in name or "llama-" in name or "mixtral" in name:
        # Groq often uses llama; ollama uses llama3.1:8b
        if ":" in model:
            return "ollama"
        return "groq"
    return "unknown"


def _messages_to_text(messages: list) -> str:
    """Flatten messages to a single string for approximate token counting."""
    parts = []
    for m in messages:
        role = getattr(m, "role", "")
        content = getattr(m, "content", None) or ""
        if isinstance(content, str):
            parts.append(f"{role}: {content}")
        elif isinstance(content, list):
            for block in content:
                if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                    parts.append(f"{role}: {block.text}")
        if getattr(m, "tool_calls", None):
            for tc in m.tool_calls:
                parts.append(f"tool_call: {getattr(tc, 'name', '')} {getattr(tc, 'arguments', {})}")
    return "\n".join(parts)


def _get_tiktoken_encoding(model_name: str) -> Any | None:
    """
    Get a cached tiktoken encoding for the given model name.

    Falls back to "cl100k_base" for unknown models. Returns None if tiktoken
    is not installed or an encoding cannot be created.
    """
    global _TIKTOKEN_ENCODERS

    # Return cached encoder if available
    enc = _TIKTOKEN_ENCODERS.get(model_name)
    if enc is not None:
        return enc

    try:
        import tiktoken
    except ImportError:
        logger.warning("tiktoken not installed; using approximate token count")
        return None

    try:
        enc = tiktoken.encoding_for_model(model_name)
    except KeyError:
        # Unknown model; use cl100k_base for recent OpenAI/Groq models
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return None

    _TIKTOKEN_ENCODERS[model_name] = enc
    return enc


def _count_tokens_tiktoken_with_encoding(messages: list, encoding: Any) -> int:
    """Count tokens using a provided tiktoken encoding."""

    # OpenAI chat format: every message has ~4 tokens overhead (role, etc.)
    # https://github.com/openai/openai-cookbook/blob/main/examples/How_to_count_tokens_with_tiktoken.ipynb
    tokens_per_message = 4
    tokens_per_name = -1  # name is optional

    total = 0
    for m in messages:
        total += tokens_per_message
        role = getattr(m, "role", "")
        content = getattr(m, "content", None) or ""
        name = getattr(m, "name", None)

        if isinstance(content, str):
            total += len(encoding.encode(content))
        elif isinstance(content, list):
            for block in content:
                if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                    total += len(encoding.encode(block.text))
                # image_url / tool_use / tool_result: approximate or skip
                if getattr(block, "type", None) == "image_url" and getattr(block, "image_url", None):
                    total += 85  # rough estimate for image
                if getattr(block, "type", None) == "tool_use" and getattr(block, "tool_call", None):
                    tc = block.tool_call
                    total += len(encoding.encode(getattr(tc, "name", "") or ""))
                    total += len(encoding.encode(str(getattr(tc, "arguments", {}) or {})))

        if name:
            total += tokens_per_name + len(encoding.encode(name))

        tool_calls = getattr(m, "tool_calls", None)
        if tool_calls:
            for tc in tool_calls:
                total += len(encoding.encode(getattr(tc, "name", "") or ""))
                total += len(encoding.encode(str(getattr(tc, "arguments", {}) or {})))

    return total


def _count_tokens_tiktoken(messages: list, model_name: str) -> int:
    """Count tokens using tiktoken for OpenAI-compatible models."""
    encoding = _get_tiktoken_encoding(model_name)
    if encoding is None:
        return _count_tokens_approximate(messages)
    return _count_tokens_tiktoken_with_encoding(messages, encoding)


def _count_tokens_anthropic(messages: list, model_name: str, tools: list | None = None) -> int:
    """Count tokens using Anthropic's API when the SDK is available."""
    try:
        import anthropic
    except ImportError:
        return _count_tokens_approximate(messages)

    # Convert our Message list to Anthropic format
    anthropic_messages = []
    system_content = ""

    for m in messages:
        role = getattr(m, "role", "")
        content = getattr(m, "content", None)
        tool_call_id = getattr(m, "tool_call_id", None)
        tool_calls = getattr(m, "tool_calls", None)

        if role == "system":
            system_content = m.text if content else ""
            continue

        if role == "assistant":
            if tool_calls:
                blocks = []
                if content and (isinstance(content, str) and content or isinstance(content, list)):
                    text = m.text
                    if text:
                        blocks.append({"type": "text", "text": text})
                for tc in tool_calls:
                    blocks.append({
                        "type": "tool_use",
                        "id": getattr(tc, "id", ""),
                        "name": getattr(tc, "name", ""),
                        "input": getattr(tc, "arguments", {}) or {},
                    })
                anthropic_messages.append({"role": "assistant", "content": blocks})
            else:
                anthropic_messages.append({"role": "assistant", "content": m.text or ""})
            continue

        if role == "tool":
            anthropic_messages.append({
                "role": "user",
                "content": [{
                    "type": "tool_result",
                    "tool_use_id": tool_call_id or "",
                    "content": m.text if content else "",
                }],
            })
            continue

        # user
        if isinstance(content, list):
            blocks = []
            for block in content:
                if getattr(block, "type", None) == "text" and getattr(block, "text", None):
                    blocks.append({"type": "text", "text": block.text})
                if getattr(block, "type", None) == "image_url" and getattr(block, "image_url", None):
                    blocks.append({"type": "image", "source": {"type": "url", "url": block.image_url}})
            anthropic_messages.append({"role": "user", "content": blocks if blocks else m.text})
        else:
            anthropic_messages.append({"role": "user", "content": content or ""})

    try:
        client = anthropic.Anthropic()
        params = {"model": model_name or "claude-sonnet-4-6", "messages": anthropic_messages}
        if system_content:
            params["system"] = system_content
        if tools:
            params["tools"] = [t.to_anthropic_format() for t in tools]
        result = client.messages.count_tokens(**params)
        return result.input_tokens
    except Exception as e:
        logger.warning("Anthropic count_tokens failed (%s), using approximate", e)
        return _count_tokens_approximate(messages)


def _count_tokens_approximate(messages: list) -> int:
    """Approximate token count using character count (e.g. ~4 chars per token)."""
    text = _messages_to_text(messages)
    return max(0, (len(text) + APPROXIMATE_CHARS_PER_TOKEN - 1) // APPROXIMATE_CHARS_PER_TOKEN)


def count_tokens(
    messages: list,
    model: str,
    tools: list | None = None,
) -> int:
    """
    Count input tokens for a list of messages for the given model.

    Uses tiktoken for OpenAI/Groq, Anthropic's counting API for Anthropic models
    when the SDK is available, and an approximate (chars/4) fallback otherwise.

    This function is optimized for repeated calls by caching tiktoken encoders
    per model to avoid repeated encoding lookup overhead.

    Args:
        messages: List of Message objects.
        model: Model identifier, e.g. "gpt-4o", "openai:gpt-4o-mini", "claude-sonnet-4-6".
        tools: Optional list of ToolSchema for tool-definition token overhead (Anthropic).

    Returns:
        Estimated or exact input token count.
    """
    provider = _infer_provider(model)
    model_name = _get_model_name(model)

    if provider in ("openai", "groq"):
        return _count_tokens_tiktoken(messages, model_name)
    if provider == "anthropic":
        return _count_tokens_anthropic(messages, model_name, tools)
    # ollama, unknown, etc.
    return _count_tokens_approximate(messages)


def count_tokens_batch(
    batches: list[list["Message"]],
    model: str,
    tools: list["ToolSchema"] | None = None,
) -> list[int]:
    """
    Count input tokens for multiple message lists for the same model.

    This is more efficient than calling count_tokens() repeatedly because it
    reuses provider/model detection and cached tiktoken encoders.
    """
    provider = _infer_provider(model)
    model_name = _get_model_name(model)

    if provider in ("openai", "groq"):
        encoding = _get_tiktoken_encoding(model_name)
        if encoding is None:
            return [_count_tokens_approximate(msgs) for msgs in batches]
        return [_count_tokens_tiktoken_with_encoding(msgs, encoding) for msgs in batches]

    if provider == "anthropic":
        # Anthropic currently has no bulk count API; fall back to per-batch calls.
        return [_count_tokens_anthropic(msgs, model_name, tools) for msgs in batches]

    # ollama, unknown, etc.
    return [_count_tokens_approximate(msgs) for msgs in batches]
