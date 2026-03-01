"""
Unit tests for ContextManager (Phase 14 — Context & Credentials).

Uses approximate token counting (model like ollama/unknown) for deterministic tests.
"""

import pytest

from curio_agent_sdk.core.context.context import (
    ContextManager,
    SUMMARIZE_PLACEHOLDER,
)
from curio_agent_sdk.models.llm import Message


# Use a model that triggers approximate counting (chars/4) for predictable token counts
APPROX_MODEL = "ollama:llama3"


@pytest.mark.unit
def test_count_tokens_basic():
    """Basic token counting."""
    mgr = ContextManager(max_tokens=4096, reserve_tokens=0)
    messages = [
        Message.system("You are helpful."),
        Message.user("Hello"),
    ]
    count = mgr.count_tokens(messages, APPROX_MODEL)
    assert count >= 1
    # Approximate: "system: You are helpful.\nuser: Hello" -> ~35 chars -> ~9 tokens
    assert 5 <= count <= 20


@pytest.mark.unit
def test_fit_messages_under_limit():
    """All messages fit within budget."""
    mgr = ContextManager(max_tokens=8192, reserve_tokens=500)
    messages = [
        Message.system("Short system."),
        Message.user("Hi"),
        Message.assistant("Hi back."),
    ]
    fitted = mgr.fit_messages(messages, model=APPROX_MODEL)
    assert len(fitted) == 3
    assert fitted[0].content == "Short system."
    assert fitted[1].content == "Hi"
    assert fitted[2].content == "Hi back."


@pytest.mark.unit
def test_fit_messages_truncate():
    """Truncate oldest strategy drops oldest non-system messages."""
    # Budget after reserve: 40 tokens. Approximate counting: ~4 chars per token.
    # 5 messages with long_content (80 chars each) + system -> well over 40 tokens.
    mgr = ContextManager(
        max_tokens=40,
        reserve_tokens=0,
        strategy="truncate_oldest",
    )
    long_content = "x" * 80  # ~20 tokens per message with role overhead
    messages = [
        Message.system("System prompt."),
        Message.user(long_content),
        Message.assistant(long_content),
        Message.user(long_content),
        Message.assistant(long_content),
    ]
    fitted = mgr.fit_messages(messages, model=APPROX_MODEL)
    assert len(fitted) < len(messages)
    assert fitted[0].role == "system"
    assert fitted[0].content == "System prompt."


@pytest.mark.unit
def test_fit_messages_summarize():
    """Summarize strategy replaces truncated prefix with placeholder or summarizer output."""
    # Budget below full message set (~43 tokens) so summarize path runs.
    mgr = ContextManager(
        max_tokens=35,
        reserve_tokens=0,
        strategy="summarize",
    )
    long_content = "x" * 60  # ~15+ tokens each
    messages = [
        Message.system("System."),
        Message.user(long_content),
        Message.assistant(long_content),
        Message.user("Last user"),
    ]
    fitted = mgr.fit_messages(messages, model=APPROX_MODEL)
    assert fitted[0].role == "system"
    assert fitted[0].content == "System."
    # Second message should be summary placeholder (no summarizer callback)
    assert fitted[1].content == SUMMARIZE_PLACEHOLDER
    assert any(m.content == "Last user" for m in fitted)


@pytest.mark.unit
def test_reserve_tokens():
    """Reserve tokens reduce effective budget for message fitting."""
    mgr = ContextManager(max_tokens=100, reserve_tokens=50)
    assert mgr._budget == 50
    messages = [
        Message.system("Sys."),
        Message.user("A" * 200),  # ~50+ tokens
    ]
    fitted = mgr.fit_messages(messages, model=APPROX_MODEL)
    # With budget 50, we may get only system + part of user or truncation
    assert len(fitted) >= 1
    assert fitted[0].role == "system"


@pytest.mark.unit
def test_empty_messages():
    """Handle empty message list."""
    mgr = ContextManager(max_tokens=4096)
    fitted = mgr.fit_messages([], model=APPROX_MODEL)
    assert fitted == []


@pytest.mark.unit
def test_system_message_preserved():
    """System message is never truncated; only first system message kept at start."""
    mgr = ContextManager(max_tokens=100, reserve_tokens=0, strategy="truncate_oldest")
    messages = [
        Message.system("The one system message."),
        Message.user("U1"),
        Message.assistant("A1"),
        Message.user("U2"),
    ]
    fitted = mgr.fit_messages(messages, model=APPROX_MODEL)
    assert fitted[0].role == "system"
    assert fitted[0].content == "The one system message."


@pytest.mark.unit
def test_fit_messages_with_tools():
    """fit_messages accepts tools param; token count includes tool definitions."""
    mgr = ContextManager(max_tokens=4096, reserve_tokens=0)
    messages = [Message.system("Sys."), Message.user("Hi")]
    tools = [{"type": "function", "name": "foo", "description": "bar"}]
    fitted = mgr.fit_messages(messages, tools=tools, model=APPROX_MODEL)
    assert len(fitted) == 2
    assert mgr.count_tokens(messages, APPROX_MODEL, tools) >= mgr.count_tokens(messages, APPROX_MODEL)


@pytest.mark.unit
def test_summarizer_callback_invoked():
    """When summarizer is provided it is called and its result is used."""
    summary_content = "Summary of prior messages."
    def summarizer(msgs, model, tools):
        return Message.system(summary_content)
    mgr = ContextManager(
        max_tokens=35,
        reserve_tokens=0,
        strategy="summarize",
        summarizer=summarizer,
    )
    messages = [
        Message.system("System."),
        Message.user("x" * 60),
        Message.assistant("x" * 60),
        Message.user("Last"),
    ]
    fitted = mgr.fit_messages(messages, model=APPROX_MODEL)
    assert fitted[1].content == summary_content
    assert any(m.content == "Last" for m in fitted)


@pytest.mark.unit
def test_summarizer_exception_uses_placeholder():
    """When summarizer raises, placeholder is used."""
    def summarizer(msgs, model, tools):
        raise RuntimeError("summarizer failed")
    mgr = ContextManager(
        max_tokens=35,
        reserve_tokens=0,
        strategy="summarize",
        summarizer=summarizer,
    )
    messages = [
        Message.system("System."),
        Message.user("x" * 60),
        Message.assistant("x" * 60),
        Message.user("Last"),
    ]
    fitted = mgr.fit_messages(messages, model=APPROX_MODEL)
    assert fitted[1].content == SUMMARIZE_PLACEHOLDER


@pytest.mark.unit
def test_truncate_preserves_tool_call_result_pairs():
    """Truncate keeps assistant tool-call messages with their tool-result pairs."""
    from curio_agent_sdk.models.llm import ToolCall
    mgr = ContextManager(max_tokens=60, reserve_tokens=0, strategy="truncate_oldest")
    messages = [
        Message.system("Sys."),
        Message.user("Old user"),
        Message.assistant("", tool_calls=[ToolCall(id="call_1", name="foo", arguments={})]),
        Message.tool_result("call_1", "result for call_1"),
        Message.user("New user"),
    ]
    fitted = mgr.fit_messages(messages, model=APPROX_MODEL)
    roles = [m.role for m in fitted]
    assert "system" in roles
    assert "user" in roles
    # Assistant + tool pair should be kept together (grouped)
    if "assistant" in roles:
        idx = roles.index("assistant")
        if idx + 1 < len(roles) and roles[idx + 1] == "tool":
            assert fitted[idx + 1].tool_call_id == "call_1"
