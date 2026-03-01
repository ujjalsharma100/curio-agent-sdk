"""
Integration tests: Agent + Sessions (Phase 17 §21.6)

Validates multi-turn conversations, session isolation, and session resume.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.state.session import SessionManager, InMemorySessionStore
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_multi_turn_conversation():
    """Messages persist across turns within a session."""
    store = InMemorySessionStore()
    sm = SessionManager(store)

    mock = MockLLM()
    mock.add_text_response("Hello! I'll remember you.")
    mock.add_text_response("You said your name is Alice.")

    agent = Agent(
        system_prompt="Remember names.",
        session_manager=sm,
        llm=mock,
    )
    session = await sm.create(agent.agent_id)

    result1 = await agent.arun("My name is Alice", session_id=session.id)
    assert result1.status == "completed"

    result2 = await agent.arun("What's my name?", session_id=session.id)
    assert result2.status == "completed"

    # Verify messages were stored in session
    messages = await sm.get_messages(session.id)
    assert len(messages) >= 2  # At least user + assistant from both turns


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_isolation():
    """Different sessions don't share messages."""
    store = InMemorySessionStore()
    sm = SessionManager(store)

    mock = MockLLM()
    mock.add_text_response("Session A response.")
    mock.add_text_response("Session B response.")

    agent = Agent(
        system_prompt="Test.",
        session_manager=sm,
        llm=mock,
    )

    session_a = await sm.create(agent.agent_id)
    session_b = await sm.create(agent.agent_id)

    await agent.arun("Message in A", session_id=session_a.id)
    await agent.arun("Message in B", session_id=session_b.id)

    msgs_a = await sm.get_messages(session_a.id)
    msgs_b = await sm.get_messages(session_b.id)

    # Each session should have its own messages
    assert len(msgs_a) >= 1
    assert len(msgs_b) >= 1

    # Content should differ
    a_content = " ".join(getattr(m, "content", "") or "" for m in msgs_a)
    b_content = " ".join(getattr(m, "content", "") or "" for m in msgs_b)
    assert "Message in A" in a_content or "Session A" in a_content
    assert "Message in B" in b_content or "Session B" in b_content


@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_resume():
    """Resume an existing session preserves previous context."""
    store = InMemorySessionStore()
    sm = SessionManager(store)

    mock = MockLLM()
    mock.add_text_response("First response.")
    mock.add_text_response("Resumed response.")

    agent = Agent(
        system_prompt="Test.",
        session_manager=sm,
        llm=mock,
    )

    session = await sm.create(agent.agent_id)
    await agent.arun("First message", session_id=session.id)

    # "Resume" by using the same session_id
    result = await agent.arun("Second message", session_id=session.id)
    assert result.status == "completed"

    # Verify continuity
    messages = await sm.get_messages(session.id)
    assert len(messages) >= 3  # First user + first assistant + second user (at least)
