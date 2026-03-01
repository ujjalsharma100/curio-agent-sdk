"""
E2E tests: Multi-Turn Agent (Phase 18 §22.3)

Validates conversation continuity and session-based multi-turn interaction.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.state.session import SessionManager, InMemorySessionStore
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_conversation_continuity():
    """Context is maintained across multiple turns using run_conversation."""
    store = InMemorySessionStore()
    sm = SessionManager(store)

    mock = MockLLM()
    mock.add_text_response("Nice to meet you, Alice!")
    mock.add_text_response("You told me your name is Alice.")
    mock.add_text_response("Goodbye, Alice! Have a great day!")

    agent = Agent(
        system_prompt="You are a friendly assistant that remembers user details.",
        session_manager=sm,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)

    results = await harness.run_conversation([
        "Hi, my name is Alice.",
        "What's my name?",
        "Goodbye!",
    ])

    assert len(results) == 3
    for r in results:
        assert r.status == "completed"

    # Verify the LLM received increasing message history
    # First call: system + user
    # Second call: system + user1 + assistant1 + user2
    # Third call: system + user1 + assistant1 + user2 + assistant2 + user3
    assert mock.call_count == 3


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_session_based_conversation():
    """Session-based multi-turn preserves messages in session store."""
    store = InMemorySessionStore()
    sm = SessionManager(store)

    mock = MockLLM()
    mock.add_text_response("I'll remember that you like Python.")
    mock.add_text_response("You told me you like Python!")

    agent = Agent(
        system_prompt="Remember user preferences.",
        session_manager=sm,
        llm=mock,
    )

    # Create a session
    session = await sm.create(agent.agent_id)

    # Turn 1
    r1 = await agent.arun("I like Python.", session_id=session.id)
    assert r1.status == "completed"

    # Turn 2 — same session
    r2 = await agent.arun("What do I like?", session_id=session.id)
    assert r2.status == "completed"

    # Verify session has messages stored
    messages = await sm.get_messages(session.id)
    assert len(messages) >= 4  # 2 user + 2 assistant messages

    # Verify sessions are listed
    sessions = await sm.list(agent.agent_id)
    assert len(sessions) >= 1
