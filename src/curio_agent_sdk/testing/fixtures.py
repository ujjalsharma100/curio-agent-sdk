"""
Pytest fixtures for Curio Agent SDK test setups.

Use these fixtures in your test suite by registering the plugin in your
``conftest.py``:

    pytest_plugins = ["curio_agent_sdk.testing.fixtures"]

Then in your tests you can use: ``mock_llm``, ``agent_test_harness``,
``in_memory_state_store``, ``in_memory_session_store``, ``in_memory_persistence``,
``tool_test_kit``, and ``agent``.

Requires: pytest (optional dependency for tests).
"""

from __future__ import annotations

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.state_store import InMemoryStateStore
from curio_agent_sdk.core.session import InMemorySessionStore
from curio_agent_sdk.persistence.memory import InMemoryPersistence
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness
from curio_agent_sdk.testing.toolkit import ToolTestKit


@pytest.fixture
def mock_llm() -> MockLLM:
    """Provide a fresh MockLLM for each test."""
    return MockLLM()


@pytest.fixture
def in_memory_state_store() -> InMemoryStateStore:
    """Provide a fresh InMemoryStateStore for isolated tests."""
    return InMemoryStateStore()


@pytest.fixture
def in_memory_session_store() -> InMemorySessionStore:
    """Provide a fresh InMemorySessionStore for isolated tests."""
    return InMemorySessionStore()


@pytest.fixture
def in_memory_persistence() -> InMemoryPersistence:
    """Provide a fresh InMemoryPersistence for isolated tests."""
    return InMemoryPersistence()


@pytest.fixture
def tool_test_kit() -> ToolTestKit:
    """Provide a fresh ToolTestKit for tool-level assertions."""
    return ToolTestKit()


@pytest.fixture
def agent(mock_llm: MockLLM) -> Agent:
    """
    Provide a minimal Agent wired with MockLLM and no tools.

    Override in tests or use builder for custom tools/memory.
    """
    return Agent(
        system_prompt="You are a test assistant.",
        tools=[],
        llm=mock_llm,
    )


@pytest.fixture
def agent_test_harness(agent: Agent, mock_llm: MockLLM) -> AgentTestHarness:
    """
    Provide an AgentTestHarness with the fixture agent and mock LLM.

    Use harness.run() / harness.run_sync() and then assert on
    harness.tool_calls, harness.llm_calls, harness.result.
    """
    return AgentTestHarness(agent, llm=mock_llm)
