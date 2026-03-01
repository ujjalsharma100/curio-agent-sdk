"""
Integration tests: Connector + Bridge + Agent (Phase 17 §21.12)

Validates connector integration with the agent.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.connectors.base import Connector, ConnectorResource
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


class MockConnector(Connector):
    """A mock connector for testing."""

    name = "mock_connector"

    def __init__(self):
        self._connected = False

    async def connect(self, credentials=None):
        self._connected = True

    async def disconnect(self):
        self._connected = False

    def get_tools(self):
        return []

    def get_resources(self):
        return [
            ConnectorResource(
                uri="mock://resource/1",
                content="Test resource content",
            )
        ]


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connector_with_agent():
    """Connector integrates with agent."""
    connector = MockConnector()

    mock = MockLLM()
    mock.add_text_response("Connector ready.")

    agent = Agent(
        system_prompt="Test connectors.",
        connectors=[connector],
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Test connector")

    assert result.status == "completed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connector_lists_resources():
    """Connector can list available resources."""
    connector = MockConnector()
    await connector.connect()

    resources = connector.get_resources()
    assert len(resources) == 1
    assert resources[0].content == "Test resource content"

    await connector.disconnect()


@pytest.mark.integration
@pytest.mark.asyncio
async def test_connector_reads_resource():
    """Connector provides resource content for injection."""
    connector = MockConnector()
    await connector.connect()

    resources = connector.get_resources()
    assert len(resources) == 1
    assert resources[0].uri == "mock://resource/1"
    assert "Test resource" in resources[0].content

    await connector.disconnect()
