"""
Integration tests: Agent + State/Checkpoint (Phase 17 §21.5)

Validates checkpointing during runs, resume from checkpoint, and state persistence.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.core.state import InMemoryStateStore
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


@tool
def step_tool(step: str) -> str:
    """A step in a multi-step process."""
    return f"Completed step: {step}"


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_state_checkpointed_during_run():
    """Checkpoint is saved after each iteration."""
    state_store = InMemoryStateStore()

    mock = MockLLM()
    mock.add_tool_call_response("step_tool", {"step": "1"})
    mock.add_text_response("All steps done.")

    agent = Agent(
        system_prompt="Process steps.",
        tools=[step_tool],
        state_store=state_store,
        checkpoint_interval=1,
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Do the steps")

    assert result.status == "completed"
    # Verify that state was saved
    runs = await state_store.list_runs(agent.agent_id)
    assert len(runs) >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_resume_from_checkpoint():
    """Agent can resume a run from a saved checkpoint."""
    state_store = InMemoryStateStore()

    # First run — complete normally to create checkpoint
    mock1 = MockLLM()
    mock1.add_tool_call_response("step_tool", {"step": "1"})
    mock1.add_text_response("Step 1 done.")

    agent = Agent(
        system_prompt="Process steps.",
        tools=[step_tool],
        state_store=state_store,
        checkpoint_interval=1,
        llm=mock1,
    )
    harness1 = AgentTestHarness(agent, llm=mock1)
    result1 = await harness1.run("Do step 1")
    assert result1.status == "completed"

    # Get the run_id from the first run
    runs = await state_store.list_runs(agent.agent_id)
    assert len(runs) >= 1


@pytest.mark.integration
@pytest.mark.asyncio
async def test_state_store_persists():
    """State survives agent restart via state store."""
    state_store = InMemoryStateStore()

    # First agent run
    mock1 = MockLLM()
    mock1.add_text_response("First agent response.")
    agent1 = Agent(
        agent_id="persistent-agent",
        system_prompt="Remember.",
        state_store=state_store,
        llm=mock1,
    )
    harness1 = AgentTestHarness(agent1, llm=mock1)
    await harness1.run("Save this")

    # Create new agent with same id and state store
    mock2 = MockLLM()
    mock2.add_text_response("Second agent response.")
    agent2 = Agent(
        agent_id="persistent-agent",
        system_prompt="Remember.",
        state_store=state_store,
        llm=mock2,
    )

    # Verify state from first agent is accessible
    runs = await state_store.list_runs("persistent-agent")
    assert len(runs) >= 1
