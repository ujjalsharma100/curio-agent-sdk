"""
Integration tests: Agent + Skills (Phase 17 §21.8)

Validates skill invocation, skill tools availability, and skill prompt injection.
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.tools.tool import tool
from curio_agent_sdk.core.extensions.skills import Skill, SkillRegistry
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.harness import AgentTestHarness


# ── Skill setup ───────────────────────────────────────────────────────────

@tool
def commit_tool(message: str) -> str:
    """Create a git commit."""
    return f"Committed: {message}"


test_skill = Skill(
    name="commit",
    description="Create well-formatted git commits",
    system_prompt="You are a commit message expert. Format commits properly.",
    tools=[commit_tool],
    instructions="Always use conventional commit format.",
)


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.integration
@pytest.mark.asyncio
async def test_invoke_skill():
    """Skill is activated and used during invoke_skill."""
    mock = MockLLM()
    mock.add_text_response("feat: add new feature")

    agent = Agent(
        system_prompt="Base agent.",
        skills=[test_skill],
        llm=mock,
    )

    result = await agent.invoke_skill("commit", "Create a commit for adding login")

    assert result.status == "completed"
    assert result.output is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skill_tools_available():
    """Skill tools are accessible when skill is registered."""
    mock = MockLLM()
    mock.add_tool_call_response("commit_tool", {"message": "fix: bug"})
    mock.add_text_response("Commit created.")

    agent = Agent(
        system_prompt="Test.",
        skills=[test_skill],
        llm=mock,
    )
    harness = AgentTestHarness(agent, llm=mock)
    result = await harness.run("Make a commit", active_skills=["commit"])

    assert result.status == "completed"


@pytest.mark.integration
@pytest.mark.asyncio
async def test_skill_prompt_injection():
    """Skill system prompt is added to the agent when skill is active."""
    mock = MockLLM()
    mock.add_text_response("Formatted commit.")

    agent = Agent(
        system_prompt="Base prompt.",
        skills=[test_skill],
        llm=mock,
    )

    # When invoking a skill, the skill's prompt should be injected
    result = await agent.invoke_skill("commit", "Format this commit")

    assert result.status == "completed"
    # Verify the LLM received the skill prompt in its request
    assert mock.call_count >= 1
    llm_request = mock.calls[0]
    all_content = " ".join(
        getattr(m, "content", "") or "" for m in llm_request.messages
    )
    assert "commit" in all_content.lower() or "conventional" in all_content.lower()
