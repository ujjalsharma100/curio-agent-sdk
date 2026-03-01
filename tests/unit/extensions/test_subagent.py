"""
Unit tests for the Subagent system — SubagentConfig, AgentOrchestrator.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch, PropertyMock

import pytest

from curio_agent_sdk.core.extensions.subagent import (
    SubagentConfig,
    AgentOrchestrator,
)
from curio_agent_sdk.models.agent import AgentRunResult
from curio_agent_sdk.models.llm import Message


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_agent(**overrides):
    """Create a mock Agent with the minimum attributes AgentOrchestrator needs."""
    agent = MagicMock()
    agent.llm = MagicMock()
    agent.registry = MagicMock()
    agent.registry.tools = []
    agent.registry.get_llm_schemas = MagicMock(return_value=[])
    agent.memory_manager_instance = None
    agent.hook_registry = MagicMock()
    agent.runtime = None
    agent.agent_id = "parent-001"
    agent.max_iterations = 25
    for k, v in overrides.items():
        setattr(agent, k, v)
    return agent


def _success_result(output: str = "done", run_id: str = "run-1") -> AgentRunResult:
    return AgentRunResult(status="completed", output=output, run_id=run_id)


# ---------------------------------------------------------------------------
# 13.2  SubagentConfig
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSubagentConfig:
    def test_defaults(self):
        cfg = SubagentConfig(name="helper", system_prompt="You help.")
        assert cfg.name == "helper"
        assert cfg.system_prompt == "You help."
        assert cfg.tools == []
        assert cfg.model is None
        assert cfg.inherit_memory is False
        assert cfg.inherit_tools is False
        assert cfg.inherit_hooks is True
        assert cfg.max_iterations == 10
        assert cfg.timeout is None

    def test_custom_values(self):
        cfg = SubagentConfig(
            name="researcher",
            system_prompt="Research things.",
            model="gpt-4",
            inherit_memory=True,
            inherit_tools=True,
            inherit_hooks=False,
            max_iterations=5,
            timeout=30.0,
        )
        assert cfg.model == "gpt-4"
        assert cfg.inherit_memory is True
        assert cfg.inherit_tools is True
        assert cfg.inherit_hooks is False
        assert cfg.max_iterations == 5
        assert cfg.timeout == 30.0


# ---------------------------------------------------------------------------
# 13.2  AgentOrchestrator — register / get / list
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorRegistry:
    def test_register_and_get(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        cfg = SubagentConfig(name="sub1", system_prompt="Hi")
        orch.register("sub1", cfg)
        assert orch.get_config("sub1") is cfg

    def test_get_unknown_returns_none(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        assert orch.get_config("nope") is None

    def test_list_names(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        orch.register("a", SubagentConfig(name="a", system_prompt="A"))
        orch.register("b", SubagentConfig(name="b", system_prompt="B"))
        assert set(orch.list_names()) == {"a", "b"}

    def test_register_renames_config(self):
        """When name in config doesn't match register key, config gets renamed."""
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        cfg = SubagentConfig(name="old_name", system_prompt="Hi")
        orch.register("new_name", cfg)
        stored = orch.get_config("new_name")
        assert stored is not None
        assert stored.name == "new_name"


# ---------------------------------------------------------------------------
# 13.2  AgentOrchestrator — spawn
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorSpawn:
    @pytest.mark.asyncio
    async def test_spawn_with_config(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)

        expected_result = _success_result()
        cfg = SubagentConfig(name="worker", system_prompt="Work.")

        with patch.object(orch, "_build_subagent") as mock_build:
            mock_subagent = AsyncMock()
            mock_subagent.arun = AsyncMock(return_value=expected_result)
            mock_subagent.start = AsyncMock()
            mock_subagent.close = AsyncMock()
            mock_build.return_value = mock_subagent

            result = await orch.spawn(cfg, "do something")

        assert result.status == "completed"
        mock_subagent.arun.assert_awaited_once()
        mock_subagent.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_spawn_by_name(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        cfg = SubagentConfig(name="named", system_prompt="Named.")
        orch.register("named", cfg)

        expected_result = _success_result()

        with patch.object(orch, "_build_subagent") as mock_build:
            mock_subagent = AsyncMock()
            mock_subagent.arun = AsyncMock(return_value=expected_result)
            mock_subagent.start = AsyncMock()
            mock_subagent.close = AsyncMock()
            mock_build.return_value = mock_subagent

            result = await orch.spawn("named", "task")

        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_spawn_unknown_name_raises(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        with pytest.raises(ValueError, match="Unknown subagent"):
            await orch.spawn("ghost", "task")

    @pytest.mark.asyncio
    async def test_spawn_closes_on_error(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        cfg = SubagentConfig(name="fail", system_prompt="Fail.")

        with patch.object(orch, "_build_subagent") as mock_build:
            mock_subagent = AsyncMock()
            mock_subagent.arun = AsyncMock(side_effect=RuntimeError("boom"))
            mock_subagent.start = AsyncMock()
            mock_subagent.close = AsyncMock()
            mock_build.return_value = mock_subagent

            with pytest.raises(RuntimeError, match="boom"):
                await orch.spawn(cfg, "task")

            mock_subagent.close.assert_awaited_once()


# ---------------------------------------------------------------------------
# 13.2  AgentOrchestrator — spawn_background / get_result
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorBackground:
    @pytest.mark.asyncio
    async def test_spawn_background_returns_task_id(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        cfg = SubagentConfig(name="bg", system_prompt="BG.")

        with patch.object(orch, "spawn", new_callable=AsyncMock) as mock_spawn:
            mock_spawn.return_value = _success_result()
            task_id = await orch.spawn_background(cfg, "background task")

        assert isinstance(task_id, str)
        assert len(task_id) > 0

    @pytest.mark.asyncio
    async def test_get_result_returns_none_while_running(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        cfg = SubagentConfig(name="slow", system_prompt="Slow.")

        # Make spawn hang until we cancel
        hang_event = asyncio.Event()

        async def slow_spawn(*args, **kwargs):
            await hang_event.wait()
            return _success_result()

        with patch.object(orch, "spawn", side_effect=slow_spawn):
            task_id = await orch.spawn_background(cfg, "slow task")
            result = await orch.get_result(task_id)
            assert result is None

            # Let it finish
            hang_event.set()
            await asyncio.sleep(0.05)

            result = await orch.get_result(task_id)
            assert result is not None
            assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_get_result_unknown_task_returns_none(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        assert await orch.get_result("nonexistent-id") is None

    @pytest.mark.asyncio
    async def test_spawn_background_handles_error(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        cfg = SubagentConfig(name="err", system_prompt="Err.")

        with patch.object(orch, "spawn", new_callable=AsyncMock) as mock_spawn:
            mock_spawn.side_effect = RuntimeError("crash")
            task_id = await orch.spawn_background(cfg, "will fail")
            await asyncio.sleep(0.05)

            result = await orch.get_result(task_id)
            assert result is not None
            assert result.status == "error"
            assert "crash" in result.error


# ---------------------------------------------------------------------------
# 13.2  AgentOrchestrator — handoff
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestOrchestratorHandoff:
    @pytest.mark.asyncio
    async def test_handoff_without_messages(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)

        target = AsyncMock()
        target.arun = AsyncMock(return_value=_success_result("handoff done"))

        result = await orch.handoff(target, "Review this code")
        target.arun.assert_awaited_once_with("Review this code")
        assert result.output == "handoff done"

    @pytest.mark.asyncio
    async def test_handoff_with_messages(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)

        target = MagicMock()
        target.registry = MagicMock()
        target.registry.tools = []
        target.registry.get_llm_schemas = MagicMock(return_value=[])
        target.agent_id = "target-001"
        target.max_iterations = 10

        expected = _success_result("with messages")
        target.runtime = MagicMock()
        target.runtime.run_with_state = AsyncMock(return_value=expected)

        messages = [Message.user("Hello"), Message.assistant("Hi there")]
        result = await orch.handoff(target, "Continue the conversation", parent_messages=messages)

        assert result.output == "with messages"
        target.runtime.run_with_state.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_handoff_with_empty_messages_calls_arun(self):
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)

        target = AsyncMock()
        target.arun = AsyncMock(return_value=_success_result())

        result = await orch.handoff(target, "context", parent_messages=[])
        target.arun.assert_awaited_once_with("context")


# ---------------------------------------------------------------------------
# 13.2  AgentOrchestrator — _build_subagent (inheritance flags)
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestBuildSubagent:
    def test_inherit_tools(self):
        """When inherit_tools=True, parent's tools are added to subagent."""
        parent_tool = MagicMock()
        parent_tool.name = "parent_tool"
        parent = _make_mock_agent()
        parent.registry.tools = [parent_tool]

        orch = AgentOrchestrator(parent)
        cfg = SubagentConfig(name="sub", system_prompt="Hi", inherit_tools=True)

        with patch("curio_agent_sdk.core.agent.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            orch._build_subagent(cfg)

            call_kwargs = MockAgent.call_args
            tools_arg = call_kwargs.kwargs.get("tools", [])
            assert parent_tool in tools_arg

    def test_inherit_memory(self):
        """When inherit_memory=True, parent's memory manager is passed."""
        parent = _make_mock_agent()
        parent.memory_manager_instance = MagicMock()

        orch = AgentOrchestrator(parent)
        cfg = SubagentConfig(name="sub", system_prompt="Hi", inherit_memory=True)

        with patch("curio_agent_sdk.core.agent.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            orch._build_subagent(cfg)

            call_kwargs = MockAgent.call_args
            mm_arg = call_kwargs.kwargs.get("memory_manager")
            assert mm_arg is parent.memory_manager_instance

    def test_inherit_hooks(self):
        """When inherit_hooks=True (default), parent's hook registry is passed."""
        parent = _make_mock_agent()
        parent.hook_registry = MagicMock()

        orch = AgentOrchestrator(parent)
        cfg = SubagentConfig(name="sub", system_prompt="Hi", inherit_hooks=True)

        with patch("curio_agent_sdk.core.agent.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            orch._build_subagent(cfg)

            call_kwargs = MockAgent.call_args
            hr_arg = call_kwargs.kwargs.get("hook_registry")
            assert hr_arg is parent.hook_registry

    def test_custom_model(self):
        """When config has a model, Agent is constructed with model= instead of llm=."""
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        cfg = SubagentConfig(name="sub", system_prompt="Hi", model="gpt-4")

        with patch("curio_agent_sdk.core.agent.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            orch._build_subagent(cfg)

            call_args = MockAgent.call_args
            # model is passed as first positional arg
            if call_args[0]:
                assert call_args[0][0] == "gpt-4"
            else:
                assert call_args.kwargs.get("model") == "gpt-4"

    def test_no_model_uses_parent_llm(self):
        """When config has no model, parent's llm is passed."""
        parent = _make_mock_agent()
        orch = AgentOrchestrator(parent)
        cfg = SubagentConfig(name="sub", system_prompt="Hi", model=None)

        with patch("curio_agent_sdk.core.agent.Agent") as MockAgent:
            MockAgent.return_value = MagicMock()
            orch._build_subagent(cfg)

            call_kwargs = MockAgent.call_args
            assert call_kwargs.kwargs.get("llm") is parent.llm
