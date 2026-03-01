"""
Unit tests for EventBus, InMemoryEventBus — publish, subscribe, pattern, replay, dead letters.
"""

import pytest
from datetime import datetime

from curio_agent_sdk.core.events.event_bus import (
    EventBus,
    InMemoryEventBus,
    EventFilter,
    DeadLetterEntry,
    EventBusBridge,
)
from curio_agent_sdk.core.events.hooks import HookRegistry, HookContext, AGENT_RUN_BEFORE
from curio_agent_sdk.models.events import AgentEvent, EventType


@pytest.mark.unit
class TestEventFilter:
    def test_filter_star_matches_all(self):
        f = EventFilter("*")
        assert f.matches("agent.run.before") is True
        assert f.matches("llm.call.after") is True

    def test_filter_agent_star(self):
        f = EventFilter("agent.*")
        assert f.matches("agent.run.before") is True
        assert f.matches("agent.iteration.after") is True
        assert f.matches("llm.call.after") is False

    def test_filter_exact(self):
        f = EventFilter("llm.call.after")
        assert f.matches("llm.call.after") is True
        assert f.matches("llm.call.before") is False

    def test_filter_repr(self):
        f = EventFilter("tool.*")
        assert "tool.*" in repr(f)


@pytest.mark.unit
class TestInMemoryEventBus:
    @pytest.mark.asyncio
    async def test_publish_event(self):
        bus = InMemoryEventBus(max_history=100)
        await bus.startup()
        received = []

        def handler(ev: AgentEvent):
            received.append(ev)

        await bus.subscribe("*", handler)
        event = AgentEvent(type=EventType.RUN_STARTED, run_id="r1", agent_id="a1")
        await bus.publish(event)
        assert len(received) == 1
        assert received[0].run_id == "r1"
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_subscribe_handler(self):
        bus = InMemoryEventBus()
        await bus.startup()
        count = [0]

        def handler(ev: AgentEvent):
            count[0] += 1

        await bus.subscribe("*", handler)
        await bus.publish(AgentEvent(type=EventType.LLM_CALL_COMPLETED))
        await bus.publish(AgentEvent(type=EventType.TOOL_CALL_COMPLETED))
        assert count[0] == 2
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_subscribe_pattern(self):
        bus = InMemoryEventBus()
        await bus.startup()
        agent_events = []
        llm_events = []

        def on_agent(ev: AgentEvent):
            agent_events.append(ev)

        def on_llm(ev: AgentEvent):
            llm_events.append(ev)

        await bus.subscribe("agent.*", on_agent)
        await bus.subscribe("llm.call.*", on_llm)
        await bus.publish(AgentEvent(type=EventType.RUN_STARTED))
        await bus.publish(AgentEvent(type=EventType.LLM_CALL_COMPLETED))
        await bus.publish(AgentEvent(type=EventType.TOOL_CALL_STARTED))
        assert len(agent_events) == 1
        assert len(llm_events) == 1
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_unsubscribe(self):
        bus = InMemoryEventBus()
        await bus.startup()
        count = [0]

        def handler(ev: AgentEvent):
            count[0] += 1

        await bus.subscribe("*", handler)
        await bus.publish(AgentEvent(type=EventType.RUN_STARTED))
        await bus.unsubscribe("*", handler)
        await bus.publish(AgentEvent(type=EventType.RUN_COMPLETED))
        assert count[0] == 1
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_publish_triggers_subscriber(self):
        bus = InMemoryEventBus()
        await bus.startup()
        payload = []

        async def async_handler(ev: AgentEvent):
            payload.append(ev.type)

        await bus.subscribe("*", async_handler)
        await bus.publish(AgentEvent(type=EventType.ITERATION_STARTED, data={"x": 1}))
        assert payload == [EventType.ITERATION_STARTED]
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_replay_events(self):
        bus = InMemoryEventBus(max_history=100)
        await bus.startup()
        t0 = datetime.now().timestamp()
        await bus.publish(AgentEvent(type=EventType.RUN_STARTED, run_id="r1"))
        await bus.publish(AgentEvent(type=EventType.LLM_CALL_STARTED, run_id="r1"))
        await bus.publish(AgentEvent(type=EventType.TOOL_CALL_STARTED, run_id="r1"))
        replayed = []
        async for ev in bus.replay(t0, pattern="*"):
            replayed.append(ev.type)
        assert EventType.RUN_STARTED in replayed
        assert EventType.LLM_CALL_STARTED in replayed
        assert EventType.TOOL_CALL_STARTED in replayed
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_dead_letter_queue(self):
        bus = InMemoryEventBus(max_history=100, max_dead_letters=10)
        await bus.startup()

        def failing_handler(ev: AgentEvent):
            raise RuntimeError("handler failed")

        await bus.subscribe("*", failing_handler)
        await bus.publish(AgentEvent(type=EventType.RUN_STARTED))
        assert len(bus.dead_letters) == 1
        dle = bus.dead_letters[0]
        assert dle.error == "handler failed"
        assert dle.event.type == EventType.RUN_STARTED
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_in_memory_bus_lifecycle(self):
        bus = InMemoryEventBus()
        assert await bus.health_check() is False
        await bus.startup()
        assert await bus.health_check() is True
        await bus.shutdown()
        assert await bus.health_check() is False

    @pytest.mark.asyncio
    async def test_clear_history_and_dead_letters(self):
        bus = InMemoryEventBus(max_history=10, max_dead_letters=10)
        await bus.startup()

        def fail(_: AgentEvent):
            raise ValueError("oops")

        await bus.subscribe("*", fail)
        await bus.publish(AgentEvent(type=EventType.RUN_STARTED))
        assert len(bus.dead_letters) == 1
        bus.clear_dead_letters()
        assert len(bus.dead_letters) == 0
        bus.clear_history()
        replayed = list()
        async for _ in bus.replay(0, "*"):
            replayed.append(_)
        assert len(replayed) == 0
        await bus.shutdown()

    @pytest.mark.asyncio
    async def test_event_bus_bridge_forwards_hooks_to_bus(self):
        bus = InMemoryEventBus(max_history=100)
        registry = HookRegistry()
        bridge = EventBusBridge(bus=bus, hook_registry=registry)
        await bridge.startup()
        received = []

        async def collect(ev: AgentEvent):
            received.append(ev)

        await bus.subscribe("*", collect)
        ctx = HookContext(event=AGENT_RUN_BEFORE, run_id="run1", agent_id="agent1")
        await registry.emit(AGENT_RUN_BEFORE, ctx)
        assert len(received) == 1
        assert received[0].run_id == "run1"
        assert received[0].agent_id == "agent1"
        await bridge.shutdown()
