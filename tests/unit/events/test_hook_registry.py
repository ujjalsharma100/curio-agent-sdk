"""
Unit tests for HookRegistry — on/off/emit, priority, sync/async handlers, error isolation.
"""

import pytest

from curio_agent_sdk.core.events.hooks import (
    HookRegistry,
    HookContext,
    AGENT_RUN_BEFORE,
    LLM_CALL_BEFORE,
    TOOL_CALL_BEFORE,
)


@pytest.mark.unit
class TestHookRegistry:
    @pytest.mark.asyncio
    async def test_register_sync_handler(self):
        reg = HookRegistry()
        seen = []

        def handler(ctx: HookContext):
            seen.append(ctx.event)

        reg.on(AGENT_RUN_BEFORE, handler)
        ctx = HookContext(event=AGENT_RUN_BEFORE)
        await reg.emit(AGENT_RUN_BEFORE, ctx)
        assert seen == [AGENT_RUN_BEFORE]

    @pytest.mark.asyncio
    async def test_register_async_handler(self):
        reg = HookRegistry()
        seen = []

        async def handler(ctx: HookContext):
            seen.append(ctx.event)

        reg.on(AGENT_RUN_BEFORE, handler)
        ctx = HookContext(event=AGENT_RUN_BEFORE)
        await reg.emit(AGENT_RUN_BEFORE, ctx)
        assert seen == [AGENT_RUN_BEFORE]

    @pytest.mark.asyncio
    async def test_emit_event(self):
        reg = HookRegistry()
        calls = []

        def h1(ctx: HookContext):
            calls.append("h1")

        reg.on(LLM_CALL_BEFORE, h1)
        ctx = HookContext(event=LLM_CALL_BEFORE)
        await reg.emit(LLM_CALL_BEFORE, ctx)
        assert calls == ["h1"]

    @pytest.mark.asyncio
    async def test_emit_no_handlers(self):
        reg = HookRegistry()
        ctx = HookContext(event="custom.unknown")
        result = await reg.emit("custom.unknown", ctx)
        assert result is ctx
        assert result.cancelled is False

    @pytest.mark.asyncio
    async def test_handler_priority(self):
        reg = HookRegistry()
        order = []

        def low(ctx: HookContext):
            order.append("low")

        def high(ctx: HookContext):
            order.append("high")

        reg.on(TOOL_CALL_BEFORE, low, priority=10)
        reg.on(TOOL_CALL_BEFORE, high, priority=0)
        ctx = HookContext(event=TOOL_CALL_BEFORE)
        await reg.emit(TOOL_CALL_BEFORE, ctx)
        assert order == ["high", "low"]

    @pytest.mark.asyncio
    async def test_multiple_handlers(self):
        reg = HookRegistry()
        count = [0]

        def h1(ctx: HookContext):
            count[0] += 1

        def h2(ctx: HookContext):
            count[0] += 10

        reg.on(AGENT_RUN_BEFORE, h1)
        reg.on(AGENT_RUN_BEFORE, h2)
        ctx = HookContext(event=AGENT_RUN_BEFORE)
        await reg.emit(AGENT_RUN_BEFORE, ctx)
        assert count[0] == 11

    @pytest.mark.asyncio
    async def test_remove_handler(self):
        reg = HookRegistry()
        seen = []

        def handler(ctx: HookContext):
            seen.append(1)

        reg.on(AGENT_RUN_BEFORE, handler)
        await reg.emit(AGENT_RUN_BEFORE, HookContext(event=AGENT_RUN_BEFORE))
        assert len(seen) == 1

        reg.off(AGENT_RUN_BEFORE, handler)
        await reg.emit(AGENT_RUN_BEFORE, HookContext(event=AGENT_RUN_BEFORE))
        assert len(seen) == 1

    @pytest.mark.asyncio
    async def test_remove_nonexistent_handler(self):
        reg = HookRegistry()

        def handler(ctx: HookContext):
            pass

        reg.off(AGENT_RUN_BEFORE, handler)
        ctx = HookContext(event=AGENT_RUN_BEFORE)
        result = await reg.emit(AGENT_RUN_BEFORE, ctx)
        assert result is ctx

    @pytest.mark.asyncio
    async def test_handler_error_isolation(self):
        reg = HookRegistry()
        other_called = []

        def failing(ctx: HookContext):
            raise ValueError("hook failed")

        def other(ctx: HookContext):
            other_called.append(True)

        reg.on(AGENT_RUN_BEFORE, failing)
        reg.on(AGENT_RUN_BEFORE, other)
        ctx = HookContext(event=AGENT_RUN_BEFORE)
        result = await reg.emit(AGENT_RUN_BEFORE, ctx)
        assert result is ctx
        assert other_called == [True]

    @pytest.mark.asyncio
    async def test_convenience_event_names(self):
        from curio_agent_sdk.core.events.hooks import (
            AGENT_RUN_AFTER,
            LLM_CALL_AFTER,
        )
        reg = HookRegistry()
        seen = []

        def h(ctx: HookContext):
            seen.append(ctx.event)

        reg.on(AGENT_RUN_AFTER, h)
        reg.on(LLM_CALL_AFTER, h)
        await reg.emit(AGENT_RUN_AFTER, HookContext(event=AGENT_RUN_AFTER))
        await reg.emit(LLM_CALL_AFTER, HookContext(event=LLM_CALL_AFTER))
        assert AGENT_RUN_AFTER in seen and LLM_CALL_AFTER in seen

    def test_load_hooks_from_config_creates_registry(self):
        from curio_agent_sdk.core.events.hooks import load_hooks_from_config
        reg = load_hooks_from_config([])
        assert isinstance(reg, HookRegistry)

    def test_load_hooks_from_config_with_hooks_key(self):
        from curio_agent_sdk.core.events.hooks import load_hooks_from_config
        reg = load_hooks_from_config({"hooks": []})
        assert isinstance(reg, HookRegistry)
