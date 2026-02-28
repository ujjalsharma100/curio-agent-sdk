"""
Unit tests for curio_agent_sdk.base.component

Covers: Component ABC lifecycle (startup, shutdown, health_check)
"""

import pytest
from abc import ABC

from curio_agent_sdk.base.component import Component


# ===================================================================
# Component ABC
# ===================================================================


class TestComponent:
    def test_is_abstract(self):
        """Component is an ABC."""
        assert issubclass(Component, ABC)

    def test_can_subclass(self):
        """Can create a concrete subclass."""

        class MyComponent(Component):
            pass

        comp = MyComponent()
        assert isinstance(comp, Component)

    @pytest.mark.asyncio
    async def test_startup_default(self):
        """Default startup() does nothing (no error)."""

        class MyComp(Component):
            pass

        comp = MyComp()
        await comp.startup()  # should not raise

    @pytest.mark.asyncio
    async def test_shutdown_default(self):
        """Default shutdown() does nothing (no error)."""

        class MyComp(Component):
            pass

        comp = MyComp()
        await comp.shutdown()  # should not raise

    @pytest.mark.asyncio
    async def test_health_check_default(self):
        """Default health_check() returns True."""

        class MyComp(Component):
            pass

        comp = MyComp()
        result = await comp.health_check()
        assert result is True

    @pytest.mark.asyncio
    async def test_custom_startup(self):
        """Subclass can override startup()."""

        class MyComp(Component):
            started = False

            async def startup(self) -> None:
                self.started = True

        comp = MyComp()
        assert comp.started is False
        await comp.startup()
        assert comp.started is True

    @pytest.mark.asyncio
    async def test_custom_shutdown(self):
        """Subclass can override shutdown()."""

        class MyComp(Component):
            closed = False

            async def shutdown(self) -> None:
                self.closed = True

        comp = MyComp()
        await comp.shutdown()
        assert comp.closed is True

    @pytest.mark.asyncio
    async def test_custom_health_check(self):
        """Subclass can override health_check()."""

        class MyComp(Component):
            healthy = False

            async def health_check(self) -> bool:
                return self.healthy

        comp = MyComp()
        assert await comp.health_check() is False
        comp.healthy = True
        assert await comp.health_check() is True

    @pytest.mark.asyncio
    async def test_lifecycle_order(self):
        """startup → use → shutdown runs in order."""
        order: list[str] = []

        class MyComp(Component):
            async def startup(self) -> None:
                order.append("startup")

            async def shutdown(self) -> None:
                order.append("shutdown")

        comp = MyComp()
        await comp.startup()
        order.append("use")
        await comp.shutdown()
        assert order == ["startup", "use", "shutdown"]

    @pytest.mark.asyncio
    async def test_startup_idempotent(self):
        """Multiple startup() calls should be safe."""
        call_count = 0

        class MyComp(Component):
            async def startup(self) -> None:
                nonlocal call_count
                call_count += 1

        comp = MyComp()
        await comp.startup()
        await comp.startup()
        assert call_count == 2  # called twice, both succeed

    @pytest.mark.asyncio
    async def test_shutdown_idempotent(self):
        """Multiple shutdown() calls should be safe."""
        call_count = 0

        class MyComp(Component):
            async def shutdown(self) -> None:
                nonlocal call_count
                call_count += 1

        comp = MyComp()
        await comp.shutdown()
        await comp.shutdown()
        assert call_count == 2
