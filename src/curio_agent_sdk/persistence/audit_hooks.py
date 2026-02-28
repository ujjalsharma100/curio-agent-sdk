"""
Audit logging helpers wired into the HookRegistry event flow.

This module provides a small integration layer that listens to core lifecycle
hooks (agent runs, LLM calls, tool calls) and emits structured audit events
to a persistence backend that implements `BasePersistence`.

Usage:
    from curio_agent_sdk.persistence.audit_hooks import register_audit_hooks
    from curio_agent_sdk.persistence.sqlite import SQLitePersistence

    persistence = SQLitePersistence("agent_sdk.db")
    hooks = HookRegistry()
    register_audit_hooks(hooks, persistence)
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from curio_agent_sdk.core.events import (
    HookRegistry,
    HookContext,
    AGENT_RUN_BEFORE,
    AGENT_RUN_AFTER,
    AGENT_RUN_ERROR,
    TOOL_CALL_BEFORE,
    TOOL_CALL_AFTER,
    TOOL_CALL_ERROR,
    LLM_CALL_BEFORE,
    LLM_CALL_AFTER,
    LLM_CALL_ERROR,
)
from curio_agent_sdk.persistence.base import BasePersistence


def _base_event(ctx: HookContext) -> dict[str, Any]:
    """Common fields for all audit events."""
    return {
        "agent_id": ctx.agent_id or None,
        "run_id": ctx.run_id or None,
        "timestamp": datetime.utcnow(),
        "metadata": dict(ctx.data or {}),
    }


def register_audit_hooks(registry: HookRegistry, persistence: BasePersistence) -> None:
    """
    Register audit log handlers on the given HookRegistry.

    The handlers emit high-level audit events such as:
    - agent_run_started / agent_run_completed / agent_run_error
    - tool_call_before / tool_call_after / tool_call_error
    - llm_call_before / llm_call_after / llm_call_error
    """

    async def _on_agent_run_before(ctx: HookContext) -> None:
        await persistence.alog_audit_event(
            {
                **_base_event(ctx),
                "actor_type": "user",
                "actor_id": ctx.data.get("user_id"),
                "action": "agent_run_started",
                "resource": ctx.data.get("input"),
                "resource_type": "agent_input",
            }
        )

    async def _on_agent_run_after(ctx: HookContext) -> None:
        await persistence.alog_audit_event(
            {
                **_base_event(ctx),
                "actor_type": "agent",
                "actor_id": ctx.agent_id or None,
                "action": "agent_run_completed",
                "resource": None,
                "resource_type": "agent_run",
            }
        )

    async def _on_agent_run_error(ctx: HookContext) -> None:
        await persistence.alog_audit_event(
            {
                **_base_event(ctx),
                "actor_type": "agent",
                "actor_id": ctx.agent_id or None,
                "action": "agent_run_error",
                "resource": ctx.data.get("error"),
                "resource_type": "agent_error",
            }
        )

    async def _on_tool_before(ctx: HookContext) -> None:
        await persistence.alog_audit_event(
            {
                **_base_event(ctx),
                "actor_type": "agent",
                "actor_id": ctx.agent_id or None,
                "action": "tool_call_before",
                "resource": ctx.data.get("tool_name") or ctx.data.get("tool"),
                "resource_type": "tool",
            }
        )

    async def _on_tool_after(ctx: HookContext) -> None:
        await persistence.alog_audit_event(
            {
                **_base_event(ctx),
                "actor_type": "agent",
                "actor_id": ctx.agent_id or None,
                "action": "tool_call_after",
                "resource": ctx.data.get("tool_name"),
                "resource_type": "tool",
            }
        )

    async def _on_tool_error(ctx: HookContext) -> None:
        await persistence.alog_audit_event(
            {
                **_base_event(ctx),
                "actor_type": "agent",
                "actor_id": ctx.agent_id or None,
                "action": "tool_call_error",
                "resource": ctx.data.get("tool_name"),
                "resource_type": "tool",
            }
        )

    async def _on_llm_before(ctx: HookContext) -> None:
        await persistence.alog_audit_event(
            {
                **_base_event(ctx),
                "actor_type": "agent",
                "actor_id": ctx.agent_id or None,
                "action": "llm_call_before",
                "resource": None,
                "resource_type": "llm",
            }
        )

    async def _on_llm_after(ctx: HookContext) -> None:
        await persistence.alog_audit_event(
            {
                **_base_event(ctx),
                "actor_type": "agent",
                "actor_id": ctx.agent_id or None,
                "action": "llm_call_after",
                "resource": None,
                "resource_type": "llm",
            }
        )

    async def _on_llm_error(ctx: HookContext) -> None:
        await persistence.alog_audit_event(
            {
                **_base_event(ctx),
                "actor_type": "agent",
                "actor_id": ctx.agent_id or None,
                "action": "llm_call_error",
                "resource": ctx.data.get("error"),
                "resource_type": "llm",
            }
        )

    # Agent run lifecycle
    registry.on(AGENT_RUN_BEFORE, _on_agent_run_before, priority=100)
    registry.on(AGENT_RUN_AFTER, _on_agent_run_after, priority=100)
    registry.on(AGENT_RUN_ERROR, _on_agent_run_error, priority=100)

    # Tool calls
    registry.on(TOOL_CALL_BEFORE, _on_tool_before, priority=100)
    registry.on(TOOL_CALL_AFTER, _on_tool_after, priority=100)
    registry.on(TOOL_CALL_ERROR, _on_tool_error, priority=100)

    # LLM calls
    registry.on(LLM_CALL_BEFORE, _on_llm_before, priority=100)
    registry.on(LLM_CALL_AFTER, _on_llm_after, priority=100)
    registry.on(LLM_CALL_ERROR, _on_llm_error, priority=100)

