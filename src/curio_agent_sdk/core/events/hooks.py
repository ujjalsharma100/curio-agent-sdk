"""
Hooks / lifecycle system for agent customization.

Hooks let users register callbacks that run at specific lifecycle events
and can mutate behavior (cancel actions, modify request/response, inject context).

Design:
- HookRegistry is the central registry; handlers run in priority order (lower = earlier).
- HookContext is mutable; handlers can cancel the action or modify context.data.
- Sync and async handlers are supported.
- Event names follow a dotted convention (e.g. agent.run.before, llm.call.after).
"""

from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable, TYPE_CHECKING

if TYPE_CHECKING:
    from curio_agent_sdk.core.state import AgentState

logger = logging.getLogger(__name__)

# Hook event names (string-based for extensibility; custom.* for user events)
AGENT_RUN_BEFORE = "agent.run.before"
AGENT_RUN_AFTER = "agent.run.after"
AGENT_RUN_ERROR = "agent.run.error"

AGENT_ITERATION_BEFORE = "agent.iteration.before"
AGENT_ITERATION_AFTER = "agent.iteration.after"

LLM_CALL_BEFORE = "llm.call.before"
LLM_CALL_AFTER = "llm.call.after"
LLM_CALL_ERROR = "llm.call.error"

TOOL_CALL_BEFORE = "tool.call.before"
TOOL_CALL_AFTER = "tool.call.after"
TOOL_CALL_ERROR = "tool.call.error"

MEMORY_INJECT_BEFORE = "memory.inject.before"
MEMORY_SAVE_BEFORE = "memory.save.before"
MEMORY_QUERY_BEFORE = "memory.query.before"

STATE_CHECKPOINT_BEFORE = "state.checkpoint.before"
STATE_CHECKPOINT_AFTER = "state.checkpoint.after"

HOOK_EVENTS = [
    AGENT_RUN_BEFORE,
    AGENT_RUN_AFTER,
    AGENT_RUN_ERROR,
    AGENT_ITERATION_BEFORE,
    AGENT_ITERATION_AFTER,
    LLM_CALL_BEFORE,
    LLM_CALL_AFTER,
    LLM_CALL_ERROR,
    TOOL_CALL_BEFORE,
    TOOL_CALL_AFTER,
    TOOL_CALL_ERROR,
    MEMORY_INJECT_BEFORE,
    MEMORY_SAVE_BEFORE,
    MEMORY_QUERY_BEFORE,
    STATE_CHECKPOINT_BEFORE,
    STATE_CHECKPOINT_AFTER,
]


@dataclass
class HookContext:
    """
    Mutable context passed through the hook chain.

    Handlers can:
    - ctx.cancel() to cancel the action (e.g. block a tool call)
    - ctx.modify(key, value) to update context.data for downstream handlers / caller
    """

    event: str
    data: dict[str, Any] = field(default_factory=dict)
    state: AgentState | None = None
    run_id: str = ""
    agent_id: str = ""
    iteration: int = 0
    cancelled: bool = False

    def cancel(self) -> None:
        """Mark the action as cancelled (e.g. skip tool call, block iteration)."""
        self.cancelled = True

    def modify(self, key: str, value: Any) -> None:
        """Update a value in context.data (e.g. modify request, args, result)."""
        self.data[key] = value


Handler = Callable[[HookContext], Any] | Callable[[HookContext], Awaitable[Any]]


def _is_async_callable(fn: Handler) -> bool:
    if asyncio.iscoroutinefunction(fn):
        return True
    if callable(fn) and hasattr(fn, "__call__"):
        return asyncio.iscoroutinefunction(getattr(fn, "__call__", None))
    return False


class HookRegistry:
    """
    Central registry for lifecycle hooks.

    Handlers are called in priority order (lower number = higher priority).
    Sync and async handlers are supported.
    """

    def __init__(self) -> None:
        # event -> list of (priority, handler)
        self._handlers: dict[str, list[tuple[int, Handler]]] = defaultdict(list)

    def on(self, event: str, handler: Handler, *, priority: int = 0) -> None:
        """Register a handler for an event. Lower priority runs first."""
        self._handlers[event].append((priority, handler))
        self._handlers[event].sort(key=lambda x: x[0])

    def off(self, event: str, handler: Handler) -> None:
        """Unregister a handler for an event."""
        self._handlers[event] = [(p, h) for p, h in self._handlers[event] if h is not handler]

    async def emit(self, event: str, context: HookContext) -> HookContext:
        """
        Run all handlers for the event in priority order.

        If a handler sets context.cancelled, remaining handlers still run
        (so e.g. logging hooks still see the event), but the caller should
        check context.cancelled and abort the action if True.

        Returns the same context (possibly mutated).
        """
        for _priority, handler in self._handlers[event]:
            try:
                if _is_async_callable(handler):
                    await handler(context)
                else:
                    handler(context)
            except Exception as e:
                logger.exception("Hook %r for event %r failed: %s", handler, event, e)
        return context


def run_shell_hook(shell_command: str, context: HookContext) -> None:
    """
    Run a shell command as a hook (e.g. from config).

    The command is run with env vars: HOOK_EVENT, HOOK_RUN_ID, HOOK_AGENT_ID.
    Use for non-Python integrations (e.g. "echo 'Agent completed' >> /tmp/agent.log").
    """
    import os
    import subprocess
    env = os.environ.copy()
    env["HOOK_EVENT"] = context.event
    env["HOOK_RUN_ID"] = context.run_id
    env["HOOK_AGENT_ID"] = context.agent_id
    subprocess.run(
        shell_command,
        shell=True,
        env=env,
        capture_output=True,
        timeout=30,
    )


def _resolve_handler(handler_ref: str) -> Handler:
    """Resolve 'module:callable' or 'module:Class.method' to a callable."""
    if ":" not in handler_ref:
        raise ValueError(f"Handler must be 'module:callable', got {handler_ref!r}")
    mod_path, name = handler_ref.strip().rsplit(":", 1)
    import importlib
    mod = importlib.import_module(mod_path)
    obj = getattr(mod, name)
    if not callable(obj):
        raise ValueError(f"{handler_ref!r} is not callable")
    return obj


def load_hooks_from_config(
    config: dict[str, Any] | list[dict[str, Any]],
    registry: HookRegistry | None = None,
) -> HookRegistry:
    """
    Load hooks from a config structure (e.g. from YAML/TOML).

    Config format (list of hook entries):
        - event: tool.call.before
          handler: my_module:my_function
          priority: 0
        - event: agent.run.after
          shell: "echo 'done' >> /tmp/agent.log"
          priority: 10

    Either "handler" (module:callable string) or "shell" (shell command) per entry.
    Returns the registry (creates one if not provided).
    """
    if registry is None:
        registry = HookRegistry()
    entries = config if isinstance(config, list) else config.get("hooks", config.get("hook", []))
    if not isinstance(entries, list):
        entries = [entries]
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        event = entry.get("event")
        if not event:
            continue
        priority = int(entry.get("priority", 0))
        if "shell" in entry:
            cmd = entry["shell"]
            registry.on(event, lambda ctx, c=cmd: run_shell_hook(c, ctx), priority=priority)
        elif "handler" in entry:
            handler = _resolve_handler(entry["handler"])
            registry.on(event, handler, priority=priority)
    return registry


def load_hooks_from_file(path: str | bytes, registry: HookRegistry | None = None) -> HookRegistry:
    """
    Load hooks from a YAML or TOML file.

    Example agent.hooks.yaml:
        - event: tool.call.before
          handler: my_app:hooks.validate_tool
        - event: agent.run.after
          shell: "echo Done >> /tmp/agent.log"
    """
    from pathlib import Path
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    raw = p.read_text()
    if p.suffix in (".yaml", ".yml"):
        try:
            import yaml
            config = yaml.safe_load(raw)
        except ImportError:
            raise ImportError("PyYAML required for YAML hook config: pip install pyyaml")
    elif p.suffix == ".toml":
        try:
            import tomllib
            with open(p, "rb") as f:
                config = tomllib.load(f)
        except ImportError:
            try:
                import toml
                config = toml.loads(raw)
            except ImportError:
                raise ImportError("toml or tomllib required for TOML hook config")
    else:
        raise ValueError("Hook config must be .yaml, .yml, or .toml")
    return load_hooks_from_config(config or [], registry)
