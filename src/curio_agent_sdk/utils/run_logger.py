"""
Optional run logger — writes full agent run execution to a .log file.

Use for debugging and audit: run start/end, every LLM request (messages, model, tools),
every LLM response (content, tool calls, usage), and every tool call (name, args, result).
Opt-in only; no effect when not attached.

Example:
    from curio_agent_sdk import Agent, use_run_logger

    builder = Agent.builder().model("openai:gpt-4o-mini").tools([...])
    logger = use_run_logger(builder, base_name="my-run", output_dir="./logs")
    agent = builder.build()
    result = await agent.arun("Hello")
    print("Log file:", logger.get_log_path())
"""

from __future__ import annotations

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any

SEP = "\n" + "=" * 80 + "\n"


def _safe_str(obj: Any) -> str:
    if obj is None:
        return "<none>"
    if isinstance(obj, str):
        return obj
    try:
        return json.dumps(obj, indent=2)
    except Exception:
        return str(obj)


def _val(v: Any, default: str = "—") -> str:
    """Format value for log; use default only when v is None (so 0 or False are shown)."""
    if v is None:
        return default
    return str(v)


def _usage_for_log(usage: Any) -> str:
    """Deconstruct usage into TypeScript-style JSON: promptTokens, completionTokens, totalTokens (+ cache if present)."""
    if usage is None:
        return "—"
    inp = getattr(usage, "input_tokens", None) or getattr(usage, "prompt_tokens", 0) or 0
    out = getattr(usage, "output_tokens", None) or getattr(usage, "completion_tokens", 0) or 0
    total = getattr(usage, "total_tokens", None)
    if total is None and (inp is not None or out is not None):
        total = (inp or 0) + (out or 0)
    d: dict[str, Any] = {
        "promptTokens": inp,
        "completionTokens": out,
        "totalTokens": total or 0,
    }
    cache_read = getattr(usage, "cache_read_tokens", None) or getattr(usage, "cacheReadTokens", None)
    cache_write = getattr(usage, "cache_write_tokens", None) or getattr(usage, "cacheWriteTokens", None)
    if cache_read is not None and cache_read != 0:
        d["cacheReadTokens"] = cache_read
    if cache_write is not None and cache_write != 0:
        d["cacheWriteTokens"] = cache_write
    return json.dumps(d, indent=2)


def _messages_for_log(messages: Any) -> list[dict[str, Any]]:
    """Serialize message list for log (role, content, tool_calls if present)."""
    out: list[dict[str, Any]] = []
    for m in messages or []:
        entry: dict[str, Any] = {"role": getattr(m, "role", "user")}
        content = getattr(m, "content", None)
        if isinstance(content, str):
            entry["content"] = content
        elif content is not None:
            entry["content"] = content
        if getattr(m, "tool_calls", None):
            entry["toolCalls"] = [
                {"id": getattr(tc, "id", ""), "name": getattr(tc, "name", ""), "arguments": getattr(tc, "arguments", {})}
                for tc in (m.tool_calls or [])
            ]
        if getattr(m, "tool_call_id", None):
            entry["tool_call_id"] = m.tool_call_id
        if getattr(m, "name", None):
            entry["name"] = m.name
        out.append(entry)
    return out


def _tools_for_log(tools: Any) -> list[dict[str, Any]]:
    """Serialize tools (LLM schema) for log: name, description, parameters."""
    out: list[dict[str, Any]] = []
    for t in tools or []:
        if isinstance(t, dict):
            out.append(t)
        elif hasattr(t, "to_openai_format"):
            fn = t.to_openai_format().get("function", {})
            out.append(fn if isinstance(fn, dict) else {"name": getattr(t, "name", ""), "description": getattr(t, "description", ""), "parameters": getattr(t, "parameters", {})})
        else:
            out.append({"name": getattr(t, "name", ""), "description": getattr(t, "description", ""), "parameters": getattr(t, "parameters", {})})
    return out


class RunLogger:
    """Writes granular run logs to a timestamped file. Use with :func:`use_run_logger` or :func:`create_run_logger`."""

    def __init__(
        self,
        output_dir: str | None = None,
        base_name: str = "agent-run",
    ):
        self.output_dir = output_dir or os.getcwd()
        self.base_name = base_name
        self._log_path: str | None = None
        self._last_response_model: str | None = None

    def _ensure_path(self) -> str:
        if self._log_path is not None:
            return self._log_path
        ts = datetime.utcnow().isoformat()[:19].replace(":", "-").replace(".", "-")
        self._log_path = str(Path(self.output_dir) / f"{self.base_name}-{ts}.log")
        return self._log_path

    def _write(self, text: str) -> None:
        path = self._ensure_path()
        with open(path, "a", encoding="utf-8") as f:
            f.write(text)

    def get_log_path(self) -> str | None:
        """Return the path to the log file once the first event has been written; ``None`` before that."""
        return self._log_path

    async def on_run_before(self, ctx: Any) -> None:
        self._ensure_path()
        input_text = ctx.data.get("input", "")
        self._write(
            f"[AGENT RUN START] {datetime.utcnow().isoformat()}\n"
            f"run_id: {ctx.run_id or '—'}\nagent_id: {ctx.agent_id or '—'}\n"
            f"input: {_safe_str(input_text)}\n{SEP}"
        )

    async def on_run_after(self, ctx: Any) -> None:
        if self._log_path is None:
            return
        output = ctx.data.get("output")
        result = ctx.data.get("result")
        if output is None and isinstance(result, dict):
            output = result.get("output")
        state = ctx.data.get("state")
        if output is None and state is not None:
            for msg in reversed(getattr(state, "messages", [])):
                if getattr(msg, "role", None) == "assistant":
                    text = getattr(msg, "text", None)
                    if text is None and hasattr(msg, "content"):
                        text = msg.content if isinstance(msg.content, str) else ""
                    if text:
                        output = text
                        break
        tool_calls_count = 0
        if isinstance(result, dict) and "tool_calls" in result:
            tool_calls_count = result["tool_calls"]
        elif state is not None:
            tool_calls_count = getattr(state, "total_tool_calls", 0)
        self._write(
            f"[AGENT RUN END] {datetime.utcnow().isoformat()}\n"
            f"output: {_safe_str(output) if output is not None else '<none>'}\n"
            f"toolCalls count: {tool_calls_count}\n{SEP}"
        )

    async def on_run_error(self, ctx: Any) -> None:
        if self._log_path is None:
            return
        err = ctx.data.get("error", "")
        self._write(
            f"[AGENT RUN ERROR] {datetime.utcnow().isoformat()}\n"
            f"error: {_safe_str(err)}\n{SEP}"
        )

    async def on_llm_before(self, ctx: Any) -> None:
        self._ensure_path()
        req = ctx.data.get("request")
        if not req:
            return
        messages = getattr(req, "messages", [])
        tools = getattr(req, "tools", None) or []
        model_str = ctx.data.get("model_for_log") or getattr(req, "model", None) or self._last_response_model
        iteration_str = _val(getattr(ctx, "iteration", None))
        self._write(
            f"[LLM REQUEST] {datetime.utcnow().isoformat()} iteration: {iteration_str}\n"
            f"model: {_val(model_str)}\n"
            f"temperature: {_val(getattr(req, 'temperature', None))} maxTokens: {_val(getattr(req, 'max_tokens', None))}\n"
            f"messages ({len(messages)}):\n{_safe_str(_messages_for_log(messages))}\n"
            f"tools ({len(tools)}):\n{_safe_str(_tools_for_log(tools))}\n{SEP}"
        )

    async def on_llm_after(self, ctx: Any) -> None:
        if self._log_path is None:
            return
        req = ctx.data.get("request")
        resp = ctx.data.get("response")
        duration_ms = ctx.data.get("duration")
        if not resp:
            return
        self._last_response_model = getattr(resp, "model", None) or getattr(req, "model", None)
        content = getattr(resp.message, "text", "") or (resp.message.content if isinstance(getattr(resp.message, "content", None), str) else "")
        self._write(
            f"[LLM RESPONSE] {datetime.utcnow().isoformat()}\n"
            f"model: {_val(self._last_response_model)} durationMs: {_val(duration_ms)}\n"
            f"content: {_safe_str(content)}\n"
            f"toolCalls: {len(resp.tool_calls)}\n"
        )
        for tc in resp.tool_calls:
            self._write(f"  - {tc.name} id={tc.id} args={_safe_str(tc.arguments)}\n")
        if getattr(resp, "usage", None):
            self._write(f"usage: {_usage_for_log(resp.usage)}\n")
        self._write(SEP)

    async def on_llm_error(self, ctx: Any) -> None:
        if self._log_path is None:
            return
        err = ctx.data.get("error", "")
        self._write(f"[LLM ERROR] {datetime.utcnow().isoformat()}\nerror: {_safe_str(err)}\n{SEP}")

    async def on_tool_before(self, ctx: Any) -> None:
        self._ensure_path()
        name = ctx.data.get("tool_name") or ctx.data.get("tool", "—")
        args = ctx.data.get("args", {})
        tid = _val(ctx.data.get("tool_call_id"))
        self._write(
            f"[TOOL CALL START] {datetime.utcnow().isoformat()}\n"
            f"tool: {name} toolCallId: {tid}\n"
            f"arguments: {_safe_str(args)}\n{SEP}"
        )

    async def on_tool_after(self, ctx: Any) -> None:
        if self._log_path is None:
            return
        name = ctx.data.get("tool_name", "—")
        args = ctx.data.get("args", {})
        result = ctx.data.get("result", "")
        tid = _val(ctx.data.get("tool_call_id"))
        duration_ms = ctx.data.get("duration")
        self._write(
            f"[TOOL CALL END] {datetime.utcnow().isoformat()}\n"
            f"tool: {name} toolCallId: {tid} durationMs: {_val(duration_ms)}\n"
            f"arguments: {_safe_str(args)}\n"
            f"result: {_safe_str(result)}\n{SEP}"
        )

    async def on_tool_error(self, ctx: Any) -> None:
        if self._log_path is None:
            return
        name = ctx.data.get("tool_name") or ctx.data.get("tool", "—")
        args = ctx.data.get("args", {})
        err = ctx.data.get("error", "")
        self._write(
            f"[TOOL CALL ERROR] {datetime.utcnow().isoformat()}\n"
            f"tool: {name}\narguments: {_safe_str(args)}\nerror: {_safe_str(err)}\n{SEP}"
        )


def create_run_logger(
    *,
    output_dir: str | None = None,
    base_name: str = "agent-run",
) -> RunLogger:
    """Create a run logger instance. Register its handlers with :func:`use_run_logger` or attach to a :class:`HookRegistry` yourself."""
    return RunLogger(output_dir=output_dir, base_name=base_name)


def use_run_logger(
    builder: Any,
    base_name: str = "agent-run",
    output_dir: str | None = None,
) -> RunLogger:
    """Register run logger hooks on the builder. Returns the logger so you can call :meth:`RunLogger.get_log_path` after the run."""
    from curio_agent_sdk.core.events import (
        AGENT_RUN_AFTER,
        AGENT_RUN_BEFORE,
        AGENT_RUN_ERROR,
        LLM_CALL_AFTER,
        LLM_CALL_BEFORE,
        LLM_CALL_ERROR,
        TOOL_CALL_AFTER,
        TOOL_CALL_BEFORE,
        TOOL_CALL_ERROR,
    )
    logger = RunLogger(output_dir=output_dir, base_name=base_name)
    builder.hook(AGENT_RUN_BEFORE, logger.on_run_before)
    builder.hook(AGENT_RUN_AFTER, logger.on_run_after)
    builder.hook(AGENT_RUN_ERROR, logger.on_run_error)
    builder.hook(LLM_CALL_BEFORE, logger.on_llm_before)
    builder.hook(LLM_CALL_AFTER, logger.on_llm_after)
    builder.hook(LLM_CALL_ERROR, logger.on_llm_error)
    builder.hook(TOOL_CALL_BEFORE, logger.on_tool_before)
    builder.hook(TOOL_CALL_AFTER, logger.on_tool_after)
    builder.hook(TOOL_CALL_ERROR, logger.on_tool_error)
    return logger
