"""
Hook-based observability consumers for the Curio Agent SDK.

Consumers listen to HookRegistry events instead of wrapping LLM calls via
middleware. This is the preferred approach for new code — it avoids the
``id(request)`` correlation hack and produces cleaner trace graphs.

Available consumers:
- TracingConsumer: OpenTelemetry spans + metrics via hook events.
- LoggingConsumer: Structured logging with optional trace correlation.
- PersistenceConsumer: Writes AgentRunEvent / AgentLLMUsage to BasePersistence.

Each consumer has ``attach(registry)`` / ``detach(registry)`` methods.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime
from typing import Any, TYPE_CHECKING

from curio_agent_sdk.core.events import (
    HookContext,
    HookRegistry,
    LLM_CALL_BEFORE,
    LLM_CALL_AFTER,
    LLM_CALL_ERROR,
    TOOL_CALL_BEFORE,
    TOOL_CALL_AFTER,
    TOOL_CALL_ERROR,
    AGENT_RUN_BEFORE,
    AGENT_RUN_AFTER,
    AGENT_RUN_ERROR,
)

if TYPE_CHECKING:
    from curio_agent_sdk.persistence.base import BasePersistence

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# OTel availability check (shared with TracingMiddleware)
# ---------------------------------------------------------------------------

_otel_available = False
try:
    from opentelemetry import trace, metrics, context as otel_context
    from opentelemetry.trace import StatusCode

    _otel_available = True
except ImportError:
    trace = None  # type: ignore[assignment]
    metrics = None  # type: ignore[assignment]
    otel_context = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment,misc]


# ===================================================================
# TracingConsumer
# ===================================================================


class TracingConsumer:
    """
    Hook-based OpenTelemetry tracing consumer.

    Creates spans for ``llm.call``, ``tool.call``, and ``agent.run`` events,
    correlated by ``run_id``. Emits the same OTel metrics as
    :class:`TracingMiddleware`.

    If ``opentelemetry-api`` is not installed, the consumer is a no-op.

    Example::

        from curio_agent_sdk.middleware.consumers import TracingConsumer

        consumer = TracingConsumer(service_name="my-agent")
        consumer.attach(hook_registry)
    """

    def __init__(
        self,
        service_name: str = "curio-agent",
        tracer: Any | None = None,
        meter: Any | None = None,
    ) -> None:
        self._enabled = _otel_available

        if not self._enabled:
            logger.warning(
                "opentelemetry-api is not installed. "
                "TracingConsumer will be a no-op. "
                "Install with: pip install opentelemetry-api"
            )
            return

        self._tracer = tracer or trace.get_tracer(service_name)
        self._meter = meter or metrics.get_meter(service_name)

        # Metrics (same names as TracingMiddleware for drop-in replacement)
        self._llm_duration = self._meter.create_histogram(
            name="agent.llm.duration",
            description="Duration of LLM calls in milliseconds",
            unit="ms",
        )
        self._llm_input_tokens = self._meter.create_counter(
            name="agent.llm.tokens.input",
            description="Total input tokens sent to LLMs",
            unit="tokens",
        )
        self._llm_output_tokens = self._meter.create_counter(
            name="agent.llm.tokens.output",
            description="Total output tokens received from LLMs",
            unit="tokens",
        )
        self._tool_duration = self._meter.create_histogram(
            name="agent.tool.duration",
            description="Duration of tool calls in milliseconds",
            unit="ms",
        )
        self._tool_errors = self._meter.create_counter(
            name="agent.tool.errors",
            description="Total tool call errors",
        )

        # In-flight state keyed by run_id (no id(request) hack)
        self._llm_spans: dict[str, Any] = {}
        self._llm_start_times: dict[str, float] = {}
        self._tool_spans: dict[str, Any] = {}
        self._tool_start_times: dict[str, float] = {}
        self._run_spans: dict[str, Any] = {}

        # Keep handler references for detach()
        self._handlers: list[tuple[str, Any]] = []

    # -- attach / detach ---------------------------------------------------

    def attach(self, registry: HookRegistry) -> None:
        """Register all hook handlers on *registry*."""
        pairs = [
            (LLM_CALL_BEFORE, self._on_llm_before),
            (LLM_CALL_AFTER, self._on_llm_after),
            (LLM_CALL_ERROR, self._on_llm_error),
            (TOOL_CALL_BEFORE, self._on_tool_before),
            (TOOL_CALL_AFTER, self._on_tool_after),
            (TOOL_CALL_ERROR, self._on_tool_error),
            (AGENT_RUN_BEFORE, self._on_run_before),
            (AGENT_RUN_AFTER, self._on_run_after),
            (AGENT_RUN_ERROR, self._on_run_error),
        ]
        for event, handler in pairs:
            registry.on(event, handler)
            self._handlers.append((event, handler))

    def detach(self, registry: HookRegistry) -> None:
        """Unregister all hook handlers from *registry*."""
        for event, handler in self._handlers:
            registry.off(event, handler)
        self._handlers.clear()

    # -- LLM handlers ------------------------------------------------------

    def _on_llm_before(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        request = ctx.data.get("request")
        span = self._tracer.start_span(
            "llm.call",
            attributes={
                "llm.provider": ctx.data.get("provider", ""),
                "llm.model": ctx.data.get("model", ""),
                "run_id": ctx.run_id,
                "agent_id": ctx.agent_id,
                "llm.message_count": len(request.messages) if request and hasattr(request, "messages") else 0,
            },
        )
        key = f"llm:{ctx.run_id}:{id(ctx)}"
        ctx.data["_tracing_key"] = key
        self._llm_spans[key] = span
        self._llm_start_times[key] = time.monotonic()

    def _on_llm_after(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        key = ctx.data.get("_tracing_key", "")
        span = self._llm_spans.pop(key, None)
        start = self._llm_start_times.pop(key, None)
        latency_ms = (time.monotonic() - start) * 1000 if start else 0.0

        response = ctx.data.get("response")
        provider = getattr(response, "provider", "") if response else ""
        model = getattr(response, "model", "") if response else ""
        usage = getattr(response, "usage", None)
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0
        finish_reason = getattr(response, "finish_reason", "") if response else ""

        if span is not None:
            span.set_attribute("llm.provider", provider)
            span.set_attribute("llm.model", model)
            span.set_attribute("llm.input_tokens", input_tokens)
            span.set_attribute("llm.output_tokens", output_tokens)
            span.set_attribute("llm.finish_reason", finish_reason)
            span.set_attribute("llm.latency_ms", latency_ms)
            if finish_reason == "error":
                span.set_status(StatusCode.ERROR, "LLM call failed")
            else:
                span.set_status(StatusCode.OK)
            span.end()

        attrs = {"provider": provider, "model": model}
        self._llm_duration.record(latency_ms, attributes=attrs)
        if input_tokens:
            self._llm_input_tokens.add(input_tokens, attributes=attrs)
        if output_tokens:
            self._llm_output_tokens.add(output_tokens, attributes=attrs)

    def _on_llm_error(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        key = ctx.data.get("_tracing_key", "")
        span = self._llm_spans.pop(key, None)
        self._llm_start_times.pop(key, None)
        error = ctx.data.get("error")
        if span is not None:
            span.set_status(StatusCode.ERROR, str(error))
            if isinstance(error, Exception):
                span.record_exception(error)
            span.end()

    # -- Tool handlers ------------------------------------------------------

    def _on_tool_before(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        tool_name = ctx.data.get("tool_name", "unknown")
        span = self._tracer.start_span(
            "tool.call",
            attributes={
                "tool.name": tool_name,
                "run_id": ctx.run_id,
                "agent_id": ctx.agent_id,
            },
        )
        key = f"tool:{ctx.run_id}:{tool_name}:{id(ctx)}"
        ctx.data["_tracing_tool_key"] = key
        self._tool_spans[key] = span
        self._tool_start_times[key] = time.monotonic()

    def _on_tool_after(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        key = ctx.data.get("_tracing_tool_key", "")
        tool_name = ctx.data.get("tool_name", "unknown")
        span = self._tool_spans.pop(key, None)
        start = self._tool_start_times.pop(key, None)
        latency_ms = (time.monotonic() - start) * 1000 if start else 0.0

        if span is not None:
            span.set_attribute("tool.latency_ms", latency_ms)
            span.set_status(StatusCode.OK)
            span.end()

        self._tool_duration.record(latency_ms, attributes={"tool_name": tool_name})

    def _on_tool_error(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        key = ctx.data.get("_tracing_tool_key", "")
        tool_name = ctx.data.get("tool_name", "unknown")
        span = self._tool_spans.pop(key, None)
        self._tool_start_times.pop(key, None)
        error = ctx.data.get("error")
        if span is not None:
            span.set_status(StatusCode.ERROR, str(error))
            if isinstance(error, Exception):
                span.record_exception(error)
            span.end()
        self._tool_errors.add(1, attributes={"tool_name": tool_name})

    # -- Agent run handlers -------------------------------------------------

    def _on_run_before(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        span = self._tracer.start_span(
            "agent.run",
            attributes={
                "run_id": ctx.run_id,
                "agent_id": ctx.agent_id,
            },
        )
        self._run_spans[ctx.run_id] = span

    def _on_run_after(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        span = self._run_spans.pop(ctx.run_id, None)
        if span is not None:
            span.set_status(StatusCode.OK)
            span.end()

    def _on_run_error(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        span = self._run_spans.pop(ctx.run_id, None)
        error = ctx.data.get("error")
        if span is not None:
            span.set_status(StatusCode.ERROR, str(error))
            if isinstance(error, Exception):
                span.record_exception(error)
            span.end()


# ===================================================================
# LoggingConsumer
# ===================================================================


class LoggingConsumer:
    """
    Hook-based structured logging consumer.

    Logs LLM calls, tool calls, and agent run lifecycle events with
    configurable log level. When OpenTelemetry is available, injects
    ``trace_id`` and ``span_id`` into log ``extra`` for correlation.

    Example::

        consumer = LoggingConsumer(level=logging.DEBUG)
        consumer.attach(hook_registry)
    """

    def __init__(
        self,
        level: int = logging.INFO,
        logger_name: str = "curio_agent_sdk.consumers.logging",
    ) -> None:
        self.level = level
        self.log = logging.getLogger(logger_name)
        self._start_times: dict[str, float] = {}
        self._handlers: list[tuple[str, Any]] = []

    def attach(self, registry: HookRegistry) -> None:
        pairs = [
            (LLM_CALL_BEFORE, self._on_llm_before),
            (LLM_CALL_AFTER, self._on_llm_after),
            (LLM_CALL_ERROR, self._on_llm_error),
            (TOOL_CALL_BEFORE, self._on_tool_before),
            (TOOL_CALL_AFTER, self._on_tool_after),
            (TOOL_CALL_ERROR, self._on_tool_error),
            (AGENT_RUN_BEFORE, self._on_run_before),
            (AGENT_RUN_AFTER, self._on_run_after),
            (AGENT_RUN_ERROR, self._on_run_error),
        ]
        for event, handler in pairs:
            registry.on(event, handler)
            self._handlers.append((event, handler))

    def detach(self, registry: HookRegistry) -> None:
        for event, handler in self._handlers:
            registry.off(event, handler)
        self._handlers.clear()

    def _extra(self) -> dict[str, Any]:
        """Build extra dict with optional trace context."""
        extra: dict[str, Any] = {}
        if _otel_available:
            span = trace.get_current_span()
            sc = span.get_span_context() if span else None
            if sc and sc.trace_id:
                extra["trace_id"] = format(sc.trace_id, "032x")
                extra["span_id"] = format(sc.span_id, "016x")
        return extra

    # -- LLM ---------------------------------------------------------------

    def _on_llm_before(self, ctx: HookContext) -> None:
        key = f"llm:{ctx.run_id}:{id(ctx)}"
        ctx.data.setdefault("_logging_key", key)
        self._start_times[key] = time.monotonic()
        request = ctx.data.get("request")
        self.log.log(
            self.level,
            "LLM call started | run_id=%s model=%s provider=%s",
            ctx.run_id,
            ctx.data.get("model", getattr(request, "model", "auto") if request else "auto"),
            ctx.data.get("provider", getattr(request, "provider", "auto") if request else "auto"),
            extra=self._extra(),
        )

    def _on_llm_after(self, ctx: HookContext) -> None:
        key = ctx.data.get("_logging_key", "")
        start = self._start_times.pop(key, None)
        elapsed = (time.monotonic() - start) * 1000 if start else 0.0
        response = ctx.data.get("response")
        usage = getattr(response, "usage", None) if response else None
        self.log.log(
            self.level,
            "LLM call completed | run_id=%s model=%s finish=%s "
            "input_tokens=%d output_tokens=%d latency=%.0fms",
            ctx.run_id,
            getattr(response, "model", "") if response else "",
            getattr(response, "finish_reason", "") if response else "",
            getattr(usage, "input_tokens", 0) if usage else 0,
            getattr(usage, "output_tokens", 0) if usage else 0,
            elapsed,
            extra=self._extra(),
        )

    def _on_llm_error(self, ctx: HookContext) -> None:
        self.log.error(
            "LLM call error | run_id=%s error=%s",
            ctx.run_id,
            ctx.data.get("error"),
            extra=self._extra(),
        )

    # -- Tool ---------------------------------------------------------------

    def _on_tool_before(self, ctx: HookContext) -> None:
        tool_name = ctx.data.get("tool_name", "unknown")
        key = f"tool:{ctx.run_id}:{tool_name}:{id(ctx)}"
        ctx.data.setdefault("_logging_tool_key", key)
        self._start_times[key] = time.monotonic()
        self.log.log(
            self.level,
            "Tool call started | run_id=%s tool=%s",
            ctx.run_id,
            tool_name,
            extra=self._extra(),
        )

    def _on_tool_after(self, ctx: HookContext) -> None:
        key = ctx.data.get("_logging_tool_key", "")
        start = self._start_times.pop(key, None)
        elapsed = (time.monotonic() - start) * 1000 if start else 0.0
        tool_name = ctx.data.get("tool_name", "unknown")
        result = ctx.data.get("result")
        result_preview = str(result)[:200] if result is not None else "None"
        self.log.log(
            self.level,
            "Tool call completed | run_id=%s tool=%s latency=%.0fms result=%s",
            ctx.run_id,
            tool_name,
            elapsed,
            result_preview,
            extra=self._extra(),
        )

    def _on_tool_error(self, ctx: HookContext) -> None:
        self.log.error(
            "Tool call error | run_id=%s tool=%s error=%s",
            ctx.run_id,
            ctx.data.get("tool_name", "unknown"),
            ctx.data.get("error"),
            extra=self._extra(),
        )

    # -- Agent run ----------------------------------------------------------

    def _on_run_before(self, ctx: HookContext) -> None:
        self.log.log(
            self.level,
            "Agent run started | run_id=%s agent_id=%s",
            ctx.run_id,
            ctx.agent_id,
            extra=self._extra(),
        )

    def _on_run_after(self, ctx: HookContext) -> None:
        self.log.log(
            self.level,
            "Agent run completed | run_id=%s agent_id=%s",
            ctx.run_id,
            ctx.agent_id,
            extra=self._extra(),
        )

    def _on_run_error(self, ctx: HookContext) -> None:
        self.log.error(
            "Agent run error | run_id=%s agent_id=%s error=%s",
            ctx.run_id,
            ctx.agent_id,
            ctx.data.get("error"),
            extra=self._extra(),
        )


# ===================================================================
# TraceContextFilter
# ===================================================================


class TraceContextFilter(logging.Filter):
    """
    A :class:`logging.Filter` that injects ``trace_id`` and ``span_id``
    into every log record when OpenTelemetry is available.

    Add to any Python logger to auto-correlate logs with traces::

        import logging
        from curio_agent_sdk.middleware.consumers import TraceContextFilter

        handler = logging.StreamHandler()
        handler.addFilter(TraceContextFilter())
        logging.getLogger().addHandler(handler)
    """

    def filter(self, record: logging.LogRecord) -> bool:
        if _otel_available:
            span = trace.get_current_span()
            sc = span.get_span_context() if span else None
            if sc and sc.trace_id:
                record.trace_id = format(sc.trace_id, "032x")  # type: ignore[attr-defined]
                record.span_id = format(sc.span_id, "016x")  # type: ignore[attr-defined]
            else:
                record.trace_id = ""  # type: ignore[attr-defined]
                record.span_id = ""  # type: ignore[attr-defined]
        else:
            record.trace_id = ""  # type: ignore[attr-defined]
            record.span_id = ""  # type: ignore[attr-defined]
        return True


# ===================================================================
# PersistenceConsumer
# ===================================================================


class PersistenceConsumer:
    """
    Hook-based persistence consumer.

    Writes :class:`AgentRunEvent` and :class:`AgentLLMUsage` records to a
    :class:`BasePersistence` backend on relevant hook events.

    Example::

        from curio_agent_sdk.persistence.sqlite import SQLitePersistence
        from curio_agent_sdk.middleware.consumers import PersistenceConsumer

        persistence = SQLitePersistence("agent.db")
        consumer = PersistenceConsumer(persistence)
        consumer.attach(hook_registry)
    """

    def __init__(self, persistence: "BasePersistence") -> None:
        self._persistence = persistence
        self._llm_start_times: dict[str, float] = {}
        self._handlers: list[tuple[str, Any]] = []

    def attach(self, registry: HookRegistry) -> None:
        pairs = [
            (LLM_CALL_BEFORE, self._on_llm_before),
            (LLM_CALL_AFTER, self._on_llm_after),
            (TOOL_CALL_AFTER, self._on_tool_after),
            (AGENT_RUN_BEFORE, self._on_run_before),
            (AGENT_RUN_AFTER, self._on_run_after),
        ]
        for event, handler in pairs:
            registry.on(event, handler)
            self._handlers.append((event, handler))

    def detach(self, registry: HookRegistry) -> None:
        for event, handler in self._handlers:
            registry.off(event, handler)
        self._handlers.clear()

    def _on_llm_before(self, ctx: HookContext) -> None:
        key = f"llm:{ctx.run_id}:{id(ctx)}"
        ctx.data.setdefault("_persist_key", key)
        self._llm_start_times[key] = time.monotonic()

    def _on_llm_after(self, ctx: HookContext) -> None:
        from curio_agent_sdk.models.agent import AgentLLMUsage

        key = ctx.data.get("_persist_key", "")
        start = self._llm_start_times.pop(key, None)
        latency_ms = int((time.monotonic() - start) * 1000) if start else 0

        response = ctx.data.get("response")
        usage = getattr(response, "usage", None) if response else None

        try:
            self._persistence.log_llm_usage(AgentLLMUsage(
                agent_id=ctx.agent_id or None,
                run_id=ctx.run_id or None,
                provider=getattr(response, "provider", "") if response else "",
                model=getattr(response, "model", "") if response else "",
                input_tokens=getattr(usage, "input_tokens", 0) if usage else 0,
                output_tokens=getattr(usage, "output_tokens", 0) if usage else 0,
                latency_ms=latency_ms,
                status="success",
            ))
        except Exception as e:
            logger.warning("PersistenceConsumer failed to log LLM usage: %s", e)

    def _on_tool_after(self, ctx: HookContext) -> None:
        import json
        from curio_agent_sdk.models.agent import AgentRunEvent

        tool_name = ctx.data.get("tool_name", "unknown")
        result = ctx.data.get("result")
        result_str = str(result)[:500] if result is not None else ""

        try:
            self._persistence.log_agent_run_event(AgentRunEvent(
                agent_id=ctx.agent_id,
                run_id=ctx.run_id,
                agent_name=ctx.data.get("agent_name", ""),
                timestamp=datetime.now(),
                event_type="tool_call",
                data=json.dumps({"tool_name": tool_name, "result_preview": result_str}),
            ))
        except Exception as e:
            logger.warning("PersistenceConsumer failed to log tool event: %s", e)

    def _on_run_before(self, ctx: HookContext) -> None:
        from curio_agent_sdk.models.agent import AgentRunEvent

        try:
            self._persistence.log_agent_run_event(AgentRunEvent(
                agent_id=ctx.agent_id,
                run_id=ctx.run_id,
                agent_name=ctx.data.get("agent_name", ""),
                timestamp=datetime.now(),
                event_type="agent_run_started",
                data=None,
            ))
        except Exception as e:
            logger.warning("PersistenceConsumer failed to log run start: %s", e)

    def _on_run_after(self, ctx: HookContext) -> None:
        import json
        from curio_agent_sdk.models.agent import AgentRunEvent

        try:
            self._persistence.log_agent_run_event(AgentRunEvent(
                agent_id=ctx.agent_id,
                run_id=ctx.run_id,
                agent_name=ctx.data.get("agent_name", ""),
                timestamp=datetime.now(),
                event_type="agent_run_completed",
                data=json.dumps({"status": ctx.data.get("status", "completed")}),
            ))
        except Exception as e:
            logger.warning("PersistenceConsumer failed to log run end: %s", e)
