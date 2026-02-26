"""
OpenTelemetry tracing middleware for the Curio Agent SDK.

Provides distributed tracing and metrics for LLM and tool calls.
If opentelemetry-api is not installed, the middleware is a no-op.
"""

from __future__ import annotations

import logging
import time
from typing import Any

from curio_agent_sdk.middleware.base import Middleware
from curio_agent_sdk.models.llm import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)

_otel_available = False
try:
    from opentelemetry import trace, metrics
    from opentelemetry.trace import StatusCode

    _otel_available = True
except ImportError:
    trace = None  # type: ignore[assignment]
    metrics = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment,misc]

_NO_OTEL_WARNING_LOGGED = False


class TracingMiddleware(Middleware):
    """
    Middleware that emits OpenTelemetry spans and metrics for every LLM
    and tool call.

    If ``opentelemetry-api`` is not installed, the middleware silently
    degrades to a no-op (a warning is logged once).

    Args:
        service_name: Logical service name for the tracer/meter.
        tracer: An explicit Tracer instance (defaults to global).
        meter: An explicit Meter instance (defaults to global).

    Example:
        agent = Agent(
            middleware=[TracingMiddleware(service_name="my-agent")],
            ...
        )
    """

    def __init__(
        self,
        service_name: str = "curio-agent",
        tracer: Any | None = None,
        meter: Any | None = None,
    ) -> None:
        global _NO_OTEL_WARNING_LOGGED

        self._enabled = _otel_available

        if not self._enabled:
            if not _NO_OTEL_WARNING_LOGGED:
                logger.warning(
                    "opentelemetry-api is not installed. "
                    "TracingMiddleware will be a no-op. "
                    "Install with: pip install opentelemetry-api"
                )
                _NO_OTEL_WARNING_LOGGED = True
            return

        self._tracer = tracer or trace.get_tracer(service_name)
        self._meter = meter or metrics.get_meter(service_name)

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

        self._llm_start_times: dict[int, float] = {}
        self._llm_spans: dict[int, Any] = {}
        self._tool_start_times: dict[str, float] = {}
        self._tool_spans: dict[str, Any] = {}

    async def before_llm_call(self, request: LLMRequest) -> LLMRequest:
        if not self._enabled:
            return request

        span = self._tracer.start_span(
            "llm.call",
            attributes={
                "llm.provider": request.provider or "",
                "llm.model": request.model or "",
                "llm.max_tokens": request.max_tokens,
                "llm.temperature": request.temperature,
                "llm.message_count": len(request.messages),
            },
        )
        req_id = id(request)
        self._llm_spans[req_id] = span
        self._llm_start_times[req_id] = time.monotonic()
        return request

    async def after_llm_call(self, request: LLMRequest, response: LLMResponse) -> LLMResponse:
        if not self._enabled:
            return response

        req_id = id(request)
        span = self._llm_spans.pop(req_id, None)
        start = self._llm_start_times.pop(req_id, None)
        latency_ms = (time.monotonic() - start) * 1000 if start else 0.0

        if span is not None:
            span.set_attribute("llm.provider", response.provider)
            span.set_attribute("llm.model", response.model)
            span.set_attribute("llm.input_tokens", response.usage.input_tokens)
            span.set_attribute("llm.output_tokens", response.usage.output_tokens)
            span.set_attribute("llm.finish_reason", response.finish_reason)
            span.set_attribute("llm.latency_ms", latency_ms)

            if response.finish_reason == "error":
                span.set_status(StatusCode.ERROR, response.error or "LLM call failed")
            else:
                span.set_status(StatusCode.OK)
            span.end()

        attrs = {"provider": response.provider, "model": response.model}
        self._llm_duration.record(latency_ms, attributes=attrs)
        self._llm_input_tokens.add(response.usage.input_tokens, attributes=attrs)
        self._llm_output_tokens.add(response.usage.output_tokens, attributes=attrs)

        return response

    async def before_tool_call(self, tool_name: str, args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if not self._enabled:
            return tool_name, args

        span = self._tracer.start_span(
            "tool.call",
            attributes={"tool.name": tool_name},
        )
        call_key = f"{tool_name}:{id(args)}"
        self._tool_spans[call_key] = span
        self._tool_start_times[call_key] = time.monotonic()
        args["__tracing_call_key"] = call_key
        return tool_name, args

    async def after_tool_call(self, tool_name: str, args: dict[str, Any], result: Any) -> Any:
        if not self._enabled:
            return result

        call_key = args.pop("__tracing_call_key", f"{tool_name}:{id(args)}")
        span = self._tool_spans.pop(call_key, None)
        start = self._tool_start_times.pop(call_key, None)
        latency_ms = (time.monotonic() - start) * 1000 if start else 0.0

        if span is not None:
            span.set_attribute("tool.latency_ms", latency_ms)
            span.set_status(StatusCode.OK)
            span.end()

        self._tool_duration.record(latency_ms, attributes={"tool_name": tool_name})
        return result

    async def on_error(self, error: Exception, context: dict[str, Any]) -> Exception | None:
        if not self._enabled:
            return error

        phase = context.get("phase", "unknown")

        if phase == "llm_call":
            request = context.get("request")
            if request is not None:
                req_id = id(request)
                span = self._llm_spans.pop(req_id, None)
                self._llm_start_times.pop(req_id, None)
                if span is not None:
                    span.set_status(StatusCode.ERROR, str(error))
                    span.record_exception(error)
                    span.end()

        if phase == "tool_call":
            tool_name = context.get("tool_name", "unknown")
            self._tool_errors.add(1, attributes={"tool_name": tool_name})
            for key in list(self._tool_spans):
                if key.startswith(f"{tool_name}:"):
                    span = self._tool_spans.pop(key)
                    self._tool_start_times.pop(key, None)
                    span.set_status(StatusCode.ERROR, str(error))
                    span.record_exception(error)
                    span.end()
                    break

        return error
