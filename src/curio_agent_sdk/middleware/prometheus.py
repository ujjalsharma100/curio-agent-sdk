"""
Prometheus metrics exporter for the Curio Agent SDK.

A hook-based consumer that exposes Prometheus counters, histograms, and
gauges for LLM calls, tool calls, cost, and active runs. Optionally
starts an HTTP server for Prometheus scraping.

If ``prometheus-client`` is not installed, the exporter is a no-op.

Example::

    from curio_agent_sdk.middleware.prometheus import PrometheusExporter

    exporter = PrometheusExporter(port=9090)
    exporter.attach(hook_registry)
    exporter.start_http_server()  # optional: exposes /metrics on port 9090
"""

from __future__ import annotations

import logging
from typing import Any

from curio_agent_sdk.core.hooks import (
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

logger = logging.getLogger(__name__)

_prom_available = False
try:
    from prometheus_client import (
        Counter,
        Histogram,
        Gauge,
        start_http_server as _start_http_server,
    )

    _prom_available = True
except ImportError:
    Counter = None  # type: ignore[assignment,misc]
    Histogram = None  # type: ignore[assignment,misc]
    Gauge = None  # type: ignore[assignment,misc]
    _start_http_server = None  # type: ignore[assignment]


class PrometheusExporter:
    """
    Hook consumer that exposes Prometheus metrics.

    Metrics exported:

    - ``curio_llm_duration_ms`` (Histogram) — LLM call duration
    - ``curio_llm_tokens_input_total`` (Counter) — input tokens
    - ``curio_llm_tokens_output_total`` (Counter) — output tokens
    - ``curio_llm_errors_total`` (Counter) — LLM errors
    - ``curio_tool_duration_ms`` (Histogram) — tool call duration
    - ``curio_tool_errors_total`` (Counter) — tool errors
    - ``curio_cost_usd_total`` (Counter) — estimated cost
    - ``curio_active_runs`` (Gauge) — currently active agent runs

    If ``prometheus-client`` is not installed, the exporter degrades to a
    no-op and logs a warning.

    Args:
        port: Port for the optional HTTP metrics server.
        namespace: Prometheus metric namespace prefix.
    """

    def __init__(
        self,
        port: int = 9090,
        namespace: str = "curio",
    ) -> None:
        self._port = port
        self._enabled = _prom_available
        self._server_started = False
        self._handlers: list[tuple[str, Any]] = []

        if not self._enabled:
            logger.warning(
                "prometheus-client is not installed. "
                "PrometheusExporter will be a no-op. "
                "Install with: pip install prometheus-client"
            )
            return

        # In-flight timers keyed by context id
        self._llm_start: dict[str, float] = {}
        self._tool_start: dict[str, float] = {}

        # Metrics
        self._llm_duration = Histogram(
            f"{namespace}_llm_duration_ms",
            "LLM call duration in milliseconds",
            ["provider", "model"],
        )
        self._llm_input_tokens = Counter(
            f"{namespace}_llm_tokens_input_total",
            "Total input tokens",
            ["provider", "model"],
        )
        self._llm_output_tokens = Counter(
            f"{namespace}_llm_tokens_output_total",
            "Total output tokens",
            ["provider", "model"],
        )
        self._llm_errors = Counter(
            f"{namespace}_llm_errors_total",
            "Total LLM call errors",
            ["provider", "model"],
        )
        self._tool_duration = Histogram(
            f"{namespace}_tool_duration_ms",
            "Tool call duration in milliseconds",
            ["tool_name"],
        )
        self._tool_errors = Counter(
            f"{namespace}_tool_errors_total",
            "Total tool call errors",
            ["tool_name"],
        )
        self._cost_total = Counter(
            f"{namespace}_cost_usd_total",
            "Estimated total cost in USD",
            ["model"],
        )
        self._active_runs = Gauge(
            f"{namespace}_active_runs",
            "Number of currently active agent runs",
        )

    # -- HTTP server -------------------------------------------------------

    def start_http_server(self, port: int | None = None) -> None:
        """Start the Prometheus metrics HTTP server for scraping."""
        if not self._enabled:
            return
        if self._server_started:
            return
        p = port or self._port
        _start_http_server(p)
        self._server_started = True
        logger.info("Prometheus metrics server started on port %d", p)

    # -- attach / detach ---------------------------------------------------

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

    # -- LLM handlers ------------------------------------------------------

    def _on_llm_before(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        import time
        key = f"llm:{ctx.run_id}:{id(ctx)}"
        ctx.data.setdefault("_prom_llm_key", key)
        self._llm_start[key] = time.monotonic()

    def _on_llm_after(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        import time
        key = ctx.data.get("_prom_llm_key", "")
        start = self._llm_start.pop(key, None)
        latency_ms = (time.monotonic() - start) * 1000 if start else 0.0

        response = ctx.data.get("response")
        provider = getattr(response, "provider", "unknown") if response else "unknown"
        model = getattr(response, "model", "unknown") if response else "unknown"
        usage = getattr(response, "usage", None) if response else None
        input_tokens = getattr(usage, "input_tokens", 0) if usage else 0
        output_tokens = getattr(usage, "output_tokens", 0) if usage else 0

        self._llm_duration.labels(provider=provider, model=model).observe(latency_ms)
        self._llm_input_tokens.labels(provider=provider, model=model).inc(input_tokens)
        self._llm_output_tokens.labels(provider=provider, model=model).inc(output_tokens)

    def _on_llm_error(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        key = ctx.data.get("_prom_llm_key", "")
        self._llm_start.pop(key, None)
        provider = ctx.data.get("provider", "unknown")
        model = ctx.data.get("model", "unknown")
        self._llm_errors.labels(provider=provider, model=model).inc()

    # -- Tool handlers ------------------------------------------------------

    def _on_tool_before(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        import time
        tool_name = ctx.data.get("tool_name", "unknown")
        key = f"tool:{ctx.run_id}:{tool_name}:{id(ctx)}"
        ctx.data.setdefault("_prom_tool_key", key)
        self._tool_start[key] = time.monotonic()

    def _on_tool_after(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        import time
        key = ctx.data.get("_prom_tool_key", "")
        tool_name = ctx.data.get("tool_name", "unknown")
        start = self._tool_start.pop(key, None)
        latency_ms = (time.monotonic() - start) * 1000 if start else 0.0
        self._tool_duration.labels(tool_name=tool_name).observe(latency_ms)

    def _on_tool_error(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        key = ctx.data.get("_prom_tool_key", "")
        self._tool_start.pop(key, None)
        tool_name = ctx.data.get("tool_name", "unknown")
        self._tool_errors.labels(tool_name=tool_name).inc()

    # -- Agent run handlers -------------------------------------------------

    def _on_run_before(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        self._active_runs.inc()

    def _on_run_after(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        self._active_runs.dec()

    def _on_run_error(self, ctx: HookContext) -> None:
        if not self._enabled:
            return
        self._active_runs.dec()
