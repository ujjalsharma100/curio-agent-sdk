"""
Benchmark suite for Curio Agent SDK.

Provides high-level performance benchmarks for:
- LLM call latency through the full agent/middleware stack
- Tool execution throughput via ToolExecutor
- Context compression speed via ContextManager
- Memory query performance via MemoryManager (when configured)
- Checkpoint save/restore roundtrip time via StateStore
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, List, Sequence, Tuple

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.context import ContextManager
from curio_agent_sdk.models.llm import Message, ToolCall


@dataclass
class BenchmarkResult:
    """Container for a single benchmark's aggregated metrics."""

    name: str
    metrics: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "metrics": self.metrics}


class BenchmarkSuite:
    """
    Run a suite of performance benchmarks against an Agent.

    Example:

        from curio_agent_sdk.testing import BenchmarkSuite

        suite = BenchmarkSuite(agent)
        results = await suite.run([
            ("llm_call_latency", {"iterations": 100}),
            ("tool_throughput", {"tools": ["read_file", "write_file"], "iterations": 50}),
            ("context_compression", {"message_counts": [10, 50, 100, 500]}),
            ("checkpoint_roundtrip", {"state_sizes": ["small", "medium", "large"]}),
        ])
        suite.print_report(results)
    """

    def __init__(self, agent: Agent) -> None:
        self.agent = agent

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        benchmarks: Sequence[Tuple[str, Dict[str, Any]]],
    ) -> Dict[str, BenchmarkResult]:
        """
        Run the requested benchmarks.

        Args:
            benchmarks: Sequence of (name, config_dict) pairs. Supported names:
                - "llm_call_latency"
                - "tool_throughput"
                - "context_compression"
                - "checkpoint_roundtrip"
                - "memory_query" (optional)

        Returns:
            Mapping name -> BenchmarkResult.
        """
        results: Dict[str, BenchmarkResult] = {}
        for name, cfg in benchmarks:
            if name == "llm_call_latency":
                metrics = await self._bench_llm_call_latency(**cfg)
            elif name == "tool_throughput":
                metrics = await self._bench_tool_throughput(**cfg)
            elif name == "context_compression":
                metrics = await self._bench_context_compression(**cfg)
            elif name == "checkpoint_roundtrip":
                metrics = await self._bench_checkpoint_roundtrip(**cfg)
            elif name == "memory_query":
                metrics = await self._bench_memory_query(**cfg)
            else:
                metrics = {"error": f"Unknown benchmark '{name}'"}
            results[name] = BenchmarkResult(name=name, metrics=metrics)
        return results

    def print_report(self, results: Dict[str, BenchmarkResult]) -> None:
        """Pretty-print benchmark results to stdout."""
        lines: List[str] = []
        lines.append("=== Benchmark Suite Results ===")
        for name, result in results.items():
            lines.append(f"\n[{name}]")
            metrics = result.metrics
            if "error" in metrics:
                lines.append(f"  ERROR: {metrics['error']}")
                continue
            for key, value in metrics.items():
                lines.append(f"  {key}: {value}")
        print("\n".join(lines))

    # ------------------------------------------------------------------
    # Individual benchmarks
    # ------------------------------------------------------------------

    async def _bench_llm_call_latency(
        self,
        iterations: int = 50,
        prompt: str = "Benchmark LLM call.",
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Measure end-to-end LLM call latency via Agent.arun()."""
        latencies: List[float] = []

        for _ in range(max(1, iterations)):
            start = time.monotonic()
            await self.agent.arun(prompt, **kwargs)
            elapsed_ms = (time.monotonic() - start) * 1000.0
            latencies.append(elapsed_ms)

        return {
            "iterations": len(latencies),
            "avg_ms": round(mean(latencies), 3),
            "min_ms": round(min(latencies), 3),
            "max_ms": round(max(latencies), 3),
        }

    async def _bench_tool_throughput(
        self,
        tools: Sequence[str],
        iterations: int = 50,
        args: Dict[str, Dict[str, Any]] | None = None,
    ) -> Dict[str, Any]:
        """
        Measure throughput of tool execution via ToolExecutor.

        Args:
            tools: List of tool names to benchmark.
            iterations: Number of times to execute each tool.
            args: Optional mapping tool_name -> arguments dict to use.
                  If omitted, dummy arguments will be synthesized based
                  on the tool's schema (may fail if the tool requires
                  specific values such as file paths).
        """
        if not tools:
            return {"error": "No tools provided"}

        executor = getattr(self.agent, "executor", None)
        registry = getattr(self.agent, "registry", None)
        if executor is None or registry is None:
            return {"error": "Agent has no executor/registry configured"}

        from curio_agent_sdk.models.llm import ToolCall  # local import to avoid cycles

        total_calls = 0
        latencies: List[float] = []

        for tool_name in tools:
            if not registry.has(tool_name):
                return {"error": f"Tool '{tool_name}' is not registered on the agent"}

        for i in range(max(1, iterations)):
            for name in tools:
                tool = registry.get(name)
                arg_cfg = (args or {}).get(name)
                call_args: Dict[str, Any]
                if arg_cfg is not None:
                    call_args = dict(arg_cfg)
                else:
                    # Synthesize minimal args from schema: use defaults when present,
                    # otherwise simple dummy values based on JSON type.
                    call_args = {}
                    for param in tool.schema.parameters:
                        if param.default is not None and param.default is not ...:
                            call_args[param.name] = param.default
                        elif param.type == "string":
                            call_args[param.name] = ""
                        elif param.type == "integer":
                            call_args[param.name] = 0
                        elif param.type == "number":
                            call_args[param.name] = 0.0
                        elif param.type == "boolean":
                            call_args[param.name] = False
                        elif param.type == "array":
                            call_args[param.name] = []
                        elif param.type == "object":
                            call_args[param.name] = {}
                        else:
                            call_args[param.name] = None

                tc = ToolCall(
                    id=f"bench-{name}-{i}",
                    name=name,
                    arguments=call_args,
                )
                start = time.monotonic()
                await executor.execute(tc)
                elapsed_ms = (time.monotonic() - start) * 1000.0
                latencies.append(elapsed_ms)
                total_calls += 1

        return {
            "tools": list(tools),
            "iterations": iterations,
            "total_calls": total_calls,
            "avg_ms_per_call": round(mean(latencies), 3) if latencies else 0.0,
            "min_ms": round(min(latencies), 3) if latencies else 0.0,
            "max_ms": round(max(latencies), 3) if latencies else 0.0,
        }

    async def _bench_context_compression(
        self,
        message_counts: Sequence[int],
        model: str = "gpt-4o-mini",
    ) -> Dict[str, Any]:
        """
        Measure ContextManager.fit_messages() performance for various history sizes.
        """
        if not message_counts:
            return {"error": "message_counts must be non-empty"}

        cm: ContextManager | None = getattr(self.agent, "context_manager", None)
        if cm is None:
            # Approximate: use agent.max_tokens if available, else a safe default
            max_tokens = getattr(self.agent, "max_tokens", 4096)
            cm = ContextManager(max_tokens=max_tokens)

        tool_schemas = getattr(self.agent, "registry", None)
        tool_defs = tool_schemas.get_llm_schemas() if tool_schemas is not None else None

        results: Dict[int, Dict[str, Any]] = {}

        for count in message_counts:
            messages: List[Message] = [Message.system("Benchmark context compression.")]
            for i in range(max(0, count - 1)):
                messages.append(Message.user(f"Message {i} " + "x" * 20))

            start = time.monotonic()
            _ = cm.fit_messages(messages, tools=tool_defs, model=model)  # type: ignore[arg-type]
            elapsed_ms = (time.monotonic() - start) * 1000.0
            results[int(count)] = {"latency_ms": round(elapsed_ms, 3)}

        return {"per_message_count": results}

    async def _bench_memory_query(
        self,
        iterations: int = 50,
        query: str = "benchmark",
        limit: int = 5,
    ) -> Dict[str, Any]:
        """
        Measure memory search/get_context performance via MemoryManager.
        """
        runtime = getattr(self.agent, "runtime", None)
        if runtime is None or getattr(runtime, "memory_manager", None) is None:
            return {"error": "Agent has no MemoryManager configured on runtime"}

        mm = runtime.memory_manager

        # Seed memory with some entries if it's empty
        try:
            count = await mm.count()
        except Exception:
            count = 0
        if count == 0:
            for i in range(100):
                await mm.add(f"Benchmark memory entry {i}", metadata={"type": "benchmark"})

        latencies: List[float] = []
        for _ in range(max(1, iterations)):
            start = time.monotonic()
            await mm.search(query, limit=limit)
            elapsed_ms = (time.monotonic() - start) * 1000.0
            latencies.append(elapsed_ms)

        return {
            "iterations": len(latencies),
            "avg_ms": round(mean(latencies), 3),
            "min_ms": round(min(latencies), 3),
            "max_ms": round(max(latencies), 3),
        }

    async def _bench_checkpoint_roundtrip(
        self,
        state_sizes: Sequence[str],
        iterations: int = 3,
    ) -> Dict[str, Any]:
        """
        Measure checkpoint save+restore time via Runtime/StateStore.

        Args:
            state_sizes: Labels indicating relative state sizes. Supported:
                "small", "medium", "large" (others map to "medium").
            iterations: Number of roundtrips per size.
        """
        runtime = getattr(self.agent, "runtime", None)
        if runtime is None or getattr(runtime, "state_store", None) is None:
            return {"error": "Agent has no StateStore configured on runtime"}

        size_map = {
            "small": 10,
            "medium": 100,
            "large": 500,
        }

        per_size: Dict[str, Dict[str, Any]] = {}

        for label in state_sizes:
            n_messages = size_map.get(label, size_map["medium"])
            latencies: List[float] = []

            for i in range(max(1, iterations)):
                # Create a synthetic state with specified history size
                state = runtime.create_state(f"Benchmark checkpoint {label}")
                for j in range(max(0, n_messages - len(state.messages))):
                    state.messages.append(
                        Message.user(f"Dummy message {j} " + "y" * 20)
                    )

                run_id = f"bench-{label}-{i}"
                start = time.monotonic()
                await runtime.save_state(state, run_id, agent_id=self.agent.agent_id)
                _ = await runtime.restore_state(run_id, agent_id=self.agent.agent_id)
                elapsed_ms = (time.monotonic() - start) * 1000.0
                latencies.append(elapsed_ms)

            per_size[label] = {
                "iterations": len(latencies),
                "avg_ms": round(mean(latencies), 3),
                "min_ms": round(min(latencies), 3),
                "max_ms": round(max(latencies), 3),
                "messages": n_messages,
            }

        return {"per_state_size": per_size}

