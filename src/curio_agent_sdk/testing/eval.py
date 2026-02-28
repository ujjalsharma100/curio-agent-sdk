"""
Agent evaluation framework for the Curio Agent SDK.

Provides structured evaluation suites for testing agent quality,
including support for datasets, built-in metrics, and A/B testing.

Example::

    from curio_agent_sdk.testing.eval import (
        AgentEvalSuite, EvalDataset, EvalCase, exact_match,
    )

    dataset = EvalDataset.from_json("cases.json")
    suite = AgentEvalSuite(metrics=[exact_match])
    results = await suite.run(agent, dataset)
    print(f"Pass rate: {results.pass_rate():.1%}")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Sequence, TYPE_CHECKING

if TYPE_CHECKING:
    from curio_agent_sdk.core.agent import Agent

logger = logging.getLogger(__name__)


# ===================================================================
# Data classes
# ===================================================================


@dataclass
class EvalCase:
    """A single evaluation test case."""

    input: str
    expected_output: str = ""
    expected_tool_calls: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "input": self.input,
            "expected_output": self.expected_output,
            "expected_tool_calls": self.expected_tool_calls,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvalCase:
        return cls(
            input=data["input"],
            expected_output=data.get("expected_output", ""),
            expected_tool_calls=data.get("expected_tool_calls", []),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )


@dataclass
class EvalResult:
    """Result of evaluating a single case."""

    case: EvalCase
    agent_output: str = ""
    agent_tool_calls: list[str] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    passed: bool = False
    error: str | None = None
    total_tokens: int = 0
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case": self.case.to_dict(),
            "agent_output": self.agent_output,
            "agent_tool_calls": self.agent_tool_calls,
            "metrics": self.metrics,
            "passed": self.passed,
            "error": self.error,
            "total_tokens": self.total_tokens,
            "latency_ms": self.latency_ms,
        }


@dataclass
class EvalSuiteResult:
    """Aggregated results from running an evaluation suite."""

    results: list[EvalResult] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def pass_rate(self) -> float:
        """Fraction of cases that passed (0.0 – 1.0)."""
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.passed) / len(self.results)

    def avg_metric(self, name: str) -> float:
        """Average value of a named metric across all results."""
        values = [r.metrics[name] for r in self.results if name in r.metrics]
        return sum(values) / len(values) if values else 0.0

    def total_tokens(self) -> int:
        return sum(r.total_tokens for r in self.results)

    def to_json(self) -> str:
        return json.dumps({
            "results": [r.to_dict() for r in self.results],
            "metadata": self.metadata,
            "pass_rate": self.pass_rate(),
        }, indent=2, default=str)

    @classmethod
    def from_json(cls, data: str) -> EvalSuiteResult:
        parsed = json.loads(data)
        results = []
        for r in parsed.get("results", []):
            case = EvalCase.from_dict(r["case"])
            results.append(EvalResult(
                case=case,
                agent_output=r.get("agent_output", ""),
                agent_tool_calls=r.get("agent_tool_calls", []),
                metrics=r.get("metrics", {}),
                passed=r.get("passed", False),
                error=r.get("error"),
                total_tokens=r.get("total_tokens", 0),
                latency_ms=r.get("latency_ms", 0.0),
            ))
        return cls(results=results, metadata=parsed.get("metadata", {}))


# ===================================================================
# EvalDataset
# ===================================================================


class EvalDataset:
    """
    A collection of :class:`EvalCase` instances, loadable from JSON/JSONL.

    Example::

        dataset = EvalDataset.from_json("cases.json")
        subset = dataset.filter_by_tag("math")
    """

    def __init__(self, cases: list[EvalCase] | None = None) -> None:
        self.cases: list[EvalCase] = cases or []

    def __len__(self) -> int:
        return len(self.cases)

    def __iter__(self):
        return iter(self.cases)

    def filter_by_tag(self, tag: str) -> EvalDataset:
        """Return a new dataset containing only cases with the given tag."""
        return EvalDataset([c for c in self.cases if tag in c.tags])

    @classmethod
    def from_json(cls, path: str | Path) -> EvalDataset:
        """Load cases from a JSON file (array of case objects)."""
        data = json.loads(Path(path).read_text())
        if isinstance(data, list):
            return cls([EvalCase.from_dict(d) for d in data])
        cases = data.get("cases", data.get("data", []))
        return cls([EvalCase.from_dict(d) for d in cases])

    @classmethod
    def from_jsonl(cls, path: str | Path) -> EvalDataset:
        """Load cases from a JSONL file (one JSON object per line)."""
        cases = []
        for line in Path(path).read_text().strip().splitlines():
            line = line.strip()
            if line:
                cases.append(EvalCase.from_dict(json.loads(line)))
        return cls(cases)


# ===================================================================
# Built-in metrics
# ===================================================================

MetricFn = Callable[[EvalCase, str, list[str]], float]


def exact_match(case: EvalCase, output: str, tool_calls: list[str]) -> float:
    """1.0 if agent output exactly matches expected, else 0.0."""
    return 1.0 if output.strip() == case.expected_output.strip() else 0.0


def contains_match(case: EvalCase, output: str, tool_calls: list[str]) -> float:
    """1.0 if expected output is contained in agent output, else 0.0."""
    if not case.expected_output:
        return 1.0
    return 1.0 if case.expected_output.strip() in output else 0.0


def tool_call_match(case: EvalCase, output: str, tool_calls: list[str]) -> float:
    """Fraction of expected tool calls that were actually made."""
    if not case.expected_tool_calls:
        return 1.0
    if not tool_calls:
        return 0.0
    matches = sum(1 for t in case.expected_tool_calls if t in tool_calls)
    return matches / len(case.expected_tool_calls)


def token_efficiency(case: EvalCase, output: str, tool_calls: list[str]) -> float:
    """
    Ratio of output length to input length (higher = more efficient).

    This is a simple proxy; the actual token count is tracked in EvalResult.
    """
    if not case.input:
        return 0.0
    return len(output) / len(case.input) if case.input else 0.0


# ===================================================================
# AgentEvalSuite
# ===================================================================


class AgentEvalSuite:
    """
    Run a dataset of eval cases against an agent and compute metrics.

    Args:
        metrics: List of metric functions to compute for each case.
            Each function has signature ``(case, output, tool_calls) -> float``.
        pass_threshold: Minimum average metric score to consider a case passed.

    Example::

        suite = AgentEvalSuite(metrics=[exact_match, contains_match])
        results = await suite.run(agent, dataset)
        print(f"Pass rate: {results.pass_rate():.1%}")
    """

    def __init__(
        self,
        metrics: list[MetricFn] | None = None,
        pass_threshold: float = 0.5,
    ) -> None:
        self.metrics = metrics or [contains_match]
        self.pass_threshold = pass_threshold

    async def run(
        self,
        agent: "Agent",
        dataset: EvalDataset,
        **agent_kwargs: Any,
    ) -> EvalSuiteResult:
        """Run all cases in *dataset* against *agent* and return aggregated results."""
        import time

        results: list[EvalResult] = []
        for case in dataset:
            start = time.monotonic()
            try:
                run_result = await agent.arun(case.input, **agent_kwargs)
                elapsed_ms = (time.monotonic() - start) * 1000

                output = run_result.output or ""
                # Extract tool call names from run result messages
                tool_calls: list[str] = []
                for msg in getattr(run_result, "messages", []):
                    for tc in getattr(msg, "tool_calls", []):
                        tool_calls.append(getattr(tc, "name", str(tc)))

                # Compute metrics
                metric_scores: dict[str, float] = {}
                for metric_fn in self.metrics:
                    metric_scores[metric_fn.__name__] = metric_fn(case, output, tool_calls)

                avg_score = (
                    sum(metric_scores.values()) / len(metric_scores)
                    if metric_scores else 0.0
                )
                passed = avg_score >= self.pass_threshold

                results.append(EvalResult(
                    case=case,
                    agent_output=output,
                    agent_tool_calls=tool_calls,
                    metrics=metric_scores,
                    passed=passed,
                    total_tokens=run_result.total_input_tokens + run_result.total_output_tokens,
                    latency_ms=elapsed_ms,
                ))
            except Exception as e:
                elapsed_ms = (time.monotonic() - start) * 1000
                logger.warning("Eval case failed: %s", e)
                results.append(EvalResult(
                    case=case,
                    error=str(e),
                    passed=False,
                    latency_ms=elapsed_ms,
                ))

        return EvalSuiteResult(results=results)

    async def run_ab(
        self,
        agent_a: "Agent",
        agent_b: "Agent",
        dataset: EvalDataset,
        **agent_kwargs: Any,
    ) -> dict[str, EvalSuiteResult]:
        """
        Run the same dataset against two agents for A/B comparison.

        Returns:
            Dict with keys ``"a"`` and ``"b"`` mapping to their respective
            :class:`EvalSuiteResult`.
        """
        result_a = await self.run(agent_a, dataset, **agent_kwargs)
        result_b = await self.run(agent_b, dataset, **agent_kwargs)
        return {"a": result_a, "b": result_b}
