"""
Unit tests for BenchmarkSuite (Phase 16 — Testing Utilities).
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.benchmark import BenchmarkSuite, BenchmarkResult


@pytest.mark.unit
@pytest.mark.asyncio
async def test_benchmark_suite():
    """Benchmark execution runs and returns results."""
    mock = MockLLM()
    mock.add_text_response("Benchmark reply.")
    # Add many responses for iterations
    for _ in range(60):
        mock.add_text_response("Benchmark reply.")
    agent = Agent(system_prompt="Bench", tools=[], llm=mock)
    suite = BenchmarkSuite(agent)
    results = await suite.run([
        ("llm_call_latency", {"iterations": 3}),
    ])
    assert "llm_call_latency" in results
    r = results["llm_call_latency"]
    assert isinstance(r, BenchmarkResult)
    assert r.name == "llm_call_latency"
    assert "avg_ms" in r.metrics
    assert "iterations" in r.metrics
    assert r.metrics["iterations"] == 3

    # Unknown benchmark returns error in metrics
    results2 = await suite.run([("unknown_bench", {})])
    assert "error" in results2["unknown_bench"].metrics


@pytest.mark.unit
def test_benchmark_result_to_dict():
    """BenchmarkResult to_dict."""
    from curio_agent_sdk.testing.benchmark import BenchmarkResult
    r = BenchmarkResult(name="test", metrics={"avg_ms": 10.5})
    assert r.to_dict() == {"name": "test", "metrics": {"avg_ms": 10.5}}


@pytest.mark.unit
@pytest.mark.asyncio
async def test_benchmark_print_report():
    """print_report runs without error."""
    mock = MockLLM()
    mock.add_text_response("x")
    agent = Agent(system_prompt="B", tools=[], llm=mock)
    suite = BenchmarkSuite(agent)
    results = await suite.run([("llm_call_latency", {"iterations": 1})])
    suite.print_report(results)
