"""
Unit tests for AgentEvalSuite and eval metrics (Phase 16 — Testing Utilities).
"""

import pytest

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.testing.mock_llm import MockLLM
from curio_agent_sdk.testing.eval import (
    AgentEvalSuite,
    EvalCase,
    EvalDataset,
    EvalResult,
    EvalSuiteResult,
    exact_match,
    contains_match,
    tool_call_match,
    token_efficiency,
)


# ---------------------------------------------------------------------------
# Eval metrics
# ---------------------------------------------------------------------------


@pytest.mark.unit
def test_eval_metrics():
    """exact_match, contains_match, tool_call_match, token_efficiency."""
    case = EvalCase(input="Hi", expected_output="Hello", expected_tool_calls=["search"])
    assert exact_match(case, "Hello", []) == 1.0
    assert exact_match(case, "Other", []) == 0.0
    assert contains_match(case, "Hello world", []) == 1.0
    assert contains_match(case, "Bye", []) == 0.0
    assert contains_match(EvalCase(input="x", expected_output=""), "anything", []) == 1.0
    assert tool_call_match(case, "", ["search"]) == 1.0
    assert tool_call_match(case, "", ["search", "other"]) == 1.0
    assert tool_call_match(case, "", []) == 0.0
    assert tool_call_match(EvalCase(input="x", expected_tool_calls=[]), "", []) == 1.0
    assert token_efficiency(EvalCase(input="ab", expected_output=""), "abcd", []) == 2.0
    assert token_efficiency(EvalCase(input="", expected_output=""), "x", []) == 0.0


# ---------------------------------------------------------------------------
# EvalSuite
# ---------------------------------------------------------------------------


@pytest.mark.unit
@pytest.mark.asyncio
async def test_eval_suite():
    """Eval suite run."""
    mock = MockLLM()
    mock.add_text_response("Answer one.")
    mock.add_text_response("Answer two.")
    agent = Agent(system_prompt="Test", tools=[], llm=mock)
    dataset = EvalDataset([
        EvalCase(input="Q1", expected_output="Answer one."),
        EvalCase(input="Q2", expected_output="Answer two."),
    ])
    suite = AgentEvalSuite(metrics=[exact_match], pass_threshold=0.5)
    results = await suite.run(agent, dataset)
    assert isinstance(results, EvalSuiteResult)
    assert len(results.results) == 2
    assert results.pass_rate() >= 0.0 and results.pass_rate() <= 1.0
    assert results.total_tokens() >= 0


@pytest.mark.unit
def test_eval_suite_result():
    """EvalSuiteResult pass_rate, avg_metric, to_json/from_json."""
    r1 = EvalResult(case=EvalCase(input="x", expected_output="y"), agent_output="y", passed=True, metrics={"m1": 1.0})
    r2 = EvalResult(case=EvalCase(input="a", expected_output="b"), agent_output="x", passed=False, metrics={"m1": 0.0})
    suite_result = EvalSuiteResult(results=[r1, r2])
    assert suite_result.pass_rate() == 0.5
    assert suite_result.avg_metric("m1") == 0.5
    assert suite_result.avg_metric("missing") == 0.0
    js = suite_result.to_json()
    loaded = EvalSuiteResult.from_json(js)
    assert loaded.pass_rate() == suite_result.pass_rate()
    assert len(loaded.results) == 2


@pytest.mark.unit
def test_eval_case_to_dict_from_dict():
    """EvalCase to_dict/from_dict roundtrip."""
    case = EvalCase(input="q", expected_output="a", expected_tool_calls=["t"], tags=["x"])
    d = case.to_dict()
    assert d["input"] == "q"
    assert d["expected_tool_calls"] == ["t"]
    assert EvalCase.from_dict(d).input == case.input


@pytest.mark.unit
def test_eval_dataset_filter_by_tag():
    """EvalDataset filter_by_tag returns subset."""
    dataset = EvalDataset([
        EvalCase(input="1", expected_output="", tags=["math"]),
        EvalCase(input="2", expected_output="", tags=["math", "hard"]),
        EvalCase(input="3", expected_output="", tags=[]),
    ])
    filtered = dataset.filter_by_tag("math")
    assert len(filtered) == 2
    assert len(dataset.filter_by_tag("hard")) == 1
