"""
Unit tests for RegressionDetector (Phase 16 — Testing Utilities).
"""

import tempfile
from pathlib import Path

import pytest

from curio_agent_sdk.testing.eval import EvalCase, EvalResult, EvalSuiteResult
from curio_agent_sdk.testing.regression import RegressionDetector, RegressionReport


@pytest.mark.unit
def test_regression_detector():
    """Regression detection: compare candidate to baseline."""
    base = EvalSuiteResult(results=[
        EvalResult(case=EvalCase(input="x", expected_output="y"), agent_output="y", passed=True, metrics={"score": 1.0}),
        EvalResult(case=EvalCase(input="a", expected_output="b"), agent_output="b", passed=True, metrics={"score": 1.0}),
    ])
    candidate = EvalSuiteResult(results=[
        EvalResult(case=EvalCase(input="x", expected_output="y"), agent_output="y", passed=True, metrics={"score": 1.0}),
        EvalResult(case=EvalCase(input="a", expected_output="b"), agent_output="x", passed=False, metrics={"score": 0.0}),
    ])
    detector = RegressionDetector(threshold=0.05)
    report = detector.compare(candidate, baseline=base)
    assert isinstance(report, RegressionReport)
    assert report.baseline_pass_rate == 1.0
    assert report.candidate_pass_rate == 0.5
    assert "score" in report.metric_deltas
    assert report.passed is False
    assert "REGRESSION" in "\n".join(report.details) or report.candidate_pass_rate < report.baseline_pass_rate


@pytest.mark.unit
def test_regression_detector_save_load():
    """Save baseline and load for compare."""
    results = EvalSuiteResult(results=[
        EvalResult(case=EvalCase(input="q", expected_output="a"), agent_output="a", passed=True, metrics={"m": 1.0}),
    ])
    detector = RegressionDetector(threshold=0.1)
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
        path = f.name
    try:
        detector.save_baseline(results, path)
        loaded = detector.load_baseline(path)
        assert loaded.pass_rate() == results.pass_rate()
        report = detector.compare(results, baseline_path=path)
        assert report.passed is True
    finally:
        Path(path).unlink(missing_ok=True)


@pytest.mark.unit
def test_regression_report_str_and_to_dict():
    """RegressionReport __str__ and to_dict."""
    report = RegressionReport(
        baseline_pass_rate=0.8,
        candidate_pass_rate=0.9,
        metric_deltas={"m1": 0.1},
        passed=True,
    )
    s = str(report)
    assert "PASSED" in s
    assert "0.80%" in s or "80" in s
    d = report.to_dict()
    assert d["baseline_pass_rate"] == 0.8
    assert d["metric_deltas"]["m1"] == 0.1


@pytest.mark.unit
def test_regression_passed_when_no_regression():
    """compare returns passed=True when candidate is same or better."""
    results = EvalSuiteResult(results=[
        EvalResult(case=EvalCase(input="x", expected_output="y"), agent_output="y", passed=True, metrics={"s": 1.0}),
    ])
    detector = RegressionDetector(threshold=0.05)
    report = detector.compare(results, baseline=results)
    assert report.passed is True
    assert report.metric_deltas.get("s", 0) == 0.0
