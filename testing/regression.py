"""
Regression detection for agent evaluation.

Compares a candidate :class:`EvalSuiteResult` against a saved baseline
and flags metric regressions beyond a configurable threshold.

Example::

    from curio_agent_sdk.testing.regression import RegressionDetector

    detector = RegressionDetector(threshold=0.05)

    # After an initial run, save the baseline
    detector.save_baseline(results, "baseline.json")

    # On subsequent runs, compare against baseline
    report = detector.compare(new_results, baseline_path="baseline.json")
    print(report)
    assert report.passed
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from curio_agent_sdk.testing.eval import EvalSuiteResult

logger = logging.getLogger(__name__)


@dataclass
class RegressionReport:
    """
    Comparison report between a baseline and a candidate evaluation run.

    Attributes:
        baseline_pass_rate: Pass rate of the baseline run.
        candidate_pass_rate: Pass rate of the candidate run.
        metric_deltas: Per-metric average-score delta (candidate - baseline).
            Negative values indicate regression.
        threshold: Maximum allowed regression fraction.
        passed: True if no metric regressed beyond *threshold*.
        details: Human-readable summary lines.
    """

    baseline_pass_rate: float = 0.0
    candidate_pass_rate: float = 0.0
    metric_deltas: dict[str, float] = field(default_factory=dict)
    threshold: float = 0.05
    passed: bool = True
    details: list[str] = field(default_factory=list)

    def __str__(self) -> str:
        status = "PASSED" if self.passed else "FAILED"
        lines = [
            f"Regression Report: {status}",
            f"  Baseline pass rate: {self.baseline_pass_rate:.2%}",
            f"  Candidate pass rate: {self.candidate_pass_rate:.2%}",
            f"  Threshold: {self.threshold:.2%}",
        ]
        for metric, delta in self.metric_deltas.items():
            direction = "improved" if delta >= 0 else "REGRESSED"
            lines.append(f"  {metric}: {delta:+.4f} ({direction})")
        lines.extend(self.details)
        return "\n".join(lines)

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_pass_rate": self.baseline_pass_rate,
            "candidate_pass_rate": self.candidate_pass_rate,
            "metric_deltas": self.metric_deltas,
            "threshold": self.threshold,
            "passed": self.passed,
            "details": self.details,
        }


class RegressionDetector:
    """
    Compare evaluation results against a saved baseline.

    Args:
        threshold: Maximum allowed regression fraction (default 5%).
            If any metric drops by more than this amount, the check fails.

    Example::

        detector = RegressionDetector(threshold=0.05)
        report = detector.compare(candidate, baseline_path="baseline.json")
        if not report.passed:
            print("Regression detected!", report)
    """

    def __init__(self, threshold: float = 0.05) -> None:
        self.threshold = threshold

    def save_baseline(self, results: EvalSuiteResult, path: str | Path) -> None:
        """Save evaluation results as a baseline JSON file."""
        Path(path).write_text(results.to_json())
        logger.info("Baseline saved to %s", path)

    def load_baseline(self, path: str | Path) -> EvalSuiteResult:
        """Load a baseline from a JSON file."""
        return EvalSuiteResult.from_json(Path(path).read_text())

    def compare(
        self,
        candidate: EvalSuiteResult,
        baseline: EvalSuiteResult | None = None,
        baseline_path: str | Path | None = None,
    ) -> RegressionReport:
        """
        Compare *candidate* results against a *baseline*.

        Provide either a ``baseline`` :class:`EvalSuiteResult` or a
        ``baseline_path`` to a saved JSON file.

        Returns:
            A :class:`RegressionReport` with pass/fail status and deltas.
        """
        if baseline is None and baseline_path is not None:
            baseline = self.load_baseline(baseline_path)
        if baseline is None:
            raise ValueError("Either baseline or baseline_path must be provided")

        # Gather all metric names across both runs
        metric_names: set[str] = set()
        for r in baseline.results:
            metric_names.update(r.metrics.keys())
        for r in candidate.results:
            metric_names.update(r.metrics.keys())

        metric_deltas: dict[str, float] = {}
        details: list[str] = []
        passed = True

        for name in sorted(metric_names):
            baseline_avg = baseline.avg_metric(name)
            candidate_avg = candidate.avg_metric(name)
            delta = candidate_avg - baseline_avg
            metric_deltas[name] = round(delta, 6)

            if delta < -self.threshold:
                passed = False
                details.append(
                    f"  REGRESSION in '{name}': baseline={baseline_avg:.4f} "
                    f"candidate={candidate_avg:.4f} delta={delta:+.4f} "
                    f"(threshold={self.threshold:.4f})"
                )

        # Check pass rate regression
        pass_rate_delta = candidate.pass_rate() - baseline.pass_rate()
        if pass_rate_delta < -self.threshold:
            passed = False
            details.append(
                f"  REGRESSION in pass_rate: baseline={baseline.pass_rate():.2%} "
                f"candidate={candidate.pass_rate():.2%} "
                f"delta={pass_rate_delta:+.4f}"
            )

        return RegressionReport(
            baseline_pass_rate=baseline.pass_rate(),
            candidate_pass_rate=candidate.pass_rate(),
            metric_deltas=metric_deltas,
            threshold=self.threshold,
            passed=passed,
            details=details,
        )
