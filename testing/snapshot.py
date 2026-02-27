"""
Snapshot testing utilities for Curio Agent SDK.

Supports capturing the full output of an agent run into golden files
and diffing future runs against those snapshots to detect regressions.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable

from curio_agent_sdk.models.agent import AgentRunResult


class SnapshotMismatchError(AssertionError):
    """Raised when a snapshot comparison fails."""

    def __init__(
        self,
        name: str,
        message: str,
        expected: Dict[str, Any] | None = None,
        actual: Dict[str, Any] | None = None,
    ) -> None:
        self.name = name
        self.expected = expected
        self.actual = actual
        super().__init__(message)


@dataclass
class SnapshotTester:
    """
    Helper for snapshot-style testing of agent outputs.

    Typical usage::

        from curio_agent_sdk.testing import SnapshotTester

        tester = SnapshotTester("tests/snapshots")
        result = await harness.run("Analyze this code")
        await tester.assert_snapshot("analyze_code_basic", result)

    On first run, the snapshot is written. On subsequent runs, the new
    output is compared against the stored snapshot and a
    SnapshotMismatchError is raised on differences.
    """

    snapshot_dir: Path
    update: bool = False
    ignore_keys: Iterable[str] = ("run_id",)

    def __init__(
        self,
        snapshot_dir: str | Path = "tests/snapshots",
        update: bool = False,
        ignore_keys: Iterable[str] | None = None,
    ) -> None:
        self.snapshot_dir = Path(snapshot_dir)
        self.snapshot_dir.mkdir(parents=True, exist_ok=True)
        self.update = update
        self.ignore_keys = tuple(ignore_keys) if ignore_keys is not None else ("run_id",)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def assert_snapshot(
        self,
        name: str,
        result: AgentRunResult | Dict[str, Any] | str,
        *,
        update: bool | None = None,
    ) -> None:
        """
        Assert that *result* matches the stored snapshot with this *name*.

        If the snapshot file does not exist (or update=True), a new
        snapshot is written and the assertion passes.
        """
        path = self._snapshot_path(name)
        payload = self._to_snapshot_payload(result)

        # Filter out ignored keys before persisting/comparing
        filtered_actual = self._filter_keys(payload)

        effective_update = self.update if update is None else update

        if not path.exists() or effective_update:
            path.write_text(json.dumps(filtered_actual, indent=2, sort_keys=True, default=str))
            return

        expected = json.loads(path.read_text())
        filtered_expected = self._filter_keys(expected)

        if filtered_expected != filtered_actual:
            # Build a concise diff message focusing on output differences.
            expected_output = filtered_expected.get("output")
            actual_output = filtered_actual.get("output")
            msg_lines = [f"Snapshot mismatch for '{name}'."]
            if expected_output != actual_output:
                msg_lines.append("  Output differs from snapshot.")
            msg_lines.append(f"  Snapshot file: {path}")
            raise SnapshotMismatchError(
                name=name,
                message="\n".join(msg_lines),
                expected=filtered_expected,
                actual=filtered_actual,
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _snapshot_path(self, name: str) -> Path:
        safe = name.replace("/", "_").replace("\\", "_")
        return self.snapshot_dir / f"{safe}.json"

    def _to_snapshot_payload(self, result: AgentRunResult | Dict[str, Any] | str) -> Dict[str, Any]:
        if isinstance(result, AgentRunResult):
            # Use the structured result, including top-level metrics and output.
            return result.to_dict()
        if isinstance(result, str):
            return {"output": result}
        if isinstance(result, dict):
            return dict(result)
        # Fallback: best-effort JSON-serializable view
        return json.loads(json.dumps(result, default=str))

    def _filter_keys(self, data: Dict[str, Any]) -> Dict[str, Any]:
        if not self.ignore_keys:
            return dict(data)
        return {k: v for k, v in data.items() if k not in self.ignore_keys}

