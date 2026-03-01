"""
Unit tests for SnapshotTester (Phase 16 — Testing Utilities).
"""

import tempfile
from pathlib import Path

import pytest

from curio_agent_sdk.models.agent import AgentRunResult
from curio_agent_sdk.testing.snapshot import SnapshotTester, SnapshotMismatchError


@pytest.mark.unit
@pytest.mark.asyncio
async def test_snapshot_tester():
    """Snapshot comparison: first run writes, second run compares."""
    with tempfile.TemporaryDirectory() as tmp:
        dir_path = Path(tmp)
        tester = SnapshotTester(snapshot_dir=dir_path, update=False)
        result = AgentRunResult(
            status="completed",
            output="Hello world",
            run_id="run-1",
        )
        await tester.assert_snapshot("hello", result)
        snap_file = dir_path / "hello.json"
        assert snap_file.exists()
        data = __import__("json").loads(snap_file.read_text())
        assert data.get("output") == "Hello world"
        assert "run_id" not in data  # ignore_keys

        # Same result (ignoring run_id) should pass
        result2 = AgentRunResult(status="completed", output="Hello world", run_id="run-2")
        await tester.assert_snapshot("hello", result2)

        # Mismatch should raise
        result3 = AgentRunResult(status="completed", output="Different", run_id="run-3")
        with pytest.raises(SnapshotMismatchError) as exc_info:
            await tester.assert_snapshot("hello", result3)
        assert "hello" in str(exc_info.value)
        assert exc_info.value.actual is not None
        assert exc_info.value.expected is not None


@pytest.mark.unit
@pytest.mark.asyncio
async def test_snapshot_tester_update_mode():
    """Snapshot with update=True overwrites stored snapshot."""
    with tempfile.TemporaryDirectory() as tmp:
        dir_path = Path(tmp)
        tester = SnapshotTester(snapshot_dir=dir_path, update=True)
        result = AgentRunResult(status="completed", output="First", run_id="r1")
        await tester.assert_snapshot("updateme", result)
        result2 = AgentRunResult(status="completed", output="Second", run_id="r2")
        await tester.assert_snapshot("updateme", result2, update=True)
        data = __import__("json").loads((dir_path / "updateme.json").read_text())
        assert data.get("output") == "Second"


@pytest.mark.unit
@pytest.mark.asyncio
async def test_snapshot_tester_dict_and_str_payload():
    """assert_snapshot accepts dict or str result."""
    with tempfile.TemporaryDirectory() as tmp:
        tester = SnapshotTester(snapshot_dir=tmp)
        await tester.assert_snapshot("dict_payload", {"output": "from dict"})
        await tester.assert_snapshot("str_payload", "raw string output")
        assert (Path(tmp) / "dict_payload.json").exists()
        assert (Path(tmp) / "str_payload.json").exists()
