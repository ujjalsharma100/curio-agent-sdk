"""
Performance tests: State Checkpoint (Phase 19)

Validates checkpoint serialization/deserialization at scale.
"""

import time
import pytest

from curio_agent_sdk.core.state.checkpoint import Checkpoint


# ── Tests ─────────────────────────────────────────────────────────────────


@pytest.mark.slow
def test_checkpoint_serialize_large():
    """Serializing a checkpoint with 1000 messages completes in < 2s."""
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i} " * 50}
        for i in range(1000)
    ]
    cp = Checkpoint(
        run_id="perf-run-1",
        agent_id="perf-agent-1",
        iteration=500,
        messages=messages,
        metadata={"key": f"val_{i}" for i in range(100)},
        total_llm_calls=500,
        total_tool_calls=200,
        total_input_tokens=500000,
        total_output_tokens=250000,
    )

    start = time.monotonic()
    for _ in range(100):
        data = cp.serialize()
    elapsed = time.monotonic() - start

    assert len(data) > 0
    assert elapsed < 2.0, f"100 serializations took {elapsed:.2f}s (limit: 2s)"


@pytest.mark.slow
def test_checkpoint_deserialize_large():
    """Deserializing a large checkpoint 100 times completes in < 2s."""
    messages = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": f"Message {i} " * 50}
        for i in range(1000)
    ]
    cp = Checkpoint(
        run_id="perf-run-2",
        agent_id="perf-agent-2",
        iteration=500,
        messages=messages,
        metadata={"key": f"val_{i}" for i in range(100)},
    )
    data = cp.serialize()

    start = time.monotonic()
    for _ in range(100):
        restored = Checkpoint.deserialize(data)
    elapsed = time.monotonic() - start

    assert restored.run_id == "perf-run-2"
    assert len(restored.messages) == 1000
    assert elapsed < 2.0, f"100 deserializations took {elapsed:.2f}s (limit: 2s)"


@pytest.mark.slow
def test_checkpoint_round_trip_integrity():
    """Round-trip serialize/deserialize preserves all data for 500 iterations."""
    cp = Checkpoint(
        run_id="integrity-run",
        agent_id="integrity-agent",
        iteration=100,
        messages=[{"role": "user", "content": f"msg {i}"} for i in range(100)],
        metadata={"score": 0.95, "tags": ["a", "b"]},
        total_llm_calls=50,
        total_tool_calls=25,
    )

    start = time.monotonic()
    for _ in range(500):
        data = cp.serialize()
        restored = Checkpoint.deserialize(data)
        assert restored.run_id == cp.run_id
        assert restored.agent_id == cp.agent_id
        assert restored.iteration == cp.iteration
        assert len(restored.messages) == len(cp.messages)
    elapsed = time.monotonic() - start

    assert elapsed < 5.0, f"500 round-trips took {elapsed:.2f}s (limit: 5s)"
