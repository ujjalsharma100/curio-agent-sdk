"""
Unit tests for curio_agent_sdk.core.state.checkpoint — Checkpoint serialize/deserialize.

Covers: checkpoint creation, serialize, deserialize, roundtrip, from_state,
restore_messages, extensions, transition_history, large state, corrupted data.
"""

import pytest

from curio_agent_sdk.core.state.state import AgentState, StateExtension
from curio_agent_sdk.core.state.checkpoint import Checkpoint
from curio_agent_sdk.models.llm import Message


# ---------------------------------------------------------------------------
# StateExtension for tests
# ---------------------------------------------------------------------------


class _SampleExtension(StateExtension):
    def __init__(self, value: str):
        self.value = value

    def to_dict(self):
        return {"value": self.value}

    @classmethod
    def from_dict(cls, data: dict):
        return cls(data["value"])


# ---------------------------------------------------------------------------
# Checkpoint creation and serialization
# ---------------------------------------------------------------------------


@pytest.mark.unit
class TestCheckpoint:
    def test_checkpoint_creation(self):
        """Create checkpoint dataclass."""
        cp = Checkpoint(
            run_id="r1",
            agent_id="a1",
            iteration=2,
            messages=[{"role": "user", "content": "hi"}],
            metadata={"k": "v"},
        )
        assert cp.run_id == "r1"
        assert cp.agent_id == "a1"
        assert cp.iteration == 2
        assert len(cp.messages) == 1
        assert cp.messages[0]["content"] == "hi"
        assert cp.metadata == {"k": "v"}
        assert cp.total_llm_calls == 0
        assert cp.extensions == {}
        assert cp.transition_history == []

    def test_checkpoint_serialize(self):
        """Serialize to bytes."""
        cp = Checkpoint(
            run_id="r1",
            agent_id="a1",
            iteration=1,
            messages=[{"role": "user", "content": "hello"}],
        )
        data = cp.serialize()
        assert isinstance(data, bytes)
        assert b"r1" in data
        assert b"hello" in data

    def test_checkpoint_deserialize(self):
        """Deserialize from bytes."""
        cp = Checkpoint(
            run_id="r2",
            agent_id="a2",
            iteration=3,
            messages=[{"role": "assistant", "content": "world"}],
            metadata={"x": 1},
        )
        raw = cp.serialize()
        restored = Checkpoint.deserialize(raw)
        assert restored.run_id == cp.run_id
        assert restored.agent_id == cp.agent_id
        assert restored.iteration == cp.iteration
        assert restored.messages == cp.messages
        assert restored.metadata == cp.metadata

    def test_checkpoint_roundtrip(self):
        """Serialize → deserialize = same."""
        cp = Checkpoint(
            run_id="round",
            agent_id="ag",
            iteration=5,
            messages=[
                {"role": "system", "content": "You are helpful."},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
            ],
            metadata={"a": 1, "b": "two"},
            total_llm_calls=2,
            total_tool_calls=1,
            total_input_tokens=100,
            total_output_tokens=50,
            transition_history=[("plan", 1.0), ("exec", 2.0)],
        )
        restored = Checkpoint.deserialize(cp.serialize())
        assert restored.run_id == cp.run_id
        assert restored.agent_id == cp.agent_id
        assert restored.iteration == cp.iteration
        assert restored.messages == cp.messages
        assert restored.metadata == cp.metadata
        assert restored.total_llm_calls == cp.total_llm_calls
        assert restored.total_tool_calls == cp.total_tool_calls
        assert restored.total_input_tokens == cp.total_input_tokens
        assert restored.total_output_tokens == cp.total_output_tokens
        assert restored.transition_history == cp.transition_history

    def test_checkpoint_from_state(self):
        """Create from AgentState."""
        state = AgentState(
            messages=[Message.user("from-state"), Message.assistant("ok")],
            iteration=4,
            metadata={"src": "state"},
            total_llm_calls=2,
            total_tool_calls=1,
            total_input_tokens=10,
            total_output_tokens=5,
        )
        state.record_transition("phase_a")
        state.record_transition("phase_b")
        cp = Checkpoint.from_state(state, run_id="fs", agent_id="ag")
        assert cp.run_id == "fs"
        assert cp.agent_id == "ag"
        assert cp.iteration == 4
        assert len(cp.messages) == 2
        assert cp.metadata == {"src": "state"}
        assert cp.total_llm_calls == 2
        assert cp.total_tool_calls == 1
        assert len(cp.transition_history) == 2
        assert cp.transition_history[0][0] == "phase_a"
        assert cp.transition_history[1][0] == "phase_b"

    def test_checkpoint_restore_messages(self):
        """Restore Message objects from serialized messages."""
        state = AgentState(
            messages=[
                Message.system("Sys"),
                Message.user("User"),
                Message.assistant("Assistant"),
            ],
        )
        cp = Checkpoint.from_state(state, run_id="rm", agent_id="ag")
        restored = cp.restore_messages()
        assert len(restored) == 3
        assert restored[0].role == "system"
        assert restored[0].content == "Sys"
        assert restored[1].role == "user"
        assert restored[1].content == "User"
        assert restored[2].role == "assistant"
        assert restored[2].content == "Assistant"

    def test_checkpoint_with_extensions(self):
        """Extensions serialized in checkpoint."""
        state = AgentState()
        state.set_ext(_SampleExtension("ext-value"))
        cp = Checkpoint.from_state(state, run_id="ex", agent_id="ag")
        assert cp.extensions
        key = f"{_SampleExtension.__module__}.{_SampleExtension.__qualname__}"
        assert key in cp.extensions or any("_SampleExtension" in k for k in cp.extensions)
        # Restore state and set extensions from checkpoint
        state2 = AgentState()
        state2.set_extensions_from_checkpoint(cp.extensions)
        ext = state2.get_ext(_SampleExtension)
        assert ext is not None
        assert ext.value == "ext-value"

    def test_checkpoint_with_transition_history(self):
        """Transition history preserved."""
        state = AgentState()
        state.record_transition("init")
        state.record_transition("loop")
        cp = Checkpoint.from_state(state, run_id="th", agent_id="ag")
        assert len(cp.transition_history) == 2
        state2 = AgentState()
        state2.set_transition_history(cp.transition_history)
        assert state2.current_phase == "loop"
        assert state2.get_transition_history() == cp.transition_history

    def test_checkpoint_large_state(self):
        """Large message list handling."""
        state = AgentState(
            messages=[Message.user(f"msg-{i}") for i in range(500)],
            iteration=10,
        )
        cp = Checkpoint.from_state(state, run_id="large", agent_id="ag")
        restored = cp.restore_messages()
        assert len(restored) == 500
        assert restored[0].content == "msg-0"
        assert restored[499].content == "msg-499"

    def test_checkpoint_corrupted_data(self):
        """Graceful error on bad data."""
        with pytest.raises(Exception):  # json.JSONDecodeError or similar
            Checkpoint.deserialize(b"not valid json {{{")
        with pytest.raises(Exception):
            Checkpoint.deserialize(b"")
        with pytest.raises(Exception):
            Checkpoint.deserialize(b'{"run_id": "x"}')  # missing required fields
