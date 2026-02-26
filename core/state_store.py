"""
StateStore — unified abstraction for saving/loading agent state.

Replaces the previous CheckpointStore with a cleaner API that
operates on AgentState directly.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.core.checkpoint import Checkpoint, _serialize_message, _deserialize_message

logger = logging.getLogger(__name__)


class StateStore(ABC):
    """
    Abstract base for persisting agent state across runs.

    Implementations can store state in memory, files, databases,
    or cloud storage.
    """

    @abstractmethod
    async def save(self, state: AgentState, run_id: str, agent_id: str) -> None:
        """Save agent state for a given run."""
        ...

    @abstractmethod
    async def load(self, run_id: str) -> AgentState | None:
        """Load agent state for a given run. Returns None if not found."""
        ...

    @abstractmethod
    async def list_runs(self, agent_id: str) -> list[str]:
        """List all run IDs for an agent (most recent first)."""
        ...

    @abstractmethod
    async def delete(self, run_id: str) -> bool:
        """Delete state for a run. Returns True if deleted."""
        ...


class InMemoryStateStore(StateStore):
    """
    In-memory state store for testing and development.

    State is lost when the process exits.
    """

    def __init__(self):
        self._states: dict[str, tuple[Checkpoint, list]] = {}  # run_id -> (checkpoint, tool_schemas)

    async def save(self, state: AgentState, run_id: str, agent_id: str) -> None:
        checkpoint = Checkpoint.from_state(state, run_id=run_id, agent_id=agent_id)
        self._states[run_id] = (checkpoint, state.tool_schemas)

    async def load(self, run_id: str) -> AgentState | None:
        entry = self._states.get(run_id)
        if entry is None:
            return None

        checkpoint, tool_schemas = entry
        messages = checkpoint.restore_messages()
        return AgentState(
            messages=messages,
            tool_schemas=tool_schemas,
            iteration=checkpoint.iteration,
            max_iterations=25,
            metadata=checkpoint.metadata,
            total_llm_calls=checkpoint.total_llm_calls,
            total_tool_calls=checkpoint.total_tool_calls,
            total_input_tokens=checkpoint.total_input_tokens,
            total_output_tokens=checkpoint.total_output_tokens,
        )

    async def list_runs(self, agent_id: str) -> list[str]:
        runs = []
        for run_id, (checkpoint, _) in self._states.items():
            if checkpoint.agent_id == agent_id:
                runs.append((run_id, checkpoint.timestamp))
        runs.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in runs]

    async def delete(self, run_id: str) -> bool:
        return self._states.pop(run_id, None) is not None


class FileStateStore(StateStore):
    """
    File-based state store.

    Saves state as JSON files in a directory.
    Each run gets one file (overwritten on each save).

    Example:
        store = FileStateStore("./state")
        agent = Agent.builder().state_store(store).build()
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        safe_id = run_id.replace("/", "_").replace("\\", "_")
        return self.directory / f"{safe_id}.json"

    async def save(self, state: AgentState, run_id: str, agent_id: str) -> None:
        checkpoint = Checkpoint.from_state(state, run_id=run_id, agent_id=agent_id)
        path = self._run_path(run_id)
        data = checkpoint.serialize()
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_bytes(data)
        tmp_path.rename(path)
        logger.debug("State saved: %s (iteration %d)", path, checkpoint.iteration)

    async def load(self, run_id: str) -> AgentState | None:
        path = self._run_path(run_id)
        if not path.exists():
            return None

        try:
            data = path.read_bytes()
            checkpoint = Checkpoint.deserialize(data)
            messages = checkpoint.restore_messages()
            return AgentState(
                messages=messages,
                iteration=checkpoint.iteration,
                max_iterations=25,
                metadata=checkpoint.metadata,
                total_llm_calls=checkpoint.total_llm_calls,
                total_tool_calls=checkpoint.total_tool_calls,
                total_input_tokens=checkpoint.total_input_tokens,
                total_output_tokens=checkpoint.total_output_tokens,
            )
        except Exception as e:
            logger.warning("Failed to load state for run %s: %s", run_id, e)
            return None

    async def list_runs(self, agent_id: str) -> list[str]:
        runs = []
        for path in self.directory.glob("*.json"):
            try:
                data = path.read_bytes()
                checkpoint = Checkpoint.deserialize(data)
                if checkpoint.agent_id == agent_id:
                    runs.append((checkpoint.run_id, checkpoint.timestamp))
            except Exception as e:
                logger.warning("Failed to read state %s: %s", path, e)
        runs.sort(key=lambda x: x[1], reverse=True)
        return [r[0] for r in runs]

    async def delete(self, run_id: str) -> bool:
        path = self._run_path(run_id)
        if path.exists():
            path.unlink()
            return True
        return False
