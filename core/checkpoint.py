"""
Checkpointing and recovery for agent state.

Allows saving agent state snapshots during execution and resuming
from them after crashes or interruptions.
"""

from __future__ import annotations

import json
import logging
import os
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Any

from curio_agent_sdk.models.llm import Message, ToolCall, ContentBlock

logger = logging.getLogger(__name__)


def _serialize_message(msg: Message) -> dict[str, Any]:
    """Serialize a Message to a JSON-compatible dict."""
    data: dict[str, Any] = {
        "role": msg.role,
        "name": msg.name,
        "tool_call_id": msg.tool_call_id,
    }

    # Serialize content
    if msg.content is None:
        data["content"] = None
    elif isinstance(msg.content, str):
        data["content"] = msg.content
    elif isinstance(msg.content, list):
        blocks = []
        for block in msg.content:
            b: dict[str, Any] = {"type": block.type}
            if block.text is not None:
                b["text"] = block.text
            if block.image_url is not None:
                b["image_url"] = block.image_url
            if block.tool_call is not None:
                b["tool_call"] = {
                    "id": block.tool_call.id,
                    "name": block.tool_call.name,
                    "arguments": block.tool_call.arguments,
                }
            if block.tool_call_id is not None:
                b["tool_call_id"] = block.tool_call_id
            blocks.append(b)
        data["content"] = blocks

    # Serialize tool calls
    if msg.tool_calls:
        data["tool_calls"] = [
            {"id": tc.id, "name": tc.name, "arguments": tc.arguments}
            for tc in msg.tool_calls
        ]
    else:
        data["tool_calls"] = None

    return data


def _deserialize_message(data: dict[str, Any]) -> Message:
    """Deserialize a Message from a dict."""
    content = data.get("content")
    if isinstance(content, list):
        blocks = []
        for b in content:
            tc_data = b.get("tool_call")
            tc = ToolCall(**tc_data) if tc_data else None
            blocks.append(ContentBlock(
                type=b["type"],
                text=b.get("text"),
                image_url=b.get("image_url"),
                tool_call=tc,
                tool_call_id=b.get("tool_call_id"),
            ))
        content = blocks

    tool_calls = None
    if data.get("tool_calls"):
        tool_calls = [ToolCall(**tc) for tc in data["tool_calls"]]

    return Message(
        role=data["role"],
        content=content,
        tool_calls=tool_calls,
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
    )


@dataclass
class Checkpoint:
    """
    A serializable snapshot of agent state.

    Captures everything needed to resume an agent run:
    - Full message history
    - Iteration counter
    - Metrics
    - Metadata
    """
    run_id: str
    agent_id: str
    iteration: int
    timestamp: datetime = field(default_factory=datetime.now)
    messages: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Metrics at checkpoint time
    total_llm_calls: int = 0
    total_tool_calls: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0

    def serialize(self) -> bytes:
        """Serialize the checkpoint to bytes (JSON)."""
        data = {
            "run_id": self.run_id,
            "agent_id": self.agent_id,
            "iteration": self.iteration,
            "timestamp": self.timestamp.isoformat(),
            "messages": self.messages,
            "metadata": self.metadata,
            "total_llm_calls": self.total_llm_calls,
            "total_tool_calls": self.total_tool_calls,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
        }
        return json.dumps(data, default=str).encode("utf-8")

    @classmethod
    def deserialize(cls, data: bytes) -> Checkpoint:
        """Deserialize a checkpoint from bytes."""
        d = json.loads(data.decode("utf-8"))
        return cls(
            run_id=d["run_id"],
            agent_id=d["agent_id"],
            iteration=d["iteration"],
            timestamp=datetime.fromisoformat(d["timestamp"]),
            messages=d.get("messages", []),
            metadata=d.get("metadata", {}),
            total_llm_calls=d.get("total_llm_calls", 0),
            total_tool_calls=d.get("total_tool_calls", 0),
            total_input_tokens=d.get("total_input_tokens", 0),
            total_output_tokens=d.get("total_output_tokens", 0),
        )

    @classmethod
    def from_state(cls, state: Any, run_id: str, agent_id: str) -> Checkpoint:
        """
        Create a checkpoint from an AgentState.

        Args:
            state: The AgentState to snapshot.
            run_id: The current run ID.
            agent_id: The agent ID.
        """
        messages = [_serialize_message(m) for m in state.messages]

        return cls(
            run_id=run_id,
            agent_id=agent_id,
            iteration=state.iteration,
            messages=messages,
            metadata=dict(state.metadata),
            total_llm_calls=state.total_llm_calls,
            total_tool_calls=state.total_tool_calls,
            total_input_tokens=state.total_input_tokens,
            total_output_tokens=state.total_output_tokens,
        )

    def restore_messages(self) -> list[Message]:
        """Deserialize the stored messages back to Message objects."""
        return [_deserialize_message(m) for m in self.messages]


class CheckpointStore(ABC):
    """
    Abstract base class for checkpoint storage.

    Implementations can persist checkpoints to files, databases,
    cloud storage, etc.
    """

    @abstractmethod
    async def save(self, checkpoint: Checkpoint) -> None:
        """Save a checkpoint."""
        ...

    @abstractmethod
    async def load(self, run_id: str) -> Checkpoint | None:
        """Load the latest checkpoint for a run."""
        ...

    @abstractmethod
    async def list_checkpoints(self, agent_id: str) -> list[Checkpoint]:
        """List all checkpoints for an agent (most recent first)."""
        ...

    async def delete(self, run_id: str) -> bool:
        """Delete checkpoints for a run. Returns True if any were deleted."""
        return False


class InMemoryCheckpointStore(CheckpointStore):
    """
    In-memory checkpoint store for testing and development.

    Checkpoints are lost when the process exits.
    """

    def __init__(self):
        self._checkpoints: dict[str, Checkpoint] = {}  # run_id -> latest

    async def save(self, checkpoint: Checkpoint) -> None:
        self._checkpoints[checkpoint.run_id] = checkpoint

    async def load(self, run_id: str) -> Checkpoint | None:
        return self._checkpoints.get(run_id)

    async def list_checkpoints(self, agent_id: str) -> list[Checkpoint]:
        results = [
            cp for cp in self._checkpoints.values()
            if cp.agent_id == agent_id
        ]
        results.sort(key=lambda c: c.timestamp, reverse=True)
        return results

    async def delete(self, run_id: str) -> bool:
        return self._checkpoints.pop(run_id, None) is not None


class FileCheckpointStore(CheckpointStore):
    """
    File-based checkpoint store.

    Saves checkpoints as JSON files in a directory.
    Each run gets one file (overwritten on each save).

    Example:
        store = FileCheckpointStore("./checkpoints")
        agent = Agent(checkpoint_store=store, ...)
    """

    def __init__(self, directory: str | Path):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)

    def _run_path(self, run_id: str) -> Path:
        # Sanitize run_id for filesystem
        safe_id = run_id.replace("/", "_").replace("\\", "_")
        return self.directory / f"{safe_id}.json"

    async def save(self, checkpoint: Checkpoint) -> None:
        path = self._run_path(checkpoint.run_id)
        data = checkpoint.serialize()
        # Write atomically via temp file
        tmp_path = path.with_suffix(".tmp")
        tmp_path.write_bytes(data)
        tmp_path.rename(path)
        logger.debug("Checkpoint saved: %s (iteration %d)", path, checkpoint.iteration)

    async def load(self, run_id: str) -> Checkpoint | None:
        path = self._run_path(run_id)
        if not path.exists():
            return None
        data = path.read_bytes()
        return Checkpoint.deserialize(data)

    async def list_checkpoints(self, agent_id: str) -> list[Checkpoint]:
        results: list[Checkpoint] = []
        for path in self.directory.glob("*.json"):
            try:
                data = path.read_bytes()
                cp = Checkpoint.deserialize(data)
                if cp.agent_id == agent_id:
                    results.append(cp)
            except Exception as e:
                logger.warning("Failed to read checkpoint %s: %s", path, e)
        results.sort(key=lambda c: c.timestamp, reverse=True)
        return results

    async def delete(self, run_id: str) -> bool:
        path = self._run_path(run_id)
        if path.exists():
            path.unlink()
            return True
        return False
