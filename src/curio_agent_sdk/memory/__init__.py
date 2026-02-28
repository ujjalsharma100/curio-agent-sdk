"""
Memory system for the Curio Agent SDK.

Provides different memory types for agent use:
- ConversationMemory: Short-term sliding window of recent turns
- VectorMemory: Long-term semantic memory using embeddings (optional persist_path)
- KeyValueMemory: Simple structured key-value store
- CompositeMemory: Combines multiple memory types
- WorkingMemory: Ephemeral scratchpad for current task
- EpisodicMemory: Experiences with temporal context
- GraphMemory: Entity-relationship knowledge graph
- SelfEditingMemory: Core + archival memory with agent tools (MemGPT/Letta style)
- FileMemory: File-based persistence (Claude Code style)

Example:
    from curio_agent_sdk.memory import ConversationMemory, VectorMemory, CompositeMemory

    memory = CompositeMemory({
        "conversation": ConversationMemory(max_entries=50),
        "knowledge": VectorMemory(persist_path="~/.agent/vector_memory.json"),
    })

    agent = Agent(memory=memory, ...)
"""

from curio_agent_sdk.memory.base import Memory, MemoryEntry
from curio_agent_sdk.memory.conversation import ConversationMemory
from curio_agent_sdk.memory.vector import VectorMemory
from curio_agent_sdk.memory.key_value import KeyValueMemory
from curio_agent_sdk.memory.composite import CompositeMemory
from curio_agent_sdk.memory.working import WorkingMemory
from curio_agent_sdk.memory.episodic import EpisodicMemory, Episode
from curio_agent_sdk.memory.graph import GraphMemory, Triple
from curio_agent_sdk.memory.self_editing import SelfEditingMemory
from curio_agent_sdk.memory.file_memory import FileMemory
from curio_agent_sdk.memory.manager import (
    MemoryManager,
    MemoryInjectionStrategy,
    MemorySaveStrategy,
    MemoryQueryStrategy,
    DefaultInjection,
    UserMessageInjection,
    NoInjection,
    DefaultSave,
    SaveEverythingStrategy,
    SaveSummaryStrategy,
    NoSave,
    PerIterationSave,
    DefaultQuery,
    KeywordQuery,
    AdaptiveTokenQuery,
)
from curio_agent_sdk.memory.policies import (
    importance_score,
    decay_score,
    combined_relevance,
    summarize_old_memories,
)

__all__ = [
    "Memory",
    "MemoryEntry",
    "ConversationMemory",
    "VectorMemory",
    "KeyValueMemory",
    "CompositeMemory",
    "WorkingMemory",
    "EpisodicMemory",
    "Episode",
    "GraphMemory",
    "Triple",
    "SelfEditingMemory",
    "FileMemory",
    # Manager and strategies
    "MemoryManager",
    "MemoryInjectionStrategy",
    "MemorySaveStrategy",
    "MemoryQueryStrategy",
    "DefaultInjection",
    "UserMessageInjection",
    "NoInjection",
    "DefaultSave",
    "SaveEverythingStrategy",
    "SaveSummaryStrategy",
    "NoSave",
    "PerIterationSave",
    "DefaultQuery",
    "KeywordQuery",
    "AdaptiveTokenQuery",
    # Policies (decay, importance, summarization)
    "importance_score",
    "decay_score",
    "combined_relevance",
    "summarize_old_memories",
]
