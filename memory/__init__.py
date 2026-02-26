"""
Memory system for the Curio Agent SDK.

Provides different memory types for agent use:
- ConversationMemory: Short-term sliding window of recent turns
- VectorMemory: Long-term semantic memory using embeddings
- KeyValueMemory: Simple structured key-value store
- CompositeMemory: Combines multiple memory types

Example:
    from curio_agent_sdk.memory import ConversationMemory, VectorMemory, CompositeMemory

    memory = CompositeMemory({
        "conversation": ConversationMemory(max_entries=50),
        "knowledge": VectorMemory(),
    })

    agent = Agent(memory=memory, ...)
"""

from curio_agent_sdk.memory.base import Memory, MemoryEntry
from curio_agent_sdk.memory.conversation import ConversationMemory
from curio_agent_sdk.memory.vector import VectorMemory
from curio_agent_sdk.memory.key_value import KeyValueMemory
from curio_agent_sdk.memory.composite import CompositeMemory
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

__all__ = [
    "Memory",
    "MemoryEntry",
    "ConversationMemory",
    "VectorMemory",
    "KeyValueMemory",
    "CompositeMemory",
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
]
