"""
MemoryManager — orchestrates how memory is used in the agent lifecycle.

All behavior is customizable via pluggable strategies:

- MemoryInjectionStrategy: Controls HOW memory is injected into conversation
- MemorySaveStrategy: Controls WHAT and WHEN memory is saved
- MemoryQueryStrategy: Controls HOW memory is queried

Users can swap strategies to completely change memory behavior without
modifying the agent or runtime code.

Example (default behavior):
    manager = MemoryManager(memory=ConversationMemory())
    # Uses DefaultInjection, DefaultSave, DefaultQuery

Example (custom strategies):
    manager = MemoryManager(
        memory=CompositeMemory({...}),
        injection_strategy=UserMessageInjection(max_tokens=4000),
        save_strategy=SaveEverythingStrategy(),
        query_strategy=SummarizedQuery(llm=my_llm),
    )

Example (agent-facing memory tools):
    tools = manager.get_tools()
    agent = Agent(tools=[*my_tools, *tools], ...)
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from typing import Any, TYPE_CHECKING

from curio_agent_sdk.memory.base import Memory, MemoryEntry
from curio_agent_sdk.models.llm import Message

if TYPE_CHECKING:
    from curio_agent_sdk.core.state import AgentState
    from curio_agent_sdk.core.tools.tool import Tool

logger = logging.getLogger(__name__)


# ════════════════════════════════════════════════════════════════════════
# Strategy ABCs
# ════════════════════════════════════════════════════════════════════════

class MemoryInjectionStrategy(ABC):
    """
    Controls HOW memory context is injected into the agent's conversation.

    Override to change:
    - Where memory appears (system message, user message, separate role)
    - How memory is formatted
    - Whether memory is always injected or conditionally
    - Token budget for memory context
    """

    @abstractmethod
    async def inject(self, state: AgentState, memory: Memory, query: str) -> None:
        """
        Inject memory context into the agent state (mutates state in-place).

        Args:
            state: The agent state to inject memory into.
            memory: The memory backend to query.
            query: The user's input (used for relevance-based retrieval).
        """
        ...


class MemorySaveStrategy(ABC):
    """
    Controls WHAT gets saved to memory and WHEN.

    Override to change:
    - What content is saved (raw input, summarized, tool results, etc.)
    - When saving happens (on run start, end, per-iteration, per-tool-call)
    - How content is formatted before saving
    - What metadata is attached
    """

    async def on_run_start(self, memory: Memory, input_text: str, state: AgentState) -> None:
        """Called at the start of a run, before the first iteration."""
        pass

    async def on_run_end(
        self, memory: Memory, input_text: str, output: str, state: AgentState
    ) -> None:
        """Called at the end of a successful run."""
        pass

    async def on_run_error(
        self, memory: Memory, input_text: str, error: str, state: AgentState
    ) -> None:
        """Called when a run ends with an error."""
        pass

    async def on_iteration(self, memory: Memory, state: AgentState, iteration: int) -> None:
        """Called after each loop iteration."""
        pass

    async def on_tool_result(
        self, memory: Memory, tool_name: str, tool_args: dict, result: Any, state: AgentState
    ) -> None:
        """Called after each tool execution."""
        pass


class MemoryQueryStrategy(ABC):
    """
    Controls HOW memory is queried for context retrieval.

    Override to change:
    - How the query is constructed (raw input, keywords, summary, multi-query)
    - Token budget for memory results
    - Relevance thresholds
    - Number of results to retrieve
    """

    @abstractmethod
    async def build_query(self, input_text: str, state: AgentState) -> str:
        """
        Build the query string used to search memory.

        Args:
            input_text: The user's raw input.
            state: Current agent state (may have prior context).

        Returns:
            The query string to pass to memory.search() / memory.get_context().
        """
        ...

    @abstractmethod
    def max_tokens(self, state: AgentState) -> int:
        """
        Return the max token budget for memory context.

        Args:
            state: Current agent state.

        Returns:
            Max tokens to allocate for memory context.
        """
        ...

    def relevance_threshold(self) -> float:
        """
        Minimum relevance score for memory entries to be included.

        Returns:
            Threshold between 0.0 and 1.0. Default 0.0 (include all).
        """
        return 0.0

    def max_results(self) -> int:
        """Maximum number of memory entries to retrieve."""
        return 10


# ════════════════════════════════════════════════════════════════════════
# Default strategy implementations
# ════════════════════════════════════════════════════════════════════════

class DefaultInjection(MemoryInjectionStrategy):
    """
    Default injection: query memory with user input, insert as system
    message at position 1 (after the main system prompt).

    Args:
        max_tokens: Max tokens for memory context. Default 2000.
        position: Where to insert the memory message. Default 1 (after system prompt).
        prefix: Text prefix for the memory message. Default "Relevant information from memory:".
    """

    def __init__(
        self,
        max_tokens: int = 2000,
        position: int = 1,
        prefix: str = "Relevant information from memory:",
    ):
        self._max_tokens = max_tokens
        self._position = position
        self._prefix = prefix

    async def inject(self, state: AgentState, memory: Memory, query: str) -> None:
        memory_context = await memory.get_context(query, max_tokens=self._max_tokens)
        if memory_context:
            state.messages.insert(
                self._position,
                Message.system(f"{self._prefix}\n{memory_context}"),
            )


class UserMessageInjection(MemoryInjectionStrategy):
    """
    Inject memory context as part of the user message instead of
    as a separate system message. Useful when you want memory to
    appear more naturally in the conversation.

    Args:
        max_tokens: Max tokens for memory context.
        prefix: Text prefix before the memory block.
    """

    def __init__(self, max_tokens: int = 2000, prefix: str = "Context from memory:"):
        self._max_tokens = max_tokens
        self._prefix = prefix

    async def inject(self, state: AgentState, memory: Memory, query: str) -> None:
        memory_context = await memory.get_context(query, max_tokens=self._max_tokens)
        if not memory_context:
            return

        # Find the last user message and append memory context to it
        for i in range(len(state.messages) - 1, -1, -1):
            if state.messages[i].role == "user":
                original = state.messages[i].text
                state.messages[i] = Message.user(
                    f"{original}\n\n{self._prefix}\n{memory_context}"
                )
                break


class NoInjection(MemoryInjectionStrategy):
    """Don't inject memory automatically. The agent manages memory via tools."""

    async def inject(self, state: AgentState, memory: Memory, query: str) -> None:
        pass


class DefaultSave(MemorySaveStrategy):
    """
    Default save: save user input and assistant output after successful runs.
    """

    async def on_run_end(
        self, memory: Memory, input_text: str, output: str, state: AgentState
    ) -> None:
        await memory.add(
            f"User: {input_text}",
            metadata={"type": "user_input", "role": "user"},
        )
        if output:
            await memory.add(
                f"Assistant: {output}",
                metadata={"type": "assistant_output", "role": "assistant"},
            )


class SaveEverythingStrategy(MemorySaveStrategy):
    """
    Save user input, assistant output, AND all tool results.
    Useful for agents that need detailed episodic memory.
    """

    async def on_run_end(
        self, memory: Memory, input_text: str, output: str, state: AgentState
    ) -> None:
        await memory.add(
            f"User: {input_text}",
            metadata={"type": "user_input", "role": "user"},
        )
        if output:
            await memory.add(
                f"Assistant: {output}",
                metadata={"type": "assistant_output", "role": "assistant"},
            )

    async def on_tool_result(
        self, memory: Memory, tool_name: str, tool_args: dict, result: Any, state: AgentState
    ) -> None:
        content = f"Tool '{tool_name}' called with {tool_args} returned: {result}"
        await memory.add(
            content,
            metadata={"type": "tool_result", "tool_name": tool_name},
        )


class SaveSummaryStrategy(MemorySaveStrategy):
    """
    Save only a summary of the interaction, not the full input/output.
    Useful for reducing memory size with long conversations.

    You must provide a summarize_fn that takes (input, output, state) -> str.
    """

    def __init__(self, summarize_fn):
        self._summarize_fn = summarize_fn

    async def on_run_end(
        self, memory: Memory, input_text: str, output: str, state: AgentState
    ) -> None:
        summary = await self._summarize_fn(input_text, output, state)
        if summary:
            await memory.add(
                summary,
                metadata={"type": "interaction_summary"},
            )


class NoSave(MemorySaveStrategy):
    """Don't save anything automatically. The agent manages memory via tools."""
    pass


class PerIterationSave(MemorySaveStrategy):
    """
    Save after every loop iteration in addition to run end.
    Useful for long-running agents where you want to capture progress.
    """

    async def on_run_end(
        self, memory: Memory, input_text: str, output: str, state: AgentState
    ) -> None:
        await memory.add(
            f"User: {input_text}",
            metadata={"type": "user_input", "role": "user"},
        )
        if output:
            await memory.add(
                f"Assistant: {output}",
                metadata={"type": "assistant_output", "role": "assistant"},
            )

    async def on_iteration(self, memory: Memory, state: AgentState, iteration: int) -> None:
        last_msg = state.last_message
        if last_msg and last_msg.role == "assistant" and last_msg.text:
            await memory.add(
                f"[Iteration {iteration}] {last_msg.text[:500]}",
                metadata={"type": "iteration_snapshot", "iteration": iteration},
            )


class DefaultQuery(MemoryQueryStrategy):
    """
    Default query: use raw input as the query, 2000 token budget.

    Args:
        max_tokens_value: Max tokens for memory context. Default 2000.
    """

    def __init__(self, max_tokens_value: int = 2000):
        self._max_tokens = max_tokens_value

    async def build_query(self, input_text: str, state: AgentState) -> str:
        return input_text

    def max_tokens(self, state: AgentState) -> int:
        return self._max_tokens


class KeywordQuery(MemoryQueryStrategy):
    """
    Extract keywords from input for more focused memory retrieval.
    Simple keyword extraction — strips common stop words.

    Args:
        max_tokens_value: Max tokens for memory context.
    """

    _STOP_WORDS = frozenset({
        "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "shall", "to", "of", "in", "for",
        "on", "with", "at", "by", "from", "as", "into", "through", "during",
        "before", "after", "above", "below", "between", "and", "but", "or",
        "not", "no", "so", "if", "than", "too", "very", "just", "about",
        "what", "how", "when", "where", "who", "which", "that", "this",
        "these", "those", "it", "its", "i", "me", "my", "we", "our",
        "you", "your", "he", "she", "they", "them", "their",
    })

    def __init__(self, max_tokens_value: int = 2000):
        self._max_tokens = max_tokens_value

    async def build_query(self, input_text: str, state: AgentState) -> str:
        words = input_text.lower().split()
        keywords = [w for w in words if w not in self._STOP_WORDS and len(w) > 2]
        return " ".join(keywords) if keywords else input_text

    def max_tokens(self, state: AgentState) -> int:
        return self._max_tokens


class AdaptiveTokenQuery(MemoryQueryStrategy):
    """
    Adjusts token budget based on how many messages are already in the
    conversation. More messages = less room for memory.

    Args:
        base_tokens: Starting token budget when conversation is short.
        min_tokens: Minimum token budget even when conversation is long.
        decay_per_message: Tokens to subtract per existing message.
    """

    def __init__(
        self,
        base_tokens: int = 4000,
        min_tokens: int = 500,
        decay_per_message: int = 100,
    ):
        self._base_tokens = base_tokens
        self._min_tokens = min_tokens
        self._decay_per_message = decay_per_message

    async def build_query(self, input_text: str, state: AgentState) -> str:
        return input_text

    def max_tokens(self, state: AgentState) -> int:
        msg_count = len(state.messages)
        budget = self._base_tokens - (msg_count * self._decay_per_message)
        return max(self._min_tokens, budget)


# ════════════════════════════════════════════════════════════════════════
# MemoryManager
# ════════════════════════════════════════════════════════════════════════

class MemoryManager:
    """
    Orchestrates how memory is used in the agent lifecycle.

    All behavior is customizable via pluggable strategies. The MemoryManager
    is the single point of contact between the Runtime and the Memory backend.

    Example:
        # Default
        manager = MemoryManager(memory=ConversationMemory())

        # Custom injection: use more tokens, inject as user message
        manager = MemoryManager(
            memory=CompositeMemory({...}),
            injection_strategy=UserMessageInjection(max_tokens=4000),
        )

        # Custom save: save everything including tool results
        manager = MemoryManager(
            memory=VectorMemory(),
            save_strategy=SaveEverythingStrategy(),
        )

        # Custom query: adaptive token budget
        manager = MemoryManager(
            memory=VectorMemory(),
            query_strategy=AdaptiveTokenQuery(base_tokens=4000),
        )

        # No automatic behavior — agent manages memory via tools
        manager = MemoryManager(
            memory=KeyValueMemory(),
            injection_strategy=NoInjection(),
            save_strategy=NoSave(),
        )
    """

    def __init__(
        self,
        memory: Memory,
        injection_strategy: MemoryInjectionStrategy | None = None,
        save_strategy: MemorySaveStrategy | None = None,
        query_strategy: MemoryQueryStrategy | None = None,
        namespace: str | None = None,
    ):
        self.memory = memory
        self.injection_strategy = injection_strategy or DefaultInjection()
        self.save_strategy = save_strategy or DefaultSave()
        self.query_strategy = query_strategy or DefaultQuery()
        self.namespace = namespace

    # ── Injection (called by Runtime before run) ────────────────────

    async def inject(self, state: AgentState, input_text: str) -> None:
        """
        Inject memory context into the agent state.

        Uses the query strategy to build the query, then the injection
        strategy to inject the results into the conversation.
        """
        try:
            query = await self.query_strategy.build_query(input_text, state)
            await self.injection_strategy.inject(state, self.memory, query)
        except Exception as e:
            logger.warning("Failed to inject memory context: %s", e)

    # ── Save (called by Runtime at various lifecycle points) ────────

    async def on_run_start(self, input_text: str, state: AgentState) -> None:
        """Called at the start of a run."""
        try:
            await self.save_strategy.on_run_start(self.memory, input_text, state)
        except Exception as e:
            logger.warning("Failed memory on_run_start: %s", e)

    async def on_run_end(self, input_text: str, output: str, state: AgentState) -> None:
        """Called at the end of a successful run."""
        try:
            await self.save_strategy.on_run_end(self.memory, input_text, output, state)
        except Exception as e:
            logger.warning("Failed memory on_run_end: %s", e)

    async def on_run_error(self, input_text: str, error: str, state: AgentState) -> None:
        """Called when a run errors."""
        try:
            await self.save_strategy.on_run_error(self.memory, input_text, error, state)
        except Exception as e:
            logger.warning("Failed memory on_run_error: %s", e)

    async def on_iteration(self, state: AgentState, iteration: int) -> None:
        """Called after each loop iteration."""
        try:
            await self.save_strategy.on_iteration(self.memory, state, iteration)
        except Exception as e:
            logger.warning("Failed memory on_iteration: %s", e)

    async def on_tool_result(
        self, tool_name: str, tool_args: dict, result: Any, state: AgentState
    ) -> None:
        """Called after a tool execution."""
        try:
            await self.save_strategy.on_tool_result(
                self.memory, tool_name, tool_args, result, state
            )
        except Exception as e:
            logger.warning("Failed memory on_tool_result: %s", e)

    # ── Direct memory access ────────────────────────────────────────

    async def add(self, content: str, metadata: dict[str, Any] | None = None) -> str:
        """Direct access: add an entry to memory."""
        meta = dict(metadata or {})
        if self.namespace:
            meta["namespace"] = self.namespace
        return await self.memory.add(content, metadata=meta)

    async def search(self, query: str, limit: int = 5) -> list[MemoryEntry]:
        """Direct access: search memory."""
        return await self.memory.search(query, limit=limit)

    async def get_context(self, query: str, max_tokens: int = 2000) -> str:
        """Direct access: get formatted context."""
        return await self.memory.get_context(query, max_tokens=max_tokens)

    async def clear(self) -> None:
        """Direct access: clear all memory."""
        await self.memory.clear()

    async def count(self) -> int:
        """Direct access: count entries."""
        return await self.memory.count()

    # ── Agent-facing tools ──────────────────────────────────────────

    def get_tools(self) -> list[Tool]:
        """
        Get tools that allow the agent to manage its own memory.

        Returns tools: save_to_memory, search_memory, forget_memory.
        These let the agent decide what to remember and when.

        Example:
            manager = MemoryManager(memory=my_memory)
            tools = manager.get_tools()
            agent = Agent(tools=[*my_tools, *tools], ...)
        """
        from curio_agent_sdk.core.tools.tool import tool as tool_decorator

        memory = self.memory
        namespace = self.namespace

        @tool_decorator
        async def save_to_memory(content: str, tags: str = "") -> str:
            """Save information to memory for future reference. Use this to remember important facts, user preferences, decisions, or anything the agent should recall later. The 'tags' parameter is an optional comma-separated list of tags for categorization."""
            metadata: dict[str, Any] = {"type": "agent_saved", "source": "agent"}
            if tags:
                metadata["tags"] = [t.strip() for t in tags.split(",")]
            if namespace:
                metadata["namespace"] = namespace
            entry_id = await memory.add(content, metadata=metadata)
            return f"Saved to memory (id: {entry_id})"

        @tool_decorator
        async def search_memory(query: str, limit: int = 5) -> str:
            """Search memory for relevant information. Returns matching entries sorted by relevance."""
            results = await memory.search(query, limit=limit)
            if not results:
                return "No relevant memories found."
            lines = []
            for entry in results:
                lines.append(f"- [{entry.id}] (relevance: {entry.relevance:.2f}) {entry.content}")
            return "\n".join(lines)

        @tool_decorator
        async def forget_memory(entry_id: str) -> str:
            """Delete a specific memory entry by its ID. Use when information is outdated or incorrect."""
            deleted = await memory.delete(entry_id)
            if deleted:
                return f"Memory entry {entry_id} deleted."
            return f"Memory entry {entry_id} not found."

        return [save_to_memory, search_memory, forget_memory]

    def __repr__(self) -> str:
        return (
            f"MemoryManager("
            f"memory={self.memory.__class__.__name__}, "
            f"injection={self.injection_strategy.__class__.__name__}, "
            f"save={self.save_strategy.__class__.__name__}, "
            f"query={self.query_strategy.__class__.__name__}"
            f")"
        )
