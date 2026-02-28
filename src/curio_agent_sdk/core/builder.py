"""
Fluent builder for constructing Agent instances.

Provides an ergonomic API for configuring all agent components
while keeping the constructor clean.

Example:
    agent = Agent.builder() \\
        .model("openai:gpt-4o") \\
        .system_prompt("You are a research assistant.") \\
        .tools([search, fetch, analyze]) \\
        .memory_manager(MemoryManager(memory=ConversationMemory())) \\
        .middleware([LoggingMiddleware(), CostTracker(budget=1.0)]) \\
        .max_iterations(25) \\
        .timeout(300) \\
        .on_event(my_event_handler) \\
        .build()
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, TYPE_CHECKING

from curio_agent_sdk.core.tools.tool import Tool
from curio_agent_sdk.core.skills import Skill

if TYPE_CHECKING:
    from curio_agent_sdk.core.subagent import SubagentConfig

if TYPE_CHECKING:
    from curio_agent_sdk.core.agent import Agent
    from curio_agent_sdk.core.loops.base import AgentLoop
    from curio_agent_sdk.core.context import ContextManager
    from curio_agent_sdk.core.state_store import StateStore
    from curio_agent_sdk.core.human_input import HumanInputHandler
    from curio_agent_sdk.core.hooks import HookRegistry
    from curio_agent_sdk.llm.client import LLMClient
    from curio_agent_sdk.middleware.base import Middleware
    from curio_agent_sdk.models.events import AgentEvent
    from curio_agent_sdk.core.plugins import Plugin

logger = logging.getLogger(__name__)


class AgentBuilder:
    """
    Fluent builder for constructing Agent instances.

    All methods return self for chaining. Call build() to create the Agent.

    Example:
        agent = AgentBuilder() \\
            .model("openai:gpt-4o") \\
            .tools([search]) \\
            .build()

        # Or via Agent.builder():
        agent = Agent.builder() \\
            .model("anthropic:claude-sonnet-4-6") \\
            .system_prompt("You are helpful.") \\
            .tools([search, edit_file]) \\
            .build()
    """

    def __init__(self):
        self._config: dict[str, Any] = {}

    # ── Core configuration ──────────────────────────────────────────

    def system_prompt(self, prompt: str) -> AgentBuilder:
        """Set the agent's system prompt."""
        self._config["system_prompt"] = prompt
        return self

    def model(self, model_str: str) -> AgentBuilder:
        """
        Set the model using 'provider:model_name' shorthand.

        Examples:
            .model("openai:gpt-4o")
            .model("anthropic:claude-sonnet-4-6")
            .model("groq:llama-3.1-8b-instant")
        """
        self._config["model"] = model_str
        return self

    def tier(self, tier: str) -> AgentBuilder:
        """Set the default tier for model routing ('tier1', 'tier2', 'tier3')."""
        self._config["tier"] = tier
        return self

    def llm(self, client: LLMClient) -> AgentBuilder:
        """Set a pre-configured LLMClient (overrides model/tier)."""
        self._config["llm"] = client
        return self

    def loop(self, loop: AgentLoop) -> AgentBuilder:
        """Set a custom agent loop (ToolCallingLoop or custom AgentLoop implementation)."""
        self._config["loop"] = loop
        return self

    # ── Tools ───────────────────────────────────────────────────────

    def tools(self, tools: list[Tool | Callable]) -> AgentBuilder:
        """
        Set the agent's tools. Replaces any previously set tools.

        Args:
            tools: List of Tool objects or @tool-decorated callables.
        """
        self._config["tools"] = list(tools)
        return self

    def add_tool(self, tool: Tool | Callable) -> AgentBuilder:
        """Add a single tool to the agent's tool list."""
        if "tools" not in self._config:
            self._config["tools"] = []
        self._config["tools"].append(tool)
        return self

    # ── Identity ────────────────────────────────────────────────────

    def agent_id(self, id: str) -> AgentBuilder:
        """Set the agent's unique ID."""
        self._config["agent_id"] = id
        return self

    def agent_name(self, name: str) -> AgentBuilder:
        """Set the agent's display name."""
        self._config["agent_name"] = name
        return self

    # ── Limits ──────────────────────────────────────────────────────

    def max_iterations(self, n: int) -> AgentBuilder:
        """Set max loop iterations before stopping."""
        self._config["max_iterations"] = n
        return self

    def timeout(self, seconds: float) -> AgentBuilder:
        """Set total run timeout in seconds."""
        self._config["timeout"] = seconds
        return self

    def iteration_timeout(self, seconds: float) -> AgentBuilder:
        """Set per-iteration timeout in seconds."""
        self._config["iteration_timeout"] = seconds
        return self

    def max_tokens(self, n: int) -> AgentBuilder:
        """Set max output tokens per LLM call."""
        self._config["max_tokens"] = n
        return self

    def temperature(self, t: float) -> AgentBuilder:
        """Set LLM temperature."""
        self._config["temperature"] = t
        return self

    # ── Context management ──────────────────────────────────────────

    def context_manager(self, cm: ContextManager) -> AgentBuilder:
        """Set a custom context manager for token budget fitting."""
        self._config["context_manager"] = cm
        return self

    # ── Middleware ───────────────────────────────────────────────────

    def middleware(self, middleware: list[Middleware]) -> AgentBuilder:
        """Set the middleware pipeline. Replaces any previously set middleware."""
        self._config["middleware"] = list(middleware)
        return self

    def add_middleware(self, mw: Middleware) -> AgentBuilder:
        """Add a single middleware to the pipeline."""
        if "middleware" not in self._config:
            self._config["middleware"] = []
        self._config["middleware"].append(mw)
        return self

    # ── Memory ──────────────────────────────────────────────────────

    def memory_manager(self, manager: Any) -> AgentBuilder:
        """
        Set a fully configured MemoryManager with custom strategies.

        Example:
            from curio_agent_sdk.memory.manager import (
                MemoryManager, UserMessageInjection, SaveEverythingStrategy,
            )
            agent = Agent.builder() \\
                .memory_manager(MemoryManager(
                    memory=VectorMemory(),
                    injection_strategy=UserMessageInjection(max_tokens=4000),
                    save_strategy=SaveEverythingStrategy(),
                )) \\
                .build()
        """
        self._config["memory_manager"] = manager
        return self

    # ── Human-in-the-loop ───────────────────────────────────────────

    def human_input(self, handler: HumanInputHandler) -> AgentBuilder:
        """Set the human-in-the-loop handler."""
        self._config["human_input"] = handler
        return self

    def permissions(self, policy: Any) -> AgentBuilder:
        """
        Set the permission policy (allow/deny/ask for tool execution, file and network access).

        Example:
            from curio_agent_sdk.core.permissions import AllowReadsAskWrites, AllowAll
            .permissions(AllowReadsAskWrites())
            .permissions(AllowAll())
        """
        self._config["permission_policy"] = policy
        return self

    # ── State persistence ───────────────────────────────────────────

    def state_store(self, store: StateStore) -> AgentBuilder:
        """Set the state store for resumable execution."""
        self._config["state_store"] = store
        return self

    def session_manager(self, manager: Any) -> AgentBuilder:
        """
        Set the session manager for multi-turn conversation sessions.

        Example:
            from curio_agent_sdk.core.session import SessionManager, InMemorySessionStore
            store = InMemorySessionStore()
            agent = Agent.builder() \\
                .session_manager(SessionManager(store)) \\
                .model("openai:gpt-4o") \\
                .build()
            session = await agent.session_manager.create(agent.agent_id)
            result = await agent.arun("Hello", session_id=session.id)
        """
        self._config["session_manager"] = manager
        return self

    def checkpoint_interval(self, n: int) -> AgentBuilder:
        """Set how often to save state (every N iterations)."""
        self._config["checkpoint_interval"] = n
        return self

    # ── Rules / instructions ────────────────────────────────────────

    def instructions(self, source: Any) -> AgentBuilder:
        """
        Set instructions from an InstructionLoader or raw string.
        Merged with system_prompt when building (instructions appended after base prompt).

        Example:
            .instructions(InstructionLoader())
            .instructions("Always respond in JSON format.")
        """
        self._config["instructions"] = source
        return self

    def instructions_file(self, path: str | Path) -> AgentBuilder:
        """
        Load instructions from a single file. Merged with system_prompt when building.

        Example:
            .instructions_file("./AGENT.md")
        """
        self._config["instructions_file"] = path
        return self

    # ── Hooks & events ──────────────────────────────────────────────

    def hook_registry(self, registry: HookRegistry) -> AgentBuilder:
        """Set a custom hook registry (lifecycle hooks)."""
        self._config["hook_registry"] = registry
        return self

    def hook(
        self,
        event: str,
        handler: Callable,
        *,
        priority: int = 0,
    ) -> AgentBuilder:
        """
        Register a lifecycle hook. Uses the default HookRegistry if none set.

        Example:
            .hook("tool.call.before", lambda ctx: ctx.cancel() if ctx.data.get("tool_name") == "rm" else None)
            .hook("agent.run.after", my_async_handler, priority=10)
        """
        if "hooks" not in self._config:
            self._config["hooks"] = []
        self._config["hooks"].append((event, handler, priority))
        return self

    def on_event(self, callback: Callable[[AgentEvent], None]) -> AgentBuilder:
        """Set the legacy event callback for observability (wired via hook adapter)."""
        self._config["on_event"] = callback
        return self

    # ── Skills ───────────────────────────────────────────────────────

    def skill(self, skill: Skill) -> AgentBuilder:
        """Register a single skill (bundled tools + prompt + hooks)."""
        if "skills" not in self._config:
            self._config["skills"] = []
        self._config["skills"].append(skill)
        return self

    def skills(self, skills: list[Skill]) -> AgentBuilder:
        """Register multiple skills. Replaces any previously set skills."""
        self._config["skills"] = list(skills)
        return self

    # ── Plan mode & todos ────────────────────────────────────────────

    def plan_mode(
        self,
        plan_mode: Any | None = None,
        todo_manager: Any | None = None,
        read_only_tool_names: list[str] | None = None,
    ) -> AgentBuilder:
        """
        Enable plan mode and/or todo tracking. Registers enter_plan_mode, exit_plan_mode,
        approve_plan, create_todo, update_todo, list_todos, get_todo tools.

        Example:
            .plan_mode(read_only_tool_names=["read_file", "list_dir"])
        """
        if "plan_mode" not in self._config:
            self._config["plan_mode"] = None
            self._config["todo_manager"] = None
            self._config["read_only_tool_names"] = None
        if plan_mode is not None:
            self._config["plan_mode"] = plan_mode
        if todo_manager is not None:
            self._config["todo_manager"] = todo_manager
        if read_only_tool_names is not None:
            self._config["read_only_tool_names"] = read_only_tool_names
        return self

    # ── MCP (Model Context Protocol) ────────────────────────────────

    def mcp_server(self, server_url: str) -> AgentBuilder:
        """
        Add an MCP server by URL. Tools are discovered at agent startup.

        Examples:
            .mcp_server("stdio://npx -y @modelcontextprotocol/server-filesystem /path")
            .mcp_server("http://localhost:8080/sse")
        """
        if "mcp_server_urls" not in self._config:
            self._config["mcp_server_urls"] = []
        self._config["mcp_server_urls"].append(server_url)
        return self

    def mcp_server_config(self, config: dict) -> AgentBuilder:
        """
        Add an MCP server from a Cursor/Claude-style config (command, args, env or url, headers).

        Example (stdio with credentials):
            .mcp_server_config({
                "command": "npx",
                "args": ["-y", "@modelcontextprotocol/server-github"],
                "env": {"GITHUB_PERSONAL_ACCESS_TOKEN": "ghp_..."},
            })
        Example (HTTP with auth):
            .mcp_server_config({
                "url": "https://api.example.com/mcp",
                "headers": {"Authorization": "Bearer ..."},
            })
        Env values like "$GITHUB_TOKEN" are resolved from os.environ at connect time.
        """
        if "mcp_server_configs" not in self._config:
            self._config["mcp_server_configs"] = []
        self._config["mcp_server_configs"].append(config)
        return self

    def mcp_servers_from_file(self, path: str | Path) -> AgentBuilder:
        """
        Load MCP servers from a JSON file (Cursor/Claude mcp.json style).

        Expected format: {"mcpServers": {"name": {"command": "...", "args": [...], "env": {...}}}}
        or {"mcpServers": {"name": {"url": "...", "headers": {...}}}}.
        Disabled servers are skipped. Use resolve_env to expand $VAR in env/headers.
        """
        from curio_agent_sdk.mcp.config import load_mcp_servers_from_file
        configs = load_mcp_servers_from_file(path)
        if "mcp_server_configs" not in self._config:
            self._config["mcp_server_configs"] = []
        for cfg in configs:
            self._config["mcp_server_configs"].append(cfg)
        return self

    def mcp_resources_for_context(self, resource_uris: list[str]) -> AgentBuilder:
        """
        MCP resource URIs to fetch and inject as context at the start of each run.
        Requires at least one .mcp_server() to be configured.
        """
        self._config["mcp_resource_uris"] = list(resource_uris)
        return self

    # ── Event bus ────────────────────────────────────────────────────

    def event_bus(self, bus: Any) -> AgentBuilder:
        """
        Set a distributed event bus. All hook events will be auto-published
        to the bus via an EventBusBridge so remote subscribers can observe
        agent execution across processes or machines.

        Example:
            from curio_agent_sdk.core.event_bus import InMemoryEventBus
            bus = InMemoryEventBus()
            agent = Agent.builder().event_bus(bus).model("openai:gpt-4o").build()

            # Subscribe from anywhere
            await bus.subscribe("tool.call.*", my_handler)
        """
        self._config["event_bus"] = bus
        return self

    # ── Connector framework ──────────────────────────────────────────

    def connector(self, connector: Any) -> AgentBuilder:
        """
        Add a connector (external service integration). Tools are registered at agent startup.

        Example:
            .connector(GitHubConnector(token="ghp_..."))
            .connector(SlackConnector(token="xoxb-..."))
        """
        if "connectors" not in self._config:
            self._config["connectors"] = []
        self._config["connectors"].append(connector)
        return self

    def connectors(self, connectors: list[Any]) -> AgentBuilder:
        """Add multiple connectors at once."""
        if "connectors" not in self._config:
            self._config["connectors"] = []
        self._config["connectors"].extend(connectors)
        return self

    # ── Subagent / multi-agent ──────────────────────────────────────

    def subagent(self, name: str, config: "SubagentConfig | dict[str, Any]") -> AgentBuilder:
        """
        Register a subagent config by name. The parent agent can spawn it via spawn_subagent(name, task).

        Example:
            .subagent("researcher", SubagentConfig(
                name="researcher",
                system_prompt="Research specialist.",
                tools=[web_search, fetch_page],
                model="openai:gpt-4o",
            ))
        """
        if "subagent_configs" not in self._config:
            self._config["subagent_configs"] = {}
        self._config["subagent_configs"][name] = config
        return self

    # ── Plugins ──────────────────────────────────────────────────────

    def plugin(self, plugin: "Plugin") -> AgentBuilder:
        """
        Register a single plugin on this builder.

        Plugins are applied during build() via apply_plugins_to_builder(), which
        wires their tools, hooks, middleware, skills, connectors, and instructions
        into the Agent configuration.
        """
        if "plugins" not in self._config:
            self._config["plugins"] = []
        self._config["plugins"].append(plugin)
        return self

    def plugins(self, plugins: list["Plugin"]) -> AgentBuilder:
        """
        Register multiple plugins at once.

        Example:
            .plugins([GitPlugin(), WebSearchPlugin()])
        """
        if "plugins" not in self._config:
            self._config["plugins"] = []
        self._config["plugins"].extend(plugins)
        return self

    def discover_plugins(self, entry_point_group: str = "curio_plugins") -> AgentBuilder:
        """
        Discover and register plugins from installed packages via entry points.

        Packages can expose plugins by defining entry points in their pyproject.toml:

            [project.entry-points."curio_plugins"]
            git = "my_package.plugins:GitPlugin"
        """
        from curio_agent_sdk.core.plugins import discover_plugins

        discovered = discover_plugins(entry_point_group)
        if not discovered:
            return self
        if "plugins" not in self._config:
            self._config["plugins"] = []
        self._config["plugins"].extend(discovered)
        return self

    # ── Build ───────────────────────────────────────────────────────

    def build(self) -> Agent:
        """
        Build and return the configured Agent.

        All unset parameters use Agent's defaults. Any .hook() registrations
        are applied to the hook registry (created or provided). Instructions
        (from .instructions() or .instructions_file()) are merged into system_prompt.
        """
        from curio_agent_sdk.core.agent import Agent
        from curio_agent_sdk.core.hooks import HookRegistry
        from curio_agent_sdk.core.instructions import (
            InstructionLoader,
            load_instructions_from_file,
        )
        from curio_agent_sdk.core.plugins import apply_plugins_to_builder
        # Apply any registered plugins before freezing the config.
        plugins = self._config.get("plugins") or []
        if plugins:
            apply_plugins_to_builder(self, plugins)

        config = dict(self._config)
        registry = config.get("hook_registry") or HookRegistry()
        for event, handler, priority in config.pop("hooks", []):
            registry.on(event, handler, priority=priority)
        config["hook_registry"] = registry

        # Resolve instructions and merge into system_prompt
        resolved_instructions: str = ""
        if "instructions_file" in config:
            path = config.pop("instructions_file")
            resolved_instructions = load_instructions_from_file(path)
        if "instructions" in config:
            source = config.pop("instructions")
            if isinstance(source, InstructionLoader):
                resolved_instructions = source.load()
            elif isinstance(source, str):
                resolved_instructions = source
        if resolved_instructions:
            base = config.get("system_prompt") or "You are a helpful assistant."
            config["system_prompt"] = f"{base}\n\n---\n\n{resolved_instructions}"

        # Pass skills list to Agent (Agent will create SkillRegistry and merge tools/hooks)
        if "skills" in config and not config.get("skill_registry"):
            config["skills"] = config.pop("skills", [])
        else:
            config.pop("skills", None)

        # MCP (optional)
        config.setdefault("mcp_server_urls", None)
        config.setdefault("mcp_server_configs", None)
        config.setdefault("mcp_resource_uris", None)

        # Connectors (optional)
        config.setdefault("connectors", None)

        # Event bus (optional)
        config.setdefault("event_bus", None)

        return Agent(**config)

    # ── Utility ─────────────────────────────────────────────────────

    def clone(self) -> AgentBuilder:
        """Create a copy of this builder with the same configuration."""
        new_builder = AgentBuilder()
        new_builder._config = dict(self._config)
        return new_builder

    def __repr__(self) -> str:
        keys = ", ".join(self._config.keys())
        return f"AgentBuilder(configured=[{keys}])"
