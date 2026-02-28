from __future__ import annotations

"""
CLI framework for interactive agents.

Provides an `AgentCLI` harness that wraps an `Agent` with:
- an async REPL loop (`run_interactive`)
- a one-shot, pipe-friendly entrypoint (`run_once`)
- a simple slash-command system (`/help`, `/clear`, `/status`, `/exit`, etc.)
- optional session persistence via `SessionManager`
- basic streaming output rendering using `Agent.astream()`

This is intentionally lightweight and stdlib-only so it can be used in
any environment without additional dependencies.
"""

import asyncio
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Dict, Optional

from curio_agent_sdk.core.agent import Agent
from curio_agent_sdk.core.state import (
    SessionManager,
    InMemorySessionStore,
)
from curio_agent_sdk.models.agent import AgentRunResult
from curio_agent_sdk.models.events import StreamEvent


CommandHandler = Callable[[str], Awaitable[None]] | Callable[[str], None]
KeybindingHandler = Callable[[], Awaitable[None]] | Callable[[], None]


@dataclass
class _CLICommand:
    name: str
    handler: CommandHandler
    description: str = ""


class AgentCLI:
    """
    CLI harness for interactive agents.

    Wraps an `Agent` and provides:
    - `run_interactive()` for a streaming REPL
    - `run_once()` for single-shot, script/pipe mode
    - a simple slash-command system and keybinding hooks

    Session persistence:
        If a `SessionManager` is provided (or attached to the agent),
        interactive runs can create and reuse conversation sessions.
        When `use_sessions=True` and streaming is disabled, runs will
        use `agent.arun(..., session_id=...)` so that history is
        persisted through the existing session system.
    """

    def __init__(
        self,
        agent: Agent,
        session_manager: SessionManager | None = None,
    ) -> None:
        self.agent = agent

        # Resolve session manager: prefer explicit, then agent-attached, then in-memory.
        mgr = session_manager or getattr(agent, "session_manager", None)
        if mgr is None:
            mgr = SessionManager(InMemorySessionStore())
        self.session_manager: SessionManager = mgr
        # Ensure the agent also sees this manager.
        self.agent.session_manager = self.session_manager

        self._commands: Dict[str, _CLICommand] = {}
        self._keybindings: Dict[str, KeybindingHandler] = {}
        self.current_session_id: Optional[str] = None
        self._should_exit: bool = False

        self._register_builtin_commands()

    # ── Public API ────────────────────────────────────────────────

    async def run_interactive(
        self,
        *,
        stream: bool = True,
        use_sessions: bool = True,
        prompt: str = ">>> ",
    ) -> None:
        """
        Run the agent in interactive REPL mode.

        Args:
            stream: If True, use streaming output (`agent.astream`).
                    If False, use `agent.arun` and print the final result.
            use_sessions: If True, maintain a conversation session using
                          the configured `SessionManager`. Session
                          persistence is applied when `stream=False`,
                          so history is preserved across calls.
            prompt: REPL prompt string.
        """
        banner = (
            f"Curio Agent CLI — {self.agent.agent_name} ({self.agent.agent_id})\n"
            "Type messages to talk to the agent.\n"
            "Use /help for commands, /exit to quit.\n"
        )
        print(banner)

        async with self.agent:
            while not self._should_exit:
                try:
                    line = await self._readline(prompt)
                except (EOFError, KeyboardInterrupt):
                    print("\nExiting.")
                    break

                if line is None:
                    continue
                line = line.strip()
                if not line:
                    continue

                # Keybindings: match exact key string (simple, no raw mode).
                if line in self._keybindings:
                    await self._invoke_keybinding(line)
                    continue

                # Slash commands: /help, /exit, etc.
                if line.startswith("/"):
                    await self._run_command(line)
                    continue

                # Regular user message.
                await self._handle_user_message(
                    line,
                    stream=stream,
                    use_sessions=use_sessions,
                )

    async def run_once(
        self,
        input_text: str,
        *,
        stream: bool = False,
        session_id: str | None = None,
    ) -> int:
        """
        Run a single command (pipe/script mode).

        Args:
            input_text: The user's input.
            stream: If True, stream output to stdout as it arrives.
            session_id: Optional existing session ID to reuse.

        Returns:
            Exit code (0 for success, non-zero for error/timeout).
        """
        async with self.agent:
            if stream:
                await self._stream_run(input_text)
                return 0

            result = await self.agent.arun(
                input_text,
                session_id=session_id,
            )
            self._print_result(result)
            return 0 if result.status == "completed" else 1

    # ── Command / keybinding registration ─────────────────────────

    def register_command(
        self,
        name: str,
        handler: CommandHandler,
        description: str = "",
    ) -> None:
        """
        Register a slash command.

        Example:
            cli.register_command("ping", lambda args: print("pong"))
            # Invoked as /ping
        """
        if not name:
            raise ValueError("Command name must be non-empty")
        if not name.startswith("/"):
            name = "/" + name
        self._commands[name] = _CLICommand(name=name, handler=handler, description=description)

    def register_keybinding(
        self,
        key: str,
        handler: KeybindingHandler,
    ) -> None:
        """
        Register a simple keybinding.

        This implementation matches the *entire* input line to `key`
        (e.g. typing `:r` could be bound to a "rerun" action). It does
        not put the terminal into raw mode; that can be layered on top
        by more advanced frontends.
        """
        if not key:
            raise ValueError("Keybinding key must be non-empty")
        self._keybindings[key] = handler

    # ── Internal helpers ──────────────────────────────────────────

    async def _readline(self, prompt: str) -> Optional[str]:
        """Read a line from stdin without blocking the event loop."""
        return await asyncio.to_thread(input, prompt)

    async def _run_command(self, line: str) -> None:
        """Parse and execute a slash command line."""
        parts = line.split(maxsplit=1)
        cmd_name = parts[0]
        args = parts[1] if len(parts) > 1 else ""

        cmd = self._commands.get(cmd_name)
        if cmd is None:
            print(f"Unknown command: {cmd_name}. Type /help for a list of commands.")
            return

        handler = cmd.handler
        if asyncio.iscoroutinefunction(handler):  # type: ignore[arg-type]
            await handler(args)  # type: ignore[misc]
        else:
            handler(args)  # type: ignore[misc]

    async def _invoke_keybinding(self, key: str) -> None:
        handler = self._keybindings.get(key)
        if handler is None:
            return
        if asyncio.iscoroutinefunction(handler):  # type: ignore[arg-type]
            await handler()  # type: ignore[misc]
        else:
            handler()  # type: ignore[misc]

    async def _handle_user_message(
        self,
        text: str,
        *,
        stream: bool,
        use_sessions: bool,
    ) -> None:
        """Send a user message to the agent, optionally with session support."""
        session_id: str | None = None
        if use_sessions and self.session_manager is not None:
            if self.current_session_id is None:
                session = await self.session_manager.create(self.agent.agent_id)
                self.current_session_id = session.id
                print(f"[Created new session: {session.id}]")
            session_id = self.current_session_id

        if stream:
            await self._stream_run(text)
        else:
            result = await self.agent.arun(
                text,
                session_id=session_id,
            )
            self._print_result(result)

    async def _stream_run(self, text: str) -> None:
        """Stream a single run to stdout using Agent.astream()."""
        print(f"\nYou: {text}\n")
        async for event in self.agent.astream(text):
            self._render_stream_event(event)
        print()  # Final newline after [Done]

    # ── Rendering helpers ─────────────────────────────────────────

    def _render_stream_event(self, event: StreamEvent) -> None:
        """Render a StreamEvent to stdout."""
        if event.type == "text_delta" and event.text:
            print(event.text, end="", flush=True)
        elif event.type == "tool_call_start":
            tool = event.tool_name or "unknown_tool"
            print(f"\n[Calling tool: {tool}]", flush=True)
        elif event.type == "tool_call_end":
            tool = event.tool_name or "unknown_tool"
            print(f"\n[Tool finished: {tool}]", flush=True)
        elif event.type == "thinking":
            print("\n[Thinking...]", flush=True)
        elif event.type == "iteration_start":
            if event.iteration:
                print(f"\n[Iteration {event.iteration}]", flush=True)
        elif event.type == "error":
            msg = event.error or str(event.data)
            print(f"\n[Error] {msg}", flush=True)
        elif event.type == "done":
            print("\n[Done]", flush=True)

    def _print_result(self, result: AgentRunResult) -> None:
        """Print a completed AgentRunResult."""
        status = result.status
        print(f"\n[{status.upper()}]")
        if result.output:
            print(result.output)
        else:
            print("(no output)")
        print(
            f"\nIterations: {result.total_iterations}  "
            f"LLM calls: {result.total_llm_calls}  "
            f"Tool calls: {result.total_tool_calls}"
        )

    # ── Built-in commands ─────────────────────────────────────────

    def _register_builtin_commands(self) -> None:
        self.register_command("help", self._cmd_help, "Show this help message.")
        self.register_command("clear", self._cmd_clear, "Clear the terminal screen.")
        self.register_command("status", self._cmd_status, "Show agent and session status.")
        self.register_command("exit", self._cmd_exit, "Exit the CLI.")
        self.register_command("sessions", self._cmd_sessions, "List known sessions (if enabled).")
        self.register_command("session", self._cmd_session, "Manage current session: /session [new|<id>].")
        self.register_command("skills", self._cmd_skills, "List registered skills on this agent.")

    def _cmd_help(self, _: str) -> None:
        print("\nAvailable commands:")
        for name in sorted(self._commands.keys()):
            cmd = self._commands[name]
            desc = f" — {cmd.description}" if cmd.description else ""
            print(f"  {name}{desc}")
        print("\nType a message to send it to the agent.")

    def _cmd_clear(self, _: str) -> None:
        # Best-effort clear; fall back to printing newlines.
        if os.name == "nt":
            os.system("cls")
        else:
            os.system("clear || tput reset || printf '\\033c'")

    def _cmd_status(self, _: str) -> None:
        print(f"\nAgent: {self.agent.agent_name} ({self.agent.agent_id})")
        if self.current_session_id:
            print(f"Current session: {self.current_session_id}")
        else:
            print("Current session: (none)")
        if getattr(self.agent, "plan_mode", None) is not None:
            try:
                in_plan = self.agent.is_in_plan_mode()
                awaiting = self.agent.is_awaiting_plan_approval()
                print(f"Plan mode: {'active' if in_plan else 'inactive'}; "
                      f"awaiting approval: {'yes' if awaiting else 'no'}")
            except Exception:
                pass

    def _cmd_exit(self, _: str) -> None:
        print("Exiting CLI.")
        self._should_exit = True

    async def _cmd_sessions(self, _: str) -> None:
        if self.session_manager is None:
            print("Session management is not enabled.")
            return
        sessions = await self.session_manager.list(self.agent.agent_id, limit=50)
        if not sessions:
            print("No sessions found.")
            return
        print("\nSessions:")
        for s in sessions:
            marker = "*" if s.id == self.current_session_id else " "
            print(f"{marker} {s.id}  updated_at={s.updated_at.isoformat()}  metadata={s.metadata}")

    async def _cmd_session(self, args: str) -> None:
        if self.session_manager is None:
            print("Session management is not enabled.")
            return
        arg = args.strip()
        if not arg:
            if self.current_session_id:
                print(f"Current session: {self.current_session_id}")
            else:
                print("No current session. Use `/session new` to create one.")
            return
        if arg == "new":
            session = await self.session_manager.create(self.agent.agent_id)
            self.current_session_id = session.id
            print(f"Created new session: {session.id}")
            return
        # Treat arg as session id
        session = await self.session_manager.get(arg)
        if session is None:
            print(f"No such session: {arg}")
            return
        self.current_session_id = session.id
        print(f"Switched to session: {session.id}")

    def _cmd_skills(self, _: str) -> None:
        registry = getattr(self.agent, "skill_registry", None)
        if registry is None:
            print("No skills registry configured on this agent.")
            return
        names = registry.list_names()
        if not names:
            print("No skills registered.")
            return
        print("\nSkills:")
        for name in names:
            skill = registry.get(name)
            desc = skill.description if skill and skill.description else ""
            suffix = f" — {desc}" if desc else ""
            print(f"  {name}{suffix}")


async def main_async() -> None:
    """
    Minimal entrypoint for `curio-agent` style CLIs.

    This builds a simple agent from environment/config and starts
    an interactive CLI using the default model.
    """
    import argparse
    from curio_agent_sdk.core.security import CLIHumanInput

    parser = argparse.ArgumentParser(description="Curio Agent CLI")
    parser.add_argument(
        "--model",
        default="openai:gpt-4o",
        help="Model identifier (provider:model). Default: %(default)s",
    )
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Disable streaming; print only final answers.",
    )
    args = parser.parse_args()

    agent = Agent(
        model=args.model,
        tools=[],
        human_input=CLIHumanInput(),
        system_prompt="You are a helpful assistant for the terminal.",
    )
    cli = AgentCLI(agent)
    await cli.run_interactive(stream=not args.no_stream, use_sessions=True)


def main() -> None:
    """Sync wrapper for CLI entrypoints."""
    asyncio.run(main_async())

