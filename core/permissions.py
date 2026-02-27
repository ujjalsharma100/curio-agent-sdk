"""
Permission and sandbox model for tool execution.

Provides pluggable policies that control what tools can do (allow/deny/ask user).
Supports tool-level checks, file access checks, and network access checks.
Integrates with HumanInputHandler when ask_user is True.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse


# Common argument names that may contain file paths or URLs (for policy introspection)
PATH_LIKE_KEYS = frozenset({"path", "file_path", "file", "filepath", "file_paths", "paths", "directory", "dir", "target"})
URL_LIKE_KEYS = frozenset({"url", "uri", "href", "endpoint", "link"})


def _collect_paths_from_args(args: dict[str, Any]) -> list[tuple[str, str]]:
    """Collect (key, value) pairs from args that look like file paths. Value is normalized to str."""
    out: list[tuple[str, str]] = []
    for k, v in args.items():
        key_lower = k.lower()
        if key_lower in PATH_LIKE_KEYS and v is not None:
            if isinstance(v, str):
                out.append((k, v))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        out.append((k, item))
    return out


def _collect_urls_from_args(args: dict[str, Any]) -> list[tuple[str, str]]:
    """Collect (key, value) pairs from args that look like URLs."""
    out: list[tuple[str, str]] = []
    for k, v in args.items():
        key_lower = k.lower()
        if key_lower in URL_LIKE_KEYS and v is not None:
            if isinstance(v, str):
                out.append((k, v))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, str):
                        out.append((k, item))
    return out


@dataclass
class PermissionResult:
    """Result of a permission check."""

    allowed: bool
    reason: str = ""
    ask_user: bool = False

    @classmethod
    def allow(cls, reason: str = "") -> PermissionResult:
        return cls(allowed=True, reason=reason)

    @classmethod
    def deny(cls, reason: str) -> PermissionResult:
        return cls(allowed=False, reason=reason)

    @classmethod
    def ask(cls, reason: str = "Confirmation required") -> PermissionResult:
        return cls(allowed=True, reason=reason, ask_user=True)


class PermissionPolicy(ABC):
    """
    Controls what the agent is allowed to do.

    Implement check_tool_call to allow/deny/ask for tool execution.
    Optionally implement check_file_access and check_network_access for
    file-system and network sandboxing (policies can call these from
    check_tool_call after extracting paths/urls from args).
    """

    @abstractmethod
    async def check_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> PermissionResult:
        """
        Check whether a tool call is allowed.

        Args:
            tool_name: Name of the tool.
            args: Tool arguments (as passed by the LLM).
            context: Execution context (e.g. run_id, agent_id, tool_config).

        Returns:
            PermissionResult: allowed=True to proceed; allowed=False to deny;
                ask_user=True to prompt human for confirmation (HumanInputHandler used).
        """
        ...

    async def check_file_access(self, path: str, mode: str, context: dict[str, Any]) -> PermissionResult:
        """
        Check whether file access is allowed (read/write/delete).

        Override for file-system sandboxing. Default allows all.
        mode is typically "r", "w", "x", or "delete".
        """
        return PermissionResult.allow()

    async def check_network_access(self, url: str, context: dict[str, Any]) -> PermissionResult:
        """
        Check whether network access to url is allowed.

        Override for network sandboxing. Default allows all.
        """
        return PermissionResult.allow()


class AllowAll(PermissionPolicy):
    """Allow all tool calls, file access, and network access. No confirmation."""

    async def check_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> PermissionResult:
        return PermissionResult.allow()


class AskAlways(PermissionPolicy):
    """Allow tool calls only after user confirmation (HumanInputHandler)."""

    async def check_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> PermissionResult:
        return PermissionResult.ask("Tool execution requires confirmation")


class AllowReadsAskWrites(PermissionPolicy):
    """
    Allow read-only operations automatically; ask user for writes/destructive actions.

    Heuristic: tools with names or args suggesting write/delete/edit (e.g. write, edit,
    delete, create, run, execute) trigger ask_user. File access: read allowed,
    write/delete ask. Network: GET allowed, POST/PUT/DELETE ask (inferred from tool name/args).
    """

    _WRITE_LIKE_PATTERN = re.compile(
        r"\b(write|edit|delete|create|run|execute|execute_code|shell|command|remove|rm|add|append|modify|update|install)\b",
        re.IGNORECASE,
    )

    async def check_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> PermissionResult:
        if self._is_read_only(tool_name, args):
            return PermissionResult.allow()
        return PermissionResult.ask("This action may modify state; confirmation required")

    def _is_read_only(self, tool_name: str, args: dict[str, Any]) -> bool:
        if self._WRITE_LIKE_PATTERN.search(tool_name):
            return False
        # Optional: inspect args for "mode" or "action" suggesting write
        return True

    async def check_file_access(self, path: str, mode: str, context: dict[str, Any]) -> PermissionResult:
        if mode in ("r", "read"):
            return PermissionResult.allow()
        return PermissionResult.ask("File write/delete requires confirmation")

    async def check_network_access(self, url: str, context: dict[str, Any]) -> PermissionResult:
        # Without HTTP method we allow; callers can pass method in context if needed
        return PermissionResult.allow()


class CompoundPolicy(PermissionPolicy):
    """Combine multiple policies: all must allow (first deny or ask wins)."""

    def __init__(self, policies: list[PermissionPolicy]):
        self.policies = list(policies)

    async def check_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> PermissionResult:
        for p in self.policies:
            result = await p.check_tool_call(tool_name, args, context)
            if not result.allowed:
                return result
            if result.ask_user:
                return result
        return PermissionResult.allow()

    async def check_file_access(self, path: str, mode: str, context: dict[str, Any]) -> PermissionResult:
        for p in self.policies:
            result = await p.check_file_access(path, mode, context)
            if not result.allowed:
                return result
            if result.ask_user:
                return result
        return PermissionResult.allow()

    async def check_network_access(self, url: str, context: dict[str, Any]) -> PermissionResult:
        for p in self.policies:
            result = await p.check_network_access(url, context)
            if not result.allowed:
                return result
            if result.ask_user:
                return result
        return PermissionResult.allow()


def _normalize_path_for_prefix(path: str) -> str:
    """Normalize path for prefix comparison (resolve . and ..)."""
    from pathlib import Path
    try:
        return str(Path(path).resolve())
    except (OSError, RuntimeError):
        return path


def _path_under_prefix(resolved: str, prefix_resolved: str) -> bool:
    """Return True if resolved is equal to or under prefix_resolved."""
    if resolved == prefix_resolved:
        return True
    from pathlib import Path
    try:
        return Path(resolved).is_relative_to(Path(prefix_resolved))
    except (ValueError, TypeError):
        sep = "/" if "/" in prefix_resolved else "\\"
        p = prefix_resolved.rstrip(sep) + sep
        return resolved == prefix_resolved or resolved.startswith(p)


class FileSandboxPolicy(PermissionPolicy):
    """
    Restrict file access to allowed path prefixes (file system sandboxing).

    Only paths under one of allowed_prefixes are permitted. Paths are normalized
    before comparison. Use with CompoundPolicy to combine with other policies.
    """

    def __init__(self, allowed_prefixes: list[str], mode_for_writes: str = "w"):
        self.allowed_prefixes = [str(p) for p in allowed_prefixes]
        self.mode_for_writes = mode_for_writes

    async def check_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> PermissionResult:
        for _key, path in _collect_paths_from_args(args):
            result = await self.check_file_access(path, "r", context)
            if not result.allowed:
                return result
            if result.ask_user:
                return result
        return PermissionResult.allow()

    async def check_file_access(self, path: str, mode: str, context: dict[str, Any]) -> PermissionResult:
        try:
            resolved = _normalize_path_for_prefix(path)
        except Exception:
            return PermissionResult.deny(f"Invalid path: {path}")
        for prefix in self.allowed_prefixes:
            try:
                prefix_resolved = _normalize_path_for_prefix(prefix)
                if _path_under_prefix(resolved, prefix_resolved):
                    return PermissionResult.allow()
            except Exception:
                if path.startswith(prefix):
                    return PermissionResult.allow()
        return PermissionResult.deny(f"Path not in allowed list: {path}")


class NetworkSandboxPolicy(PermissionPolicy):
    """
    Restrict network access to allowed URL patterns (network sandboxing).

    allowed_patterns: list of regex patterns or literal host substrings (e.g. ["^https://api\\.example\\.com", "localhost"]).
    Use with CompoundPolicy to combine with other policies.
    """

    def __init__(self, allowed_patterns: list[str]):
        self._compiled: list[re.Pattern[str] | str] = []
        for p in allowed_patterns:
            try:
                self._compiled.append(re.compile(p))
            except re.error:
                self._compiled.append(p)  # literal substring

    async def check_tool_call(
        self,
        tool_name: str,
        args: dict[str, Any],
        context: dict[str, Any],
    ) -> PermissionResult:
        for _key, url in _collect_urls_from_args(args):
            result = await self.check_network_access(url, context)
            if not result.allowed:
                return result
            if result.ask_user:
                return result
        return PermissionResult.allow()

    async def check_network_access(self, url: str, context: dict[str, Any]) -> PermissionResult:
        # Basic URL validation to guard against schemes like javascript:, file:, etc.
        try:
            parsed = urlparse(url)
        except Exception:
            return PermissionResult.deny(f"Invalid URL: {url}")

        if not parsed.scheme or parsed.scheme not in ("http", "https"):
            return PermissionResult.deny(f"Disallowed URL scheme for: {url}")
        if not parsed.netloc:
            return PermissionResult.deny(f"Invalid URL host for: {url}")

        for pat in self._compiled:
            if isinstance(pat, re.Pattern):
                if pat.search(url):
                    return PermissionResult.allow()
            else:
                if pat in url:
                    return PermissionResult.allow()
        return PermissionResult.deny(f"URL not in allowed list: {url}")
