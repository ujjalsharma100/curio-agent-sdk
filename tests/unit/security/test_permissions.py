"""
Unit tests for PermissionResult, PermissionPolicy implementations, and sandbox policies.
"""

import pytest

from curio_agent_sdk.core.security.permissions import (
    PermissionResult,
    PermissionPolicy,
    AllowAll,
    AskAlways,
    AllowReadsAskWrites,
    CompoundPolicy,
    FileSandboxPolicy,
    NetworkSandboxPolicy,
    _collect_paths_from_args,
    _collect_urls_from_args,
)


@pytest.mark.unit
class TestPermissionResult:
    def test_permission_result_allow(self):
        r = PermissionResult.allow()
        assert r.allowed is True
        assert r.ask_user is False
        assert r.reason == ""

    def test_permission_result_allow_with_reason(self):
        r = PermissionResult.allow("read-only")
        assert r.allowed is True
        assert r.reason == "read-only"

    def test_permission_result_deny(self):
        r = PermissionResult.deny("not allowed")
        assert r.allowed is False
        assert r.reason == "not allowed"
        assert r.ask_user is False

    def test_permission_result_ask(self):
        r = PermissionResult.ask("Confirmation required")
        assert r.allowed is True
        assert r.ask_user is True
        assert r.reason == "Confirmation required"

    def test_permission_result_ask_default_reason(self):
        r = PermissionResult.ask()
        assert r.allowed is True
        assert r.ask_user is True
        assert r.reason == "Confirmation required"


@pytest.mark.unit
class TestAllowAll:
    @pytest.mark.asyncio
    async def test_allow_all_policy(self):
        policy = AllowAll()
        result = await policy.check_tool_call("any_tool", {"x": 1}, {})
        assert result.allowed is True
        assert result.ask_user is False

    @pytest.mark.asyncio
    async def test_allow_all_check_file_access(self):
        policy = AllowAll()
        result = await policy.check_file_access("/any/path", "w", {})
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_allow_all_check_network_access(self):
        policy = AllowAll()
        result = await policy.check_network_access("https://example.com", {})
        assert result.allowed is True


@pytest.mark.unit
class TestAskAlways:
    @pytest.mark.asyncio
    async def test_ask_always_policy(self):
        policy = AskAlways()
        result = await policy.check_tool_call("any_tool", {}, {})
        assert result.allowed is True
        assert result.ask_user is True
        assert "confirmation" in result.reason.lower()


@pytest.mark.unit
class TestAllowReadsAskWrites:
    @pytest.mark.asyncio
    async def test_allow_reads_ask_writes_read_tool(self):
        policy = AllowReadsAskWrites()
        result = await policy.check_tool_call("read_file", {"path": "/tmp/x"}, {})
        assert result.allowed is True
        assert result.ask_user is False

    @pytest.mark.asyncio
    async def test_allow_reads_ask_writes_write_tool(self):
        policy = AllowReadsAskWrites()
        # Policy matches whole words: "write", "execute", "run", etc.
        result = await policy.check_tool_call("write", {"path": "/tmp/x"}, {})
        assert result.allowed is True
        assert result.ask_user is True
        assert "modify" in result.reason.lower() or "confirmation" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_allow_reads_ask_writes_execute_tool(self):
        policy = AllowReadsAskWrites()
        result = await policy.check_tool_call("execute_code", {"code": "1+1"}, {})
        assert result.ask_user is True

    @pytest.mark.asyncio
    async def test_allow_reads_ask_writes_file_access_read(self):
        policy = AllowReadsAskWrites()
        result = await policy.check_file_access("/tmp/x", "r", {})
        assert result.allowed is True
        assert result.ask_user is False

    @pytest.mark.asyncio
    async def test_allow_reads_ask_writes_file_access_write(self):
        policy = AllowReadsAskWrites()
        result = await policy.check_file_access("/tmp/x", "w", {})
        assert result.allowed is True
        assert result.ask_user is True


@pytest.mark.unit
class TestCompoundPolicy:
    @pytest.mark.asyncio
    async def test_compound_policy_all_allow(self):
        policy = CompoundPolicy([AllowAll(), AllowAll()])
        result = await policy.check_tool_call("tool", {}, {})
        assert result.allowed is True
        assert result.ask_user is False

    @pytest.mark.asyncio
    async def test_compound_policy_one_deny(self):
        class DenyAll(PermissionPolicy):
            async def check_tool_call(self, tool_name, args, context):
                return PermissionResult.deny("denied")

        policy = CompoundPolicy([AllowAll(), DenyAll()])
        result = await policy.check_tool_call("tool", {}, {})
        assert result.allowed is False
        assert result.reason == "denied"

    @pytest.mark.asyncio
    async def test_compound_policy_first_ask_wins(self):
        policy = CompoundPolicy([AskAlways(), AllowAll()])
        result = await policy.check_tool_call("tool", {}, {})
        assert result.allowed is True
        assert result.ask_user is True

    @pytest.mark.asyncio
    async def test_compound_policy_empty_list(self):
        policy = CompoundPolicy([])
        result = await policy.check_tool_call("tool", {}, {})
        assert result.allowed is True


@pytest.mark.unit
class TestFileSandboxPolicy:
    @pytest.mark.asyncio
    async def test_file_sandbox_policy_allowed(self, tmp_path):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        policy = FileSandboxPolicy(allowed_prefixes=[str(allowed_dir)])
        target = allowed_dir / "file.txt"
        result = await policy.check_file_access(str(target), "r", {})
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_file_sandbox_policy_denied(self, tmp_path):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        policy = FileSandboxPolicy(allowed_prefixes=[str(allowed_dir)])
        result = await policy.check_file_access("/etc/passwd", "r", {})
        assert result.allowed is False
        assert "not in allowed" in result.reason or "allowed" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_file_sandbox_path_traversal(self, tmp_path):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        (allowed_dir / "sub").mkdir(parents=True)
        policy = FileSandboxPolicy(allowed_prefixes=[str(allowed_dir)])
        # Path that escapes via ..
        traversal = str(allowed_dir / "sub" / ".." / ".." / "etc" / "passwd")
        result = await policy.check_file_access(traversal, "r", {})
        assert result.allowed is False

    @pytest.mark.asyncio
    async def test_file_sandbox_check_file_access_subpath(self, tmp_path):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        (allowed_dir / "nested").mkdir()
        policy = FileSandboxPolicy(allowed_prefixes=[str(allowed_dir)])
        result = await policy.check_file_access(str(allowed_dir / "nested" / "file.txt"), "r", {})
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_file_sandbox_check_tool_call_with_path_in_args(self, tmp_path):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        policy = FileSandboxPolicy(allowed_prefixes=[str(allowed_dir)])
        result = await policy.check_tool_call(
            "read_file", {"path": str(allowed_dir / "f.txt")}, {}
        )
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_file_sandbox_check_tool_call_denied_path_in_args(self, tmp_path):
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        policy = FileSandboxPolicy(allowed_prefixes=[str(allowed_dir)])
        result = await policy.check_tool_call("read_file", {"path": "/etc/passwd"}, {})
        assert result.allowed is False


@pytest.mark.unit
class TestNetworkSandboxPolicy:
    @pytest.mark.asyncio
    async def test_network_sandbox_allowed(self):
        policy = NetworkSandboxPolicy(allowed_patterns=["https://api.example.com", "localhost"])
        result = await policy.check_network_access("https://api.example.com/foo", {})
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_network_sandbox_denied(self):
        policy = NetworkSandboxPolicy(allowed_patterns=["https://api.example.com"])
        result = await policy.check_network_access("https://evil.com/bar", {})
        assert result.allowed is False
        assert "not in allowed" in result.reason or "allowed" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_network_sandbox_literal_substring(self):
        policy = NetworkSandboxPolicy(allowed_patterns=["localhost"])
        result = await policy.check_network_access("http://localhost:8080/api", {})
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_network_sandbox_regex_pattern(self):
        policy = NetworkSandboxPolicy(allowed_patterns=[r"^https://api\.example\.com"])
        result = await policy.check_network_access("https://api.example.com/v1", {})
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_network_sandbox_disallowed_scheme(self):
        policy = NetworkSandboxPolicy(allowed_patterns=[".*"])
        result = await policy.check_network_access("javascript:alert(1)", {})
        assert result.allowed is False
        assert "scheme" in result.reason.lower() or "disallowed" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_network_sandbox_check_tool_call_with_url_in_args(self):
        policy = NetworkSandboxPolicy(allowed_patterns=["https://safe.com"])
        result = await policy.check_tool_call(
            "fetch", {"url": "https://safe.com/page"}, {}
        )
        assert result.allowed is True

    @pytest.mark.asyncio
    async def test_network_sandbox_check_tool_call_denied_url_in_args(self):
        policy = NetworkSandboxPolicy(allowed_patterns=["https://safe.com"])
        result = await policy.check_tool_call(
            "fetch", {"url": "https://evil.com/page"}, {}
        )
        assert result.allowed is False


@pytest.mark.unit
class TestCollectPathsAndUrls:
    def test_collect_paths_from_args_string(self):
        out = _collect_paths_from_args({"path": "/tmp/foo"})
        assert out == [("path", "/tmp/foo")]

    def test_collect_paths_from_args_list(self):
        out = _collect_paths_from_args({"file_paths": ["/a", "/b"]})
        assert out == [("file_paths", "/a"), ("file_paths", "/b")]

    def test_collect_paths_from_args_ignores_non_path_keys(self):
        out = _collect_paths_from_args({"other": "/tmp/foo", "path": "/x"})
        assert ("path", "/x") in out
        assert ("other", "/tmp/foo") not in out

    def test_collect_urls_from_args_string(self):
        out = _collect_urls_from_args({"url": "https://example.com"})
        assert out == [("url", "https://example.com")]

    def test_collect_urls_from_args_list(self):
        out = _collect_urls_from_args({"url": ["https://a.com", "https://b.com"]})
        assert out == [("url", "https://a.com"), ("url", "https://b.com")]
