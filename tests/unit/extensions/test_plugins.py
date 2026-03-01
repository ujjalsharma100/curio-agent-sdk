"""
Unit tests for the Plugin system — Plugin base class, discovery,
conflict detection, dependency resolution, and apply_plugins_to_builder.
"""

from unittest.mock import MagicMock, patch, call
from dataclasses import dataclass

import pytest

from curio_agent_sdk.core.extensions.plugins import (
    Plugin,
    apply_plugins_to_builder,
    discover_plugins,
    _toposort_plugins,
    _detect_conflicts,
)


# ---------------------------------------------------------------------------
# Concrete Plugin subclass for testing (Plugin is abstract)
# ---------------------------------------------------------------------------

@dataclass
class DummyPlugin(Plugin):
    """Minimal concrete Plugin for testing."""

    def register(self, builder) -> None:
        builder.tools([])


@dataclass
class ConfiguredPlugin(Plugin):
    """Plugin with config schema validation."""

    def register(self, builder) -> None:
        builder.tools([])

    def get_config_schema(self):
        return {
            "type": "object",
            "required": ["api_key"],
            "properties": {
                "api_key": {"type": "string"},
            },
        }


@dataclass
class LifecyclePlugin(Plugin):
    """Plugin that records lifecycle calls."""

    calls: list = None

    def __post_init__(self):
        if self.calls is None:
            object.__setattr__(self, "calls", [])
        super().__post_init__()

    def on_install(self, builder):
        self.calls.append("on_install")

    def register(self, builder):
        self.calls.append("register")

    def on_enable(self, builder):
        self.calls.append("on_enable")

    def on_disable(self, builder):
        self.calls.append("on_disable")


# ---------------------------------------------------------------------------
# 13.3  Plugin is abstract
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPluginAbstract:
    def test_cannot_instantiate_base_class(self):
        """Plugin is abstract — direct instantiation should fail."""
        with pytest.raises(TypeError):
            Plugin(name="bad", version="1.0")

    def test_concrete_subclass_works(self):
        p = DummyPlugin(name="dummy", version="0.1.0")
        assert p.name == "dummy"
        assert p.version == "0.1.0"


# ---------------------------------------------------------------------------
# 13.3  Plugin metadata
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPluginMetadata:
    def test_name_and_version(self):
        p = DummyPlugin(name="git", version="2.0.0")
        assert p.name == "git"
        assert p.version == "2.0.0"

    def test_default_dependencies(self):
        p = DummyPlugin(name="x", version="1.0")
        assert p.dependencies == []

    def test_custom_dependencies(self):
        p = DummyPlugin(name="x", version="1.0", dependencies=["auth", "db"])
        assert p.dependencies == ["auth", "db"]

    def test_provides_tools_default(self):
        p = DummyPlugin(name="x", version="1.0")
        assert p.provides_tools == []

    def test_provides_hooks_default(self):
        p = DummyPlugin(name="x", version="1.0")
        assert p.provides_hooks == []

    def test_config_default_empty(self):
        p = DummyPlugin(name="x", version="1.0")
        assert p.config == {}


# ---------------------------------------------------------------------------
# 13.3  Plugin.register()
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPluginRegister:
    def test_register_called_with_builder(self):
        p = DummyPlugin(name="test", version="1.0")
        builder = MagicMock()
        p.register(builder)
        builder.tools.assert_called_once_with([])


# ---------------------------------------------------------------------------
# 13.3  Plugin config validation
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPluginConfigValidation:
    def test_valid_config(self):
        p = ConfiguredPlugin(
            name="api", version="1.0", config={"api_key": "secret"}
        )
        assert p.config["api_key"] == "secret"

    def test_missing_required_config_raises(self):
        with pytest.raises(ValueError, match="Missing required config keys"):
            ConfiguredPlugin(name="api", version="1.0", config={})

    def test_no_schema_skips_validation(self):
        # DummyPlugin has no schema => no validation
        p = DummyPlugin(name="x", version="1.0", config={"anything": True})
        assert p.config["anything"] is True


# ---------------------------------------------------------------------------
# 13.3  Plugin lifecycle hooks
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestPluginLifecycle:
    def test_lifecycle_order(self):
        p = LifecyclePlugin(name="lc", version="1.0")
        builder = MagicMock()

        p.on_install(builder)
        p.register(builder)
        p.on_enable(builder)

        assert p.calls == ["on_install", "register", "on_enable"]

    def test_on_disable(self):
        p = LifecyclePlugin(name="lc", version="1.0")
        builder = MagicMock()
        p.on_disable(builder)
        assert "on_disable" in p.calls


# ---------------------------------------------------------------------------
# 13.3  _toposort_plugins
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestToposort:
    def test_no_dependencies(self):
        a = DummyPlugin(name="a", version="1.0")
        b = DummyPlugin(name="b", version="1.0")
        result = _toposort_plugins([a, b])
        assert len(result) == 2

    def test_simple_dependency(self):
        a = DummyPlugin(name="a", version="1.0")
        b = DummyPlugin(name="b", version="1.0", dependencies=["a"])
        result = _toposort_plugins([b, a])
        names = [p.name for p in result]
        assert names.index("a") < names.index("b")

    def test_chain_dependency(self):
        a = DummyPlugin(name="a", version="1.0")
        b = DummyPlugin(name="b", version="1.0", dependencies=["a"])
        c = DummyPlugin(name="c", version="1.0", dependencies=["b"])
        result = _toposort_plugins([c, b, a])
        names = [p.name for p in result]
        assert names.index("a") < names.index("b") < names.index("c")

    def test_unknown_dependency_logged(self):
        a = DummyPlugin(name="a", version="1.0", dependencies=["unknown"])
        result = _toposort_plugins([a])
        assert len(result) == 1
        assert result[0].name == "a"

    def test_cycle_handled(self):
        a = DummyPlugin(name="a", version="1.0", dependencies=["b"])
        b = DummyPlugin(name="b", version="1.0", dependencies=["a"])
        result = _toposort_plugins([a, b])
        assert len(result) == 2

    def test_duplicate_name_keeps_last(self):
        a1 = DummyPlugin(name="a", version="1.0")
        a2 = DummyPlugin(name="a", version="2.0")
        result = _toposort_plugins([a1, a2])
        assert len(result) == 1
        assert result[0].version == "2.0"


# ---------------------------------------------------------------------------
# 13.3  _detect_conflicts
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDetectConflicts:
    def test_no_conflicts(self):
        a = DummyPlugin(name="a", version="1.0", provides_tools=["tool_a"])
        b = DummyPlugin(name="b", version="1.0", provides_tools=["tool_b"])
        conflicts = _detect_conflicts([a, b])
        assert conflicts == {}

    def test_tool_conflict(self):
        a = DummyPlugin(name="a", version="1.0", provides_tools=["read_file"])
        b = DummyPlugin(name="b", version="1.0", provides_tools=["read_file"])
        conflicts = _detect_conflicts([a, b])
        assert "read_file" in conflicts
        assert set(conflicts["read_file"]) == {"a", "b"}

    def test_no_provides(self):
        a = DummyPlugin(name="a", version="1.0")
        b = DummyPlugin(name="b", version="1.0")
        conflicts = _detect_conflicts([a, b])
        assert conflicts == {}


# ---------------------------------------------------------------------------
# 13.3  apply_plugins_to_builder
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestApplyPluginsToBuilder:
    def test_applies_in_dependency_order(self):
        calls = []

        @dataclass
        class TrackingPlugin(Plugin):
            def register(self, builder):
                calls.append(self.name)

        a = TrackingPlugin(name="a", version="1.0")
        b = TrackingPlugin(name="b", version="1.0", dependencies=["a"])

        builder = MagicMock()
        apply_plugins_to_builder(builder, [b, a])

        assert calls.index("a") < calls.index("b")

    def test_lifecycle_hooks_invoked(self):
        p = LifecyclePlugin(name="lc", version="1.0")
        builder = MagicMock()

        apply_plugins_to_builder(builder, [p])

        assert p.calls == ["on_install", "register", "on_enable"]

    def test_empty_plugins_list(self):
        builder = MagicMock()
        apply_plugins_to_builder(builder, [])
        # Should not raise, no-op

    def test_plugin_register_failure_raises(self):
        @dataclass
        class BadPlugin(Plugin):
            def register(self, builder):
                raise RuntimeError("broken")

        p = BadPlugin(name="bad", version="1.0")
        builder = MagicMock()

        with pytest.raises(RuntimeError, match="broken"):
            apply_plugins_to_builder(builder, [p])


# ---------------------------------------------------------------------------
# 13.3  discover_plugins
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestDiscoverPlugins:
    def test_discover_with_no_entry_points(self):
        """With no entry points defined, returns empty list."""
        with patch("importlib.metadata.entry_points") as mock_eps:
            mock_result = MagicMock()
            mock_result.select.return_value = []
            mock_eps.return_value = mock_result

            result = discover_plugins("curio_plugins")
            assert result == []

    def test_discover_loads_plugin_class(self):
        """Entry point that points to a Plugin subclass."""
        @dataclass
        class DefaultsPlugin(Plugin):
            name: str = "discovered"
            version: str = "0.1.0"
            def register(self, builder): pass

        ep = MagicMock()
        ep.name = "defaults"
        ep.load.return_value = DefaultsPlugin

        with patch("importlib.metadata.entry_points") as mock_eps:
            mock_result = MagicMock()
            mock_result.select.return_value = [ep]
            mock_eps.return_value = mock_result

            result = discover_plugins("curio_plugins")
            assert len(result) == 1
            assert isinstance(result[0], DefaultsPlugin)

    def test_discover_loads_plugin_instance(self):
        """Entry point that directly returns a Plugin instance."""
        instance = DummyPlugin(name="direct", version="1.0")
        ep = MagicMock()
        ep.name = "direct"
        ep.load.return_value = instance

        with patch("importlib.metadata.entry_points") as mock_eps:
            mock_result = MagicMock()
            mock_result.select.return_value = [ep]
            mock_eps.return_value = mock_result

            result = discover_plugins("curio_plugins")
            assert len(result) == 1
            assert result[0] is instance

    def test_discover_skips_non_plugin(self):
        """Entry point that yields something other than Plugin is skipped."""
        ep = MagicMock()
        ep.name = "bad"
        ep.load.return_value = "not a plugin"

        with patch("importlib.metadata.entry_points") as mock_eps:
            mock_result = MagicMock()
            mock_result.select.return_value = [ep]
            mock_eps.return_value = mock_result

            result = discover_plugins("curio_plugins")
            assert result == []

    def test_discover_handles_load_error(self):
        """Entry point that raises on load is skipped gracefully."""
        ep = MagicMock()
        ep.name = "broken"
        ep.load.side_effect = ImportError("no module")

        with patch("importlib.metadata.entry_points") as mock_eps:
            mock_result = MagicMock()
            mock_result.select.return_value = [ep]
            mock_eps.return_value = mock_result

            result = discover_plugins("curio_plugins")
            assert result == []
