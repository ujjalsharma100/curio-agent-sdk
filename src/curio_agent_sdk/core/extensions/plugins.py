"""
Plugin system — bundle tools, hooks, middleware, memory, connectors, skills, etc.
into distributable units.

Plugins are **build-time** constructs: they receive an `AgentBuilder` and register
their components (tools, middleware, hooks, instructions, connectors, skills, ...)
via the builder API. This keeps the Agent and Runtime unaware of individual plugins
while still giving users a unified extension mechanism.

Key concepts:

- `Plugin` base class:
    - `name`, `version`, `dependencies` metadata
    - optional configuration dictionary with lightweight validation
    - `register(builder)` for wiring the plugin into an agent
    - lifecycle hooks: `on_install`, `on_enable`, `on_disable`
    - optional metadata for conflict detection (`provides_tools`, `provides_hooks`)

- Discovery:
    - `discover_plugins(group="curio_plugins")` loads plugins from entry points
      declared by installed packages.

- Orchestration:
    - `apply_plugins_to_builder(builder, plugins)`:
        - resolves dependency order
        - performs basic conflict detection
        - calls lifecycle hooks and `register()` for each plugin
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Tuple, TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from curio_agent_sdk.core.agent import AgentBuilder


@dataclass
class Plugin(ABC):
    """
    Base class for Curio plugins.

    A Plugin bundles tools, hooks, middleware, memory backends, connectors,
    skills, and/or instructions into a single distributable unit.

    Subclasses should at minimum set:
        - name: unique plugin name (e.g. "git")
        - version: semantic version string
        - dependencies: list of other plugin names that must be loaded first
        - implements register(builder) to wire components into the AgentBuilder

    Optional:
        - provides_tools: logical tool names this plugin provides (for conflict detection)
        - provides_hooks: (event, priority) pairs this plugin registers (for diagnostics)
        - get_config_schema(): return a JSON-schema-like dict for config validation
    """

    name: str
    version: str
    dependencies: List[str] = field(default_factory=list)

    # Optional metadata for diagnostics / conflict detection
    provides_tools: List[str] = field(default_factory=list)
    provides_hooks: List[Tuple[str, int]] = field(default_factory=list)

    # Arbitrary plugin-specific configuration
    config: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        # Validate configuration against optional schema.
        try:
            self._validate_config()
        except Exception as exc:  # Defensive: never crash import-time
            logger.error("Plugin %s configuration invalid: %s", self.name, exc)
            raise

    # ── Core API ──────────────────────────────────────────────────────

    @abstractmethod
    def register(self, builder: "AgentBuilder") -> None:
        """
        Register all components with the given AgentBuilder.

        Typical implementation calls builder methods such as:
            - builder.tools([...])
            - builder.middleware([...])
            - builder.hook("event", handler, priority=0)
            - builder.instructions("...")
            - builder.connector(connector_instance)
            - builder.skill(skill_instance)
        """

    def get_config_schema(self) -> Dict[str, Any] | None:
        """
        Optional JSON-schema-like description of the plugin's configuration.

        Only a very small subset is honoured:
            {
                "type": "object",
                "required": ["foo", "bar"],
                "properties": {
                    "foo": {"type": "string"},
                    "bar": {"type": "integer"},
                },
            }

        This is intentionally minimal to avoid bringing in a heavy validator.
        """
        return None

    # ── Lifecycle hooks ───────────────────────────────────────────────

    def on_install(self, builder: "AgentBuilder") -> None:  # pragma: no cover - default no-op
        """
        Called once per builder before `register()` when the plugin is applied.

        Use for one-time setup that depends on the builder but doesn't mutate it
        (e.g. logging, sanity checks). The default implementation is a no-op.
        """

    def on_enable(self, builder: "AgentBuilder") -> None:  # pragma: no cover - default no-op
        """
        Called after `register()` when the plugin has successfully wired itself
        into the builder. The default implementation is a no-op.
        """

    def on_disable(self, builder: "AgentBuilder") -> None:  # pragma: no cover - default no-op
        """
        Placeholder for future runtime enable/disable support.
        Currently unused but provided to match the roadmap design.
        """

    # ── Internal helpers ──────────────────────────────────────────────

    def _validate_config(self) -> None:
        """
        Very lightweight config validation based on get_config_schema().

        Ensures required keys are present; deeper validation is left to
        plugin implementations if needed.
        """
        schema = self.get_config_schema()
        if not schema:
            return
        if not isinstance(self.config, dict):
            raise ValueError("Plugin config must be a dict.")

        required = schema.get("required") or []
        missing = [key for key in required if key not in self.config]
        if missing:
            raise ValueError(
                f"Missing required config keys for plugin {self.name}: {', '.join(missing)}"
            )


# ── Discovery & orchestration helpers ─────────────────────────────────


def _toposort_plugins(plugins: Iterable[Plugin]) -> List[Plugin]:
    """
    Topologically sort plugins by their dependencies.

    Dependencies refer to plugin.name values. Unknown dependencies are logged
    as warnings and otherwise ignored (the plugin is still loaded).
    Cycles are detected and logged; in that case, the original order is used
    for the remaining plugins in the cycle.
    """
    by_name: Dict[str, Plugin] = {}
    for plugin in plugins:
        if plugin.name in by_name:
            logger.warning(
                "Duplicate plugin name %s; keeping the last instance.", plugin.name
            )
        by_name[plugin.name] = plugin

    # Build adjacency lists for Kahn's algorithm.
    incoming_count: Dict[str, int] = {name: 0 for name in by_name}
    outgoing: Dict[str, List[str]] = {name: [] for name in by_name}

    for name, plugin in by_name.items():
        for dep in plugin.dependencies:
            if dep not in by_name:
                logger.warning(
                    "Plugin %s declares unknown dependency %s; ignoring.",
                    name,
                    dep,
                )
                continue
            incoming_count[name] += 1
            outgoing[dep].append(name)

    # Start with plugins that have no incoming edges.
    ordered: List[Plugin] = []
    queue: List[str] = [name for name, count in incoming_count.items() if count == 0]

    while queue:
        current = queue.pop()
        ordered.append(by_name[current])
        for neighbor in outgoing[current]:
            incoming_count[neighbor] -= 1
            if incoming_count[neighbor] == 0:
                queue.append(neighbor)

    if len(ordered) != len(by_name):
        # There is a cycle or unresolved dependency; append remaining in arbitrary order.
        remaining = [name for name in by_name if name not in {p.name for p in ordered}]
        logger.warning(
            "Detected cyclic or unresolved plugin dependencies among: %s",
            ", ".join(sorted(remaining)),
        )
        for name in remaining:
            ordered.append(by_name[name])

    return ordered


def _detect_conflicts(plugins: Iterable[Plugin]) -> Dict[str, List[str]]:
    """
    Detect basic conflicts between plugins based on their declared metadata.

    Currently checks:
        - duplicate tool names in `provides_tools`
    Returns a mapping: thing_name -> [plugin_name, ...] for entries with >1 owner.
    """
    owners: Dict[str, List[str]] = {}
    for plugin in plugins:
        for tool_name in getattr(plugin, "provides_tools", []) or []:
            owners.setdefault(tool_name, []).append(plugin.name)
    return {name: pls for name, pls in owners.items() if len(pls) > 1}


def apply_plugins_to_builder(builder: "AgentBuilder", plugins: Iterable[Plugin]) -> None:
    """
    Apply the given plugins to an AgentBuilder.

    - Resolves dependency order
    - Logs basic conflicts (duplicate tool names)
    - Invokes lifecycle hooks and `register()` on each plugin
    """
    plugin_list = list(plugins)
    if not plugin_list:
        return

    ordered = _toposort_plugins(plugin_list)

    conflicts = _detect_conflicts(ordered)
    for name, plugin_names in conflicts.items():
        logger.warning(
            "Plugin conflict: tool %s is provided by multiple plugins: %s",
            name,
            ", ".join(sorted(plugin_names)),
        )

    for plugin in ordered:
        try:
            plugin.on_install(builder)
            plugin.register(builder)
            plugin.on_enable(builder)
            logger.info(
                "Applied plugin %s (version %s) to builder.",
                plugin.name,
                plugin.version,
            )
        except Exception as exc:
            logger.exception(
                "Plugin %s failed to register with builder: %s", plugin.name, exc
            )
            raise


def discover_plugins(entry_point_group: str = "curio_plugins") -> List[Plugin]:
    """
    Discover plugins from installed packages via entry points.

    Packages can expose plugins by defining entry points in their pyproject.toml,
    for example:

        [project.entry-points."curio_plugins"]
        git = "my_package.plugins:GitPlugin"

    The object referenced by the entry point can be either:
        - a Plugin instance, or
        - a Plugin subclass (which will be instantiated with no arguments).
    """
    try:
        # Python 3.10+ style API
        from importlib.metadata import entry_points
    except ImportError:  # pragma: no cover - very old environments
        logger.warning("importlib.metadata is not available; plugin discovery disabled.")
        return []

    try:
        eps = entry_points()
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("Failed to load entry points for plugins: %s", exc)
        return []

    # Support both modern `.select(group=...)` and legacy `eps[group]` styles.
    selected = []
    if hasattr(eps, "select"):
        selected = list(eps.select(group=entry_point_group))
    else:  # pragma: no cover - older importlib.metadata
        selected = list(eps.get(entry_point_group, []))

    plugins: List[Plugin] = []
    for ep in selected:
        try:
            obj = ep.load()
            if isinstance(obj, Plugin):
                plugin = obj
            elif isinstance(obj, type) and issubclass(obj, Plugin):
                plugin = obj()  # type: ignore[call-arg]
            else:
                logger.warning(
                    "Entry point %s in group %s did not yield a Plugin or Plugin subclass.",
                    ep.name,
                    entry_point_group,
                )
                continue
            plugins.append(plugin)
        except Exception as exc:
            logger.warning(
                "Failed to load plugin from entry point %s (%s): %s",
                ep.name,
                entry_point_group,
                exc,
            )

    return _toposort_plugins(plugins)

