"""
Skills system — packaged, reusable agent capabilities.

A Skill bundles:
- name, description
- system_prompt and instructions (injected when skill is active)
- tools (added to the agent when skill is active or at build time)
- hooks (lifecycle handlers)

Skills can be loaded from directories (manifest + prompt + tools + hooks)
and registered on an agent. SkillRegistry supports activate/deactivate
per state for mid-run scoping (e.g. invoke_skill activates one skill only).
"""

from __future__ import annotations

import importlib.util
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from curio_agent_sdk.core.state import AgentState
from curio_agent_sdk.core.tools.tool import Tool

logger = logging.getLogger(__name__)

# Metadata keys used on state for active skills (internal)
_SKILL_TOOLS_KEY = "_skill_tools"
_SKILL_PROMPTS_KEY = "_skill_prompts"


@dataclass
class Skill:
    """
    A packaged, reusable agent capability.

    Bundles a name, description, optional system prompt and instructions,
    tools, and hooks. When a skill is activated on a state, its tools
    are added to the state and its prompt is injected into the run.
    """

    name: str
    description: str = ""
    system_prompt: str = ""
    tools: list[Tool] = field(default_factory=list)
    hooks: list[tuple[str, Callable[..., Any], int]] = field(default_factory=list)  # (event, handler, priority)
    instructions: str = ""

    def get_combined_prompt(self) -> str:
        """Return system_prompt and instructions merged for injection."""
        parts = [self.system_prompt.strip(), self.instructions.strip()]
        return "\n\n".join(p for p in parts if p).strip()

    @classmethod
    def from_directory(cls, path: str | Path) -> Skill:
        """
        Load a skill from a directory with a manifest.

        Expected layout:
            skill_dir/
                manifest.yaml   (or manifest.yml) — name, description, prompt, tools, hooks
                prompt.md      — optional; used if manifest has prompt: "prompt.md"
                tools.py       — optional; must define get_tools() -> list[Tool] or TOOLS
                hooks.py       — optional; must define get_hooks() -> list[(event, handler)] or [(e, h, priority)]

        manifest.yaml example:
            name: commit
            description: Create well-formatted git commits
            prompt: prompt.md
            tools: tools.py
            hooks: hooks.py
        """
        path = Path(path).resolve()
        if not path.is_dir():
            raise NotADirectoryError(f"Skill path is not a directory: {path}")

        manifest_path = path / "manifest.yaml"
        if not manifest_path.exists():
            manifest_path = path / "manifest.yml"
        if not manifest_path.exists():
            raise FileNotFoundError(f"No manifest.yaml or manifest.yml in {path}")

        manifest = _load_yaml(manifest_path)
        name = manifest.get("name") or path.name
        description = manifest.get("description", "")

        # Load prompt
        system_prompt = ""
        prompt_ref = manifest.get("prompt")
        if prompt_ref:
            prompt_path = path / prompt_ref if not Path(prompt_ref).is_absolute() else Path(prompt_ref)
            if prompt_path.exists():
                system_prompt = prompt_path.read_text(encoding="utf-8").strip()
            else:
                logger.warning("Skill %s: prompt file not found: %s", name, prompt_path)

        # Load tools
        tools: list[Tool] = []
        tools_ref = manifest.get("tools")
        if tools_ref:
            tools = _load_tools_from_module(path, tools_ref)

        # Load hooks: list of (event, handler) or (event, handler, priority)
        hooks: list[tuple[str, Callable[..., Any], int]] = []
        hooks_ref = manifest.get("hooks")
        if hooks_ref:
            raw_hooks = _load_hooks_from_module(path, hooks_ref)
            for item in raw_hooks:
                if len(item) == 2:
                    event, handler = item
                    hooks.append((event, handler, 0))
                else:
                    event, handler, priority = item[:3]
                    hooks.append((event, handler, int(priority)))

        instructions = manifest.get("instructions", "").strip() or ""

        return cls(
            name=name,
            description=description,
            system_prompt=system_prompt,
            tools=tools,
            hooks=hooks,
            instructions=instructions,
        )


def _load_yaml(path: Path) -> dict[str, Any]:
    """Load a YAML file; require PyYAML or use minimal parser for simple key-value."""
    try:
        import yaml
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except ImportError:
        # Minimal key: value parser for manifest (no dependency)
        out: dict[str, Any] = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if ":" in line:
                    k, _, v = line.partition(":")
                    out[k.strip()] = v.strip().strip("'\"").strip('"')
        return out


def _load_tools_from_module(base_path: Path, module_ref: str) -> list[Tool]:
    """Load tools from a Python file. Expect get_tools() -> list[Tool] or TOOLS = [Tool, ...]."""
    tools_path = base_path / module_ref if not Path(module_ref).is_absolute() else Path(module_ref)
    if not tools_path.exists():
        logger.warning("Skill tools file not found: %s", tools_path)
        return []

    spec = importlib.util.spec_from_file_location("skill_tools", tools_path)
    if spec is None or spec.loader is None:
        return []
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "get_tools") and callable(mod.get_tools):
        return list(mod.get_tools())
    if hasattr(mod, "TOOLS"):
        return list(mod.TOOLS)
    return []


def _load_hooks_from_module(base_path: Path, module_ref: str) -> list[tuple]:
    """Load hooks from a Python file. Expect get_hooks() -> list of (event, handler) or (event, handler, priority)."""
    hooks_path = base_path / module_ref if not Path(module_ref).is_absolute() else Path(module_ref)
    if not hooks_path.exists():
        logger.warning("Skill hooks file not found: %s", hooks_path)
        return []

    spec = importlib.util.spec_from_file_location("skill_hooks", hooks_path)
    if spec is None or spec.loader is None:
        return []
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    if hasattr(mod, "get_hooks") and callable(mod.get_hooks):
        return list(mod.get_hooks())
    if hasattr(mod, "HOOKS"):
        return list(mod.HOOKS)
    return []


class SkillRegistry:
    """
    Register and discover skills; activate/deactivate per state.

    At build time, skills can be merged into the agent (tools + hooks).
    At run time, activate(name, state) adds that skill's tools to the state
    and records its prompt for injection; deactivate(name, state) removes them.
    """

    def __init__(self) -> None:
        self._skills: dict[str, Skill] = {}

    def register(self, skill: Skill) -> None:
        """Register a skill by name."""
        self._skills[skill.name] = skill

    def get(self, name: str) -> Skill | None:
        """Return the skill with the given name, or None."""
        return self._skills.get(name)

    def list(self) -> list[Skill]:
        """Return all registered skills."""
        return list(self._skills.values())

    def list_names(self) -> list[str]:
        """Return names of all registered skills."""
        return list(self._skills.keys())

    def activate(self, name: str, state: AgentState) -> None:
        """
        Activate a skill on the given state: add its tools and record its prompt.

        Tools are appended to state.tools and state.tool_schemas; the skill's
        combined prompt is stored so Runtime can inject it into the system message.
        """
        skill = self._skills.get(name)
        if skill is None:
            raise ValueError(f"Unknown skill: {name}")

        # Ensure metadata dicts exist
        if _SKILL_TOOLS_KEY not in state.metadata:
            state.metadata[_SKILL_TOOLS_KEY] = {}
        if _SKILL_PROMPTS_KEY not in state.metadata:
            state.metadata[_SKILL_PROMPTS_KEY] = {}

        # Avoid double-activation
        if name in state.metadata[_SKILL_TOOLS_KEY]:
            return

        added_tools = list(skill.tools)
        state.metadata[_SKILL_TOOLS_KEY][name] = added_tools
        state.metadata[_SKILL_PROMPTS_KEY][name] = skill.get_combined_prompt()

        for t in added_tools:
            if t not in state.tools:
                state.tools.append(t)
                try:
                    state.tool_schemas.append(t.to_llm_schema())
                except Exception:
                    pass

    def deactivate(self, name: str, state: AgentState) -> None:
        """
        Deactivate a skill on the given state: remove its tools and prompt.
        """
        skill_tools = state.metadata.get(_SKILL_TOOLS_KEY) or {}
        skill_prompts = state.metadata.get(_SKILL_PROMPTS_KEY) or {}

        if name not in skill_tools:
            return

        to_remove = set(skill_tools[name])
        state.tools = [t for t in state.tools if t not in to_remove]
        state.tool_schemas = [t.to_llm_schema() for t in state.tools]

        del skill_tools[name]
        if name in skill_prompts:
            del skill_prompts[name]


def get_active_skill_prompts(state: AgentState) -> str:
    """
    Return combined prompt text for all skills active on this state.
    Used by Runtime to inject into the system message.
    """
    prompts = state.metadata.get(_SKILL_PROMPTS_KEY) or {}
    if not prompts:
        return ""
    return "\n\n---\n\n".join(prompts.values()).strip()
