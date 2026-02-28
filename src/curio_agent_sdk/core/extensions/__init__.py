"""Plugins, skills, and subagent."""

from curio_agent_sdk.core.extensions.plugins import (
    Plugin,
    apply_plugins_to_builder,
    discover_plugins,
)
from curio_agent_sdk.core.extensions.skills import (
    Skill,
    SkillRegistry,
    get_active_skill_prompts,
)
from curio_agent_sdk.core.extensions.subagent import SubagentConfig, AgentOrchestrator

__all__ = [
    "Plugin",
    "apply_plugins_to_builder",
    "discover_plugins",
    "Skill",
    "SkillRegistry",
    "get_active_skill_prompts",
    "SubagentConfig",
    "AgentOrchestrator",
]
