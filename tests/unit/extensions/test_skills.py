"""
Unit tests for the Skills system — Skill dataclass, SkillRegistry,
directory loading, and active skill prompt injection.
"""

import os
import textwrap

import pytest

from curio_agent_sdk.core.extensions.skills import (
    Skill,
    SkillRegistry,
    get_active_skill_prompts,
    _load_yaml,
    _load_tools_from_module,
    _load_hooks_from_module,
    _SKILL_TOOLS_KEY,
    _SKILL_PROMPTS_KEY,
)
from curio_agent_sdk.core.state.state import AgentState
from curio_agent_sdk.core.tools.tool import Tool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _dummy_tool(name: str = "greet") -> Tool:
    """Create a minimal Tool for testing."""
    def greet(name: str = "World") -> str:
        """Say hello."""
        return f"Hello, {name}!"
    return Tool(func=greet, name=name)


def _make_state(**kwargs) -> AgentState:
    """Create a fresh AgentState for testing."""
    return AgentState(**kwargs)


# ---------------------------------------------------------------------------
# 13.1  Skill dataclass
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSkillCreation:
    def test_skill_creation_defaults(self):
        s = Skill(name="test")
        assert s.name == "test"
        assert s.description == ""
        assert s.system_prompt == ""
        assert s.instructions == ""
        assert s.tools == []
        assert s.hooks == []

    def test_skill_creation_full(self):
        t = _dummy_tool()
        hook = ("on_start", lambda: None, 0)
        s = Skill(
            name="commit",
            description="Git commits",
            system_prompt="You are a git assistant.",
            instructions="Always sign commits.",
            tools=[t],
            hooks=[hook],
        )
        assert s.name == "commit"
        assert s.description == "Git commits"
        assert len(s.tools) == 1
        assert len(s.hooks) == 1

    def test_skill_combined_prompt_both(self):
        s = Skill(name="x", system_prompt="System.", instructions="Instructions.")
        assert s.get_combined_prompt() == "System.\n\nInstructions."

    def test_skill_combined_prompt_system_only(self):
        s = Skill(name="x", system_prompt="System only.")
        assert s.get_combined_prompt() == "System only."

    def test_skill_combined_prompt_instructions_only(self):
        s = Skill(name="x", instructions="Instructions only.")
        assert s.get_combined_prompt() == "Instructions only."

    def test_skill_combined_prompt_empty(self):
        s = Skill(name="x")
        assert s.get_combined_prompt() == ""

    def test_skill_combined_prompt_strips_whitespace(self):
        s = Skill(name="x", system_prompt="  A  ", instructions="  B  ")
        assert s.get_combined_prompt() == "A\n\nB"


# ---------------------------------------------------------------------------
# 13.1  Skill.from_directory
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSkillFromDirectory:
    def test_from_directory_basic(self, tmp_path):
        """Load skill from directory with YAML manifest and prompt file."""
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text(textwrap.dedent("""\
            name: review
            description: Code review skill
            prompt: prompt.md
        """))
        (tmp_path / "prompt.md").write_text("You are a code reviewer.")

        skill = Skill.from_directory(tmp_path)
        assert skill.name == "review"
        assert skill.description == "Code review skill"
        assert skill.system_prompt == "You are a code reviewer."
        assert skill.tools == []
        assert skill.hooks == []

    def test_from_directory_yml_extension(self, tmp_path):
        """manifest.yml is also accepted."""
        manifest = tmp_path / "manifest.yml"
        manifest.write_text("name: alt\ndescription: alt skill\n")

        skill = Skill.from_directory(tmp_path)
        assert skill.name == "alt"

    def test_from_directory_no_manifest_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="No manifest"):
            Skill.from_directory(tmp_path)

    def test_from_directory_not_a_dir(self, tmp_path):
        f = tmp_path / "file.txt"
        f.write_text("x")
        with pytest.raises(NotADirectoryError):
            Skill.from_directory(f)

    def test_from_directory_with_tools(self, tmp_path):
        """Load tools from a Python module referenced in manifest."""
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("name: tools_skill\ntools: tools.py\n")

        tools_py = tmp_path / "tools.py"
        tools_py.write_text(textwrap.dedent("""\
            from curio_agent_sdk.core.tools.tool import Tool

            def _greet(name: str = "World") -> str:
                \"\"\"Say hello.\"\"\"
                return f"Hello, {name}!"

            def get_tools():
                return [Tool(func=_greet, name="greet")]
        """))

        skill = Skill.from_directory(tmp_path)
        assert len(skill.tools) == 1
        assert skill.tools[0].name == "greet"

    def test_from_directory_with_hooks(self, tmp_path):
        """Load hooks from a Python module with get_hooks()."""
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("name: hooks_skill\nhooks: hooks.py\n")

        hooks_py = tmp_path / "hooks.py"
        hooks_py.write_text(textwrap.dedent("""\
            def _handler():
                pass

            def get_hooks():
                return [("on_start", _handler)]
        """))

        skill = Skill.from_directory(tmp_path)
        assert len(skill.hooks) == 1
        event, handler, priority = skill.hooks[0]
        assert event == "on_start"
        assert priority == 0  # default when tuple has 2 elements

    def test_from_directory_hooks_with_priority(self, tmp_path):
        """Hooks with explicit priority (3-tuple)."""
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("name: prio_skill\nhooks: hooks.py\n")

        hooks_py = tmp_path / "hooks.py"
        hooks_py.write_text(textwrap.dedent("""\
            def _handler():
                pass

            def get_hooks():
                return [("on_end", _handler, 5)]
        """))

        skill = Skill.from_directory(tmp_path)
        assert len(skill.hooks) == 1
        event, handler, priority = skill.hooks[0]
        assert event == "on_end"
        assert priority == 5

    def test_from_directory_missing_prompt_file(self, tmp_path):
        """If prompt file referenced but not found, system_prompt stays empty."""
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("name: missing_prompt\nprompt: nonexistent.md\n")

        skill = Skill.from_directory(tmp_path)
        assert skill.system_prompt == ""

    def test_from_directory_missing_tools_file(self, tmp_path):
        """If tools file referenced but not found, tools stays empty."""
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("name: missing_tools\ntools: nonexistent.py\n")

        skill = Skill.from_directory(tmp_path)
        assert skill.tools == []

    def test_from_directory_instructions(self, tmp_path):
        """Instructions field from manifest."""
        manifest = tmp_path / "manifest.yaml"
        manifest.write_text("name: instr\ninstructions: Do it carefully.\n")

        skill = Skill.from_directory(tmp_path)
        assert skill.instructions == "Do it carefully."

    def test_from_directory_name_falls_back_to_dirname(self, tmp_path):
        """When manifest has no name, fall back to directory name."""
        skill_dir = tmp_path / "my_skill"
        skill_dir.mkdir()
        manifest = skill_dir / "manifest.yaml"
        manifest.write_text("description: no name here\n")

        skill = Skill.from_directory(skill_dir)
        assert skill.name == "my_skill"


# ---------------------------------------------------------------------------
# 13.1  Internal helpers
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestYamlLoader:
    def test_load_yaml_basic(self, tmp_path):
        f = tmp_path / "test.yaml"
        f.write_text("name: hello\ndescription: world\n")
        data = _load_yaml(f)
        assert data["name"] == "hello"
        assert data["description"] == "world"

    def test_load_yaml_empty(self, tmp_path):
        f = tmp_path / "empty.yaml"
        f.write_text("")
        data = _load_yaml(f)
        assert data == {}


@pytest.mark.unit
class TestToolsFromModule:
    def test_tools_from_module_TOOLS_var(self, tmp_path):
        """Module that exports TOOLS list."""
        tools_py = tmp_path / "tools.py"
        tools_py.write_text(textwrap.dedent("""\
            from curio_agent_sdk.core.tools.tool import Tool

            def _fn(x: str = "") -> str:
                \"\"\"A tool.\"\"\"
                return x

            TOOLS = [Tool(func=_fn, name="my_tool")]
        """))

        result = _load_tools_from_module(tmp_path, "tools.py")
        assert len(result) == 1
        assert result[0].name == "my_tool"

    def test_tools_from_module_not_found(self, tmp_path):
        result = _load_tools_from_module(tmp_path, "nope.py")
        assert result == []


@pytest.mark.unit
class TestHooksFromModule:
    def test_hooks_from_module_HOOKS_var(self, tmp_path):
        """Module that exports HOOKS list."""
        hooks_py = tmp_path / "hooks.py"
        hooks_py.write_text(textwrap.dedent("""\
            def _h():
                pass

            HOOKS = [("event_a", _h, 10)]
        """))

        result = _load_hooks_from_module(tmp_path, "hooks.py")
        assert len(result) == 1
        assert result[0][0] == "event_a"

    def test_hooks_from_module_not_found(self, tmp_path):
        result = _load_hooks_from_module(tmp_path, "missing.py")
        assert result == []


# ---------------------------------------------------------------------------
# 13.1  SkillRegistry
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestSkillRegistry:
    def test_register_and_get(self):
        reg = SkillRegistry()
        s = Skill(name="git", description="Git helper")
        reg.register(s)
        assert reg.get("git") is s
        assert reg.get("unknown") is None

    def test_list_skills(self):
        reg = SkillRegistry()
        s1 = Skill(name="a")
        s2 = Skill(name="b")
        reg.register(s1)
        reg.register(s2)
        assert len(reg.list()) == 2

    def test_list_names(self):
        reg = SkillRegistry()
        reg.register(Skill(name="x"))
        reg.register(Skill(name="y"))
        assert set(reg.list_names()) == {"x", "y"}

    def test_activate_adds_tools_and_prompts(self):
        reg = SkillRegistry()
        t = _dummy_tool("review_tool")
        s = Skill(name="review", system_prompt="Review code.", tools=[t])
        reg.register(s)

        state = _make_state()
        reg.activate("review", state)

        assert t in state.tools
        assert _SKILL_PROMPTS_KEY in state.metadata
        assert "review" in state.metadata[_SKILL_PROMPTS_KEY]

    def test_activate_unknown_skill_raises(self):
        reg = SkillRegistry()
        state = _make_state()
        with pytest.raises(ValueError, match="Unknown skill"):
            reg.activate("nonexistent", state)

    def test_activate_double_is_noop(self):
        reg = SkillRegistry()
        t = _dummy_tool("t1")
        s = Skill(name="s1", tools=[t])
        reg.register(s)

        state = _make_state()
        reg.activate("s1", state)
        reg.activate("s1", state)

        # tool should appear only once
        assert state.tools.count(t) == 1

    def test_deactivate_removes_tools_and_prompts(self):
        reg = SkillRegistry()
        t = _dummy_tool("t2")
        s = Skill(name="s2", system_prompt="Prompt.", tools=[t])
        reg.register(s)

        state = _make_state()
        reg.activate("s2", state)
        assert t in state.tools

        reg.deactivate("s2", state)
        assert t not in state.tools
        assert "s2" not in state.metadata.get(_SKILL_PROMPTS_KEY, {})

    def test_deactivate_not_active_is_noop(self):
        reg = SkillRegistry()
        state = _make_state()
        # should not raise
        reg.deactivate("whatever", state)


# ---------------------------------------------------------------------------
# 13.1  get_active_skill_prompts
# ---------------------------------------------------------------------------

@pytest.mark.unit
class TestGetActiveSkillPrompts:
    def test_no_active_skills(self):
        state = _make_state()
        assert get_active_skill_prompts(state) == ""

    def test_single_active_skill(self):
        reg = SkillRegistry()
        s = Skill(name="a", system_prompt="Prompt A.")
        reg.register(s)
        state = _make_state()
        reg.activate("a", state)

        result = get_active_skill_prompts(state)
        assert "Prompt A." in result

    def test_multiple_active_skills(self):
        reg = SkillRegistry()
        reg.register(Skill(name="a", system_prompt="Alpha."))
        reg.register(Skill(name="b", system_prompt="Beta."))
        state = _make_state()
        reg.activate("a", state)
        reg.activate("b", state)

        result = get_active_skill_prompts(state)
        assert "Alpha." in result
        assert "Beta." in result
        assert "---" in result  # separator
