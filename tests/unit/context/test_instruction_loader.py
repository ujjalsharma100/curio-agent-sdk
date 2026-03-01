"""
Unit tests for InstructionLoader and load_instructions_from_file (Phase 14 — Context & Credentials).
"""

import pytest
from pathlib import Path

from curio_agent_sdk.core.context.instructions import (
    InstructionLoader,
    load_instructions_from_file,
)


@pytest.mark.unit
def test_load_from_string(tmp_path):
    """Load instructions from a file; result is returned as a single string."""
    instr = "You are an assistant. Follow these rules."
    (tmp_path / "RULES.md").write_text(instr)
    loader = InstructionLoader(
        file_names=["RULES.md"],
        search_paths=[tmp_path],
    )
    result = loader.load()
    assert result == instr


@pytest.mark.unit
def test_load_from_file(tmp_path):
    """Load instructions from file."""
    rules_file = tmp_path / "AGENT.md"
    rules_file.write_text("You are a helpful agent.\nDo your best.")
    loader = InstructionLoader(
        file_names=["AGENT.md"],
        search_paths=[tmp_path],
    )
    result = loader.load()
    assert "You are a helpful agent" in result
    assert "Do your best" in result


@pytest.mark.unit
def test_load_file_not_found():
    """Handle missing file (load_instructions_from_file returns empty string)."""
    result = load_instructions_from_file("/nonexistent/path/AGENT.md")
    assert result == ""


@pytest.mark.unit
def test_template_variables(tmp_path):
    """Variable substitution: not in loader; test merge_separator and multiple files."""
    # InstructionLoader has no template variables; test merge behavior instead
    (tmp_path / "A.md").write_text("First block")
    (tmp_path / "B.md").write_text("Second block")
    loader = InstructionLoader(
        file_names=["A.md", "B.md"],
        search_paths=[tmp_path],
        merge_separator="\n---\n",
    )
    result = loader.load()
    assert "First block" in result
    assert "Second block" in result
    assert "\n---\n" in result


@pytest.mark.unit
def test_loader_repr():
    """InstructionLoader has a useful __repr__."""
    loader = InstructionLoader(file_names=["A.md"], search_paths=[Path("/tmp")])
    r = repr(loader)
    assert "InstructionLoader" in r
    assert "A.md" in r


@pytest.mark.unit
def test_duplicate_content_deduped(tmp_path):
    """Same content from different paths is only included once."""
    content = "Same text in both files."
    (tmp_path / "dir1").mkdir()
    (tmp_path / "dir2").mkdir()
    (tmp_path / "dir1" / "R.md").write_text(content)
    (tmp_path / "dir2" / "R.md").write_text(content)
    loader = InstructionLoader(
        file_names=["R.md"],
        search_paths=[tmp_path / "dir1", tmp_path / "dir2"],
    )
    result = loader.load()
    assert result.strip() == content
    assert result.count(content) == 1
