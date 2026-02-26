"""
Rules / instructions system for Curio Agent SDK.

Loads instruction files hierarchically (global > project > directory) and
merges them into the agent's system prompt. Supports file-based rules
(AGENT.md, .agent/rules.md) and raw instruction strings.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)

# Default file names to look for (in order of precedence within each level)
DEFAULT_INSTRUCTION_FILES = ["AGENT.md", ".agent/rules.md"]

# Markers that indicate a project root when walking up from cwd
PROJECT_ROOT_MARKERS = [".git", "pyproject.toml", ".cursorrules", "AGENT.md"]


def _find_project_root(start: Path | None = None) -> Path | None:
    """
    Walk up from start (default cwd) and return the first directory that
    contains a project root marker (.git, pyproject.toml, etc.), or None.
    """
    current = Path(start or ".").resolve()
    if not current.is_dir():
        current = current.parent
    home = Path.home()
    while current != home and current != current.parent:
        for marker in PROJECT_ROOT_MARKERS:
            if (current / marker).exists():
                return current
        current = current.parent
    return None


def _default_search_paths() -> List[Path]:
    """
    Return default search path order: global (~/.agent), project root, cwd.
    Later paths override earlier in terms of specificity (directory > project > global).
    """
    cwd = Path.cwd().resolve()
    home = Path.home()
    paths: List[Path] = []

    # 1. Global: ~/.agent
    global_dir = home / ".agent"
    if global_dir.is_dir():
        paths.append(global_dir)

    # 2. Project root (if found)
    project = _find_project_root(cwd)
    if project is not None and project not in paths:
        paths.append(project)

    # 3. Current directory
    if cwd not in paths:
        paths.append(cwd)

    return paths


class InstructionLoader:
    """
    Loads instruction files hierarchically and merges them into a single string.

    Search order: global (~/.agent) → project root → current directory.
    Within each level, files are loaded in file_names order. Content is
    concatenated so that later (more specific) levels effectively override
    or extend earlier ones.

    Example:
        loader = InstructionLoader()
        instructions = loader.load()

        # Or with custom paths and file names:
        loader = InstructionLoader(
            file_names=["AGENT.md", "RULES.md", ".agent/rules.md"],
            search_paths=[Path.home() / ".agent", Path("/my/project")],
        )
        instructions = loader.load()
    """

    def __init__(
        self,
        file_names: List[str] | None = None,
        search_paths: List[Path] | None = None,
        merge_separator: str = "\n\n---\n\n",
    ):
        """
        Args:
            file_names: Names of files to load (e.g. ["AGENT.md", ".agent/rules.md"]).
                Paths in names are relative to each search path. Default: ["AGENT.md", ".agent/rules.md"].
            search_paths: Directories to search. If None, uses default: [~/.agent, project_root, cwd].
            merge_separator: String between concatenated file contents. Default: "\\n\\n---\\n\\n".
        """
        self.file_names = file_names or list(DEFAULT_INSTRUCTION_FILES)
        self.search_paths = search_paths if search_paths is not None else _default_search_paths()
        self.merge_separator = merge_separator

    def load(self) -> str:
        """
        Load and merge all instruction files from the hierarchy.

        Returns:
            Concatenated content (global + project + directory). Empty string if no files found.
        """
        parts: List[str] = []
        seen_content: set[str] = set()  # avoid duplicate file content

        for base_path in self.search_paths:
            if not base_path.exists() or not base_path.is_dir():
                continue
            for file_name in self.file_names:
                # Support file_name like ".agent/rules.md" (relative to base_path)
                full_path = (base_path / file_name).resolve()
                if not full_path.is_file():
                    continue
                try:
                    text = full_path.read_text(encoding="utf-8", errors="replace").strip()
                except OSError as e:
                    logger.warning("Could not read instruction file %s: %s", full_path, e)
                    continue
                if not text:
                    continue
                # Skip exact duplicate content (same file found via different path)
                if text in seen_content:
                    continue
                seen_content.add(text)
                parts.append(text)

        return self.merge_separator.join(parts) if parts else ""

    def __repr__(self) -> str:
        return f"InstructionLoader(file_names={self.file_names!r}, search_paths={self.search_paths!r})"


def load_instructions_from_file(path: str | Path) -> str:
    """
    Load instructions from a single file. Convenience for Builder.instructions_file().

    Args:
        path: Path to the instruction file.

    Returns:
        File content or empty string if file cannot be read.
    """
    p = Path(path)
    if not p.is_file():
        logger.warning("Instruction file not found: %s", p)
        return ""
    try:
        return p.read_text(encoding="utf-8", errors="replace").strip()
    except OSError as e:
        logger.warning("Could not read instruction file %s: %s", p, e)
        return ""
