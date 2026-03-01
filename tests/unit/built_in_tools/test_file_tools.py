"""
Unit tests for built-in file tools (file_read, file_write).
"""

import pytest

from curio_agent_sdk.core.tools.tool import Tool
from curio_agent_sdk.tools.file import file_read, file_write


@pytest.mark.unit
class TestFileRead:
    def test_file_read_existing(self, tmp_path):
        f = tmp_path / "hello.txt"
        f.write_text("hello world", encoding="utf-8")
        result = file_read.func(str(f))
        assert result == "hello world"

    def test_file_read_not_found(self):
        result = file_read.func("/nonexistent/path/xyz.txt")
        assert "Error" in result
        assert "not found" in result

    def test_file_read_is_tool(self):
        assert isinstance(file_read, Tool)
        assert file_read.name == "file_read"
        assert "Read" in file_read.description


@pytest.mark.unit
class TestFileWrite:
    def test_file_write_new(self, tmp_path):
        path = str(tmp_path / "new.txt")
        result = file_write.func(path, "new content")
        assert "Successfully wrote" in result
        assert "11 characters" in result
        assert (tmp_path / "new.txt").read_text(encoding="utf-8") == "new content"

    def test_file_write_overwrite(self, tmp_path):
        f = tmp_path / "existing.txt"
        f.write_text("old", encoding="utf-8")
        result = file_write.func(str(f), "new")
        assert "Successfully wrote" in result
        assert f.read_text(encoding="utf-8") == "new"

    def test_file_write_is_tool(self):
        assert isinstance(file_write, Tool)
        assert file_write.name == "file_write"
        assert "Write" in file_write.description
