"""
Unit tests for curio_agent_sdk.utils.run_logger.

Covers: create_run_logger, use_run_logger, RunLogger (log path, write on hooks).
"""

import asyncio
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from curio_agent_sdk.utils.run_logger import (
    RunLogger,
    create_run_logger,
    use_run_logger,
)


# ---------------------------------------------------------------------------
# create_run_logger
# ---------------------------------------------------------------------------


class TestCreateRunLogger:
    def test_returns_run_logger(self):
        logger = create_run_logger()
        assert isinstance(logger, RunLogger)

    def test_default_base_name(self):
        logger = create_run_logger()
        assert logger.base_name == "agent-run"

    def test_custom_base_name_and_output_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = create_run_logger(output_dir=tmp, base_name="custom")
            assert logger.output_dir == tmp
            assert logger.base_name == "custom"

    def test_get_log_path_none_until_write(self):
        logger = create_run_logger(base_name="test")
        assert logger.get_log_path() is None


# ---------------------------------------------------------------------------
# RunLogger — file creation and content
# ---------------------------------------------------------------------------


class TestRunLoggerWrites:
    @pytest.mark.asyncio
    async def test_on_run_before_creates_file_and_sets_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(output_dir=tmp, base_name="verify-run")
            assert logger.get_log_path() is None
            ctx = MagicMock()
            ctx.data = {"input": "Hello"}
            ctx.run_id = "run-1"
            ctx.agent_id = "agent-1"
            await logger.on_run_before(ctx)
            path = logger.get_log_path()
            assert path is not None
            assert Path(path).parent == Path(tmp)
            assert "verify-run-" in Path(path).name
            assert path.endswith(".log")
            content = Path(path).read_text()
            assert "[AGENT RUN START]" in content
            assert "Hello" in content
            assert "run-1" in content

    @pytest.mark.asyncio
    async def test_on_run_after_writes_output_and_tool_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(output_dir=tmp, base_name="verify")
            ctx_before = MagicMock()
            ctx_before.data = {"input": "Hi"}
            ctx_before.run_id = ""
            ctx_before.agent_id = ""
            await logger.on_run_before(ctx_before)
            path = logger.get_log_path()
            ctx_after = MagicMock()
            ctx_after.data = {"output": "Done.", "result": {"output": "Done.", "tool_calls": 2}}
            ctx_after.state = None
            await logger.on_run_after(ctx_after)
            content = Path(path).read_text()
            assert "[AGENT RUN END]" in content
            assert "Done." in content
            assert "toolCalls count: 2" in content

    @pytest.mark.asyncio
    async def test_on_run_error_writes_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            logger = RunLogger(output_dir=tmp, base_name="err")
            ctx_before = MagicMock()
            ctx_before.data = {"input": "x"}
            ctx_before.run_id = ctx_before.agent_id = ""
            await logger.on_run_before(ctx_before)
            ctx_err = MagicMock()
            ctx_err.data = {"error": "Something failed"}
            await logger.on_run_error(ctx_err)
            content = Path(logger.get_log_path()).read_text()
            assert "[AGENT RUN ERROR]" in content
            assert "Something failed" in content


# ---------------------------------------------------------------------------
# use_run_logger
# ---------------------------------------------------------------------------


class TestUseRunLogger:
    def test_registers_hooks_and_returns_logger(self):
        builder = MagicMock()
        logger = use_run_logger(builder, base_name="hook-test", output_dir=os.getcwd())
        assert isinstance(logger, RunLogger)
        assert logger.base_name == "hook-test"
        assert builder.hook.call_count >= 9  # run before/after/error, llm before/after/error, tool before/after/error

    @pytest.mark.asyncio
    async def test_use_run_logger_logger_writes_when_hooks_fired(self):
        with tempfile.TemporaryDirectory() as tmp:
            builder = MagicMock()
            logger = use_run_logger(builder, base_name="integ", output_dir=tmp)
            ctx = MagicMock()
            ctx.data = {"input": "Test input"}
            ctx.run_id = "r1"
            ctx.agent_id = "a1"
            await logger.on_run_before(ctx)
            assert logger.get_log_path() is not None
            assert "[AGENT RUN START]" in Path(logger.get_log_path()).read_text()
