"""
Unit tests for built-in web_fetch tool.
"""

import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from curio_agent_sdk.tools.web import web_fetch


@pytest.mark.unit
class TestWebFetch:
    def test_web_fetch_success(self):
        mock_resp = MagicMock()
        mock_resp.read.return_value = b"<html><body>Hello World</body></html>"
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_resp
        mock_cm.__exit__.return_value = False
        with patch("curio_agent_sdk.tools.web.urllib.request.urlopen", return_value=mock_cm):
            result = web_fetch.func("https://example.com")
        assert "Hello World" in result

    def test_web_fetch_error(self):
        with patch(
            "curio_agent_sdk.tools.web.urllib.request.urlopen",
            side_effect=urllib.error.URLError("connection refused"),
        ):
            result = web_fetch.func("https://example.com")
        assert "Error" in result

    def test_web_fetch_timeout(self):
        with patch(
            "curio_agent_sdk.tools.web.urllib.request.urlopen",
            side_effect=TimeoutError("timed out"),
        ):
            result = web_fetch.func("https://example.com")
        assert "Error" in result
