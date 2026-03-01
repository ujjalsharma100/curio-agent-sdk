"""
Unit tests for built-in http_request tool.
"""

import urllib.error
from unittest.mock import patch, MagicMock

import pytest

from curio_agent_sdk.tools.http import http_request


@pytest.mark.unit
class TestHttpRequest:
    def test_http_get(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b'{"ok": true}'
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_resp
        mock_cm.__exit__.return_value = False
        with patch("curio_agent_sdk.tools.http.urllib.request.urlopen", return_value=mock_cm):
            result = http_request.func("https://api.example.com/data")
        assert "Status: 200" in result
        assert "ok" in result

    def test_http_post(self):
        mock_resp = MagicMock()
        mock_resp.status = 201
        mock_resp.read.return_value = b'{"id": 1}'
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_resp
        mock_cm.__exit__.return_value = False
        with patch("curio_agent_sdk.tools.http.urllib.request.urlopen", return_value=mock_cm):
            result = http_request.func(
                "https://api.example.com/create",
                method="POST",
                body='{"name": "test"}',
            )
        assert "Status: 201" in result
        assert "id" in result

    def test_http_headers(self):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.read.return_value = b"ok"
        mock_cm = MagicMock()
        mock_cm.__enter__.return_value = mock_resp
        mock_cm.__exit__.return_value = False
        with patch("curio_agent_sdk.tools.http.urllib.request.urlopen", return_value=mock_cm) as m:
            http_request.func(
                "https://api.example.com",
                headers='{"X-Custom": "value"}',
            )
            call_args = m.call_args
            req = call_args[0][0]
            # Request normalizes header names to title-case
            assert req.has_header("X-custom")

    def test_http_error(self):
        err = urllib.error.HTTPError(
            "https://api.example.com",
            404,
            "Not Found",
            {},
            None,
        )
        with patch("curio_agent_sdk.tools.http.urllib.request.urlopen", side_effect=err):
            result = http_request.func("https://api.example.com/missing")
        assert "404" in result or "Error" in result
