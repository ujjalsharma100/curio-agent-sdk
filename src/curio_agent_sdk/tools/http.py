"""Built-in HTTP request tool."""

import json
import urllib.request
import urllib.error

from curio_agent_sdk.core.tools.tool import tool


@tool(name="http_request", description="Make an HTTP request and return the response", timeout=30.0)
def http_request(url: str, method: str = "GET", headers: str = "", body: str = "") -> str:
    """
    Make an HTTP request.

    Args:
        url: The URL to request.
        method: HTTP method (GET, POST, PUT, DELETE, PATCH).
        headers: JSON string of headers, e.g. '{"Content-Type": "application/json"}'.
        body: Request body string.
    """
    try:
        parsed_headers = {}
        if headers:
            try:
                parsed_headers = json.loads(headers)
            except json.JSONDecodeError:
                return "Error: headers must be a valid JSON string"

        data = body.encode("utf-8") if body else None
        req = urllib.request.Request(url, data=data, method=method.upper())
        req.add_header("User-Agent", "CurioAgent/1.0")
        for k, v in parsed_headers.items():
            req.add_header(k, v)

        with urllib.request.urlopen(req, timeout=25) as resp:
            status = resp.status
            resp_body = resp.read().decode("utf-8", errors="replace")

        if len(resp_body) > 10000:
            resp_body = resp_body[:10000] + "\n... [truncated]"

        return f"Status: {status}\n\n{resp_body}"

    except urllib.error.HTTPError as e:
        body_text = ""
        try:
            body_text = e.read().decode("utf-8", errors="replace")[:2000]
        except Exception:
            pass
        return f"HTTP Error {e.code}: {e.reason}\n{body_text}"
    except urllib.error.URLError as e:
        return f"URL Error: {e.reason}"
    except Exception as e:
        return f"Error: {e}"
