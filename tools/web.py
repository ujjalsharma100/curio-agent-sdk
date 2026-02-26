"""Built-in web tools."""

import re
import urllib.request
import urllib.error

from curio_agent_sdk.core.tools.tool import tool


@tool(name="web_fetch", description="Fetch a web page and return its text content", timeout=30.0)
def web_fetch(url: str) -> str:
    """Fetch a web page and return readable text content (HTML tags stripped)."""
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "CurioAgent/1.0"})
        with urllib.request.urlopen(req, timeout=25) as resp:
            html = resp.read().decode("utf-8", errors="replace")

        # Strip HTML tags
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", " ", text)
        text = re.sub(r"\s+", " ", text).strip()

        if len(text) > 10000:
            text = text[:10000] + "\n... [truncated]"
        return text
    except urllib.error.URLError as e:
        return f"Error fetching {url}: {e}"
    except Exception as e:
        return f"Error: {e}"
