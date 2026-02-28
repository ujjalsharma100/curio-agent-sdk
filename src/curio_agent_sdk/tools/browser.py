"""
Browser automation toolkit (Playwright-based): navigate, click, fill form, screenshot, get_text, execute_js.

Requires optional dependency: pip install curio-agent-sdk[browser]
Then: playwright install (one-time install of browser binaries).
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import Any

from curio_agent_sdk.core.tools.tool import Tool, tool


def _check_playwright() -> None:
    try:
        from playwright.async_api import async_playwright  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Browser tools require Playwright. Install with: pip install curio-agent-sdk[browser]\n"
            "Then run: playwright install"
        ) from e


class _PlaywrightHolder:
    """Holds Playwright browser/page; lazy init on first use."""

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._playwright = None
        self._browser = None
        self._page = None

    async def get_page(self):
        if self._page is not None:
            return self._page
        _check_playwright()
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        self._browser = await self._playwright.chromium.launch(headless=self.headless)
        self._page = await self._browser.new_page()
        return self._page

    async def close(self) -> None:
        if self._page:
            await self._page.close()
            self._page = None
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._playwright:
            await self._playwright.stop()
            self._playwright = None


def _make_browser_tools(holder: _PlaywrightHolder) -> list[Tool]:
    """Build browser tools that share the same Playwright holder."""

    @tool(
        name="browser_navigate",
        description="Navigate the browser to a URL.",
        timeout=30.0,
        require_confirmation=True,
    )
    async def navigate(url: str) -> str:
        """Navigate to the given URL."""
        page = await holder.get_page()
        resp = await page.goto(url, wait_until="domcontentloaded")
        status = resp.status if resp else "unknown"
        return f"Navigated to {url}. Status: {status}"

    @tool(
        name="browser_click_element",
        description="Click an element identified by selector (CSS or text). E.g. 'button', '#submit', 'text=Login'.",
        timeout=10.0,
        require_confirmation=True,
    )
    async def click_element(selector: str) -> str:
        """Click the element matching the selector."""
        page = await holder.get_page()
        await page.click(selector)
        return f"Clicked element: {selector}"

    @tool(
        name="browser_fill_form",
        description="Fill a form field. selector: CSS selector for the input; value: text to type.",
        timeout=10.0,
        require_confirmation=True,
    )
    async def fill_form(selector: str, value: str) -> str:
        """Fill the input matching selector with the given value."""
        page = await holder.get_page()
        await page.fill(selector, value)
        return f"Filled {selector} with {len(value)} characters."

    @tool(
        name="browser_screenshot",
        description="Take a screenshot of the current page. Returns file path or data URL.",
        timeout=15.0,
        require_confirmation=False,
    )
    async def screenshot(output_path: str = "", as_data_url: bool = False) -> str:
        """Take a screenshot of the current page."""
        page = await holder.get_page()
        if as_data_url:
            buf = await page.screenshot(type="png")
            b64 = base64.b64encode(buf).decode("ascii")
            return f"data:image/png;base64,{b64}"
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(path))
            return str(path.resolve())
        fd, path = tempfile.mkstemp(suffix=".png")
        try:
            await page.screenshot(path=path)
            return path
        finally:
            import os
            os.close(fd)

    @tool(
        name="browser_get_text",
        description="Get visible text from the page or from an element. selector: optional CSS selector; if omitted returns body text.",
        timeout=10.0,
        require_confirmation=False,
    )
    async def get_text(selector: str = "body") -> str:
        """Get text content of the page or the element matching selector."""
        page = await holder.get_page()
        el = page.locator(selector).first
        text = await el.inner_text()
        return text.strip() if text else ""

    @tool(
        name="browser_execute_js",
        description="Execute JavaScript in the page. Pass expression or statement; return value is JSON-serialized.",
        timeout=10.0,
        require_confirmation=True,
    )
    async def execute_js(script: str) -> str:
        """Execute JavaScript in the page context. Returns JSON string of the result."""
        page = await holder.get_page()
        result: Any = await page.evaluate(script)
        import json
        try:
            return json.dumps(result, default=str)
        except (TypeError, ValueError):
            return str(result)

    @tool(
        name="browser_get_url",
        description="Get the current page URL.",
        timeout=2.0,
        require_confirmation=False,
    )
    async def get_url() -> str:
        """Return the current page URL."""
        page = await holder.get_page()
        return page.url

    return [
        navigate,
        click_element,
        fill_form,
        screenshot,
        get_text,
        execute_js,
        get_url,
    ]


class BrowserToolkit:
    """
    Playwright-based browser automation tools: navigate, click, fill form, screenshot, get_text, execute_js.

    Use with: Agent.builder().tools(BrowserToolkit(headless=True).get_tools()).build()

    Requires: pip install curio-agent-sdk[browser]
    Then: playwright install

    For long-lived agents, call toolkit.close() when done to release the browser (or use async with toolkit).
    """

    def __init__(self, headless: bool = True):
        self.headless = headless
        self._holder = _PlaywrightHolder(headless=headless)

    def get_tools(self) -> list[Tool]:
        """Return all browser tools. Browser is started on first tool use."""
        return _make_browser_tools(self._holder)

    async def close(self) -> None:
        """Close the browser and release resources."""
        await self._holder.close()

    async def __aenter__(self) -> "BrowserToolkit":
        return self

    async def __aexit__(self, *args: Any) -> None:
        await self.close()
