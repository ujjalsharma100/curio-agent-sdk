"""
Computer use toolkit: screenshot, click, type, scroll, mouse, keyboard.

Requires optional dependency: pip install curio-agent-sdk[computer-use]
(installs pyautogui for cross-platform desktop automation).
"""

from __future__ import annotations

import base64
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from curio_agent_sdk.core.tools.tool import Tool, tool

if TYPE_CHECKING:
    pass


def _check_computer_use_deps() -> None:
    """Raise ImportError with install hint if optional deps are missing."""
    try:
        import pyautogui  # noqa: F401
    except ImportError as e:
        raise ImportError(
            "Computer use tools require optional dependencies. "
            "Install with: pip install curio-agent-sdk[computer-use]"
        ) from e


def _screenshot_to_path_or_data_url(
    region: tuple[int, int, int, int] | None = None,
    output_path: str | None = None,
    as_data_url: bool = False,
) -> str:
    """Take screenshot; return file path or data URL. Requires pyautogui."""
    _check_computer_use_deps()
    import pyautogui

    if region:
        im = pyautogui.screenshot(region=region)
    else:
        im = pyautogui.screenshot()
    if as_data_url:
        import io
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("ascii")
        return f"data:image/png;base64,{b64}"
    if output_path:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        im.save(str(path))
        return str(path.resolve())
    fd, path = tempfile.mkstemp(suffix=".png")
    try:
        im.save(path)
        return path
    finally:
        import os
        os.close(fd)


@tool(
    name="screenshot",
    description="Capture a screenshot of the screen or a region. Returns file path or data URL for the image.",
    timeout=15.0,
    require_confirmation=True,
)
def screenshot(
    output_path: str = "",
    as_data_url: bool = False,
    left: int | None = None,
    top: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> str:
    """
    Take a screenshot. Optionally save to output_path or return as data URL.
    If left, top, width, height are given, capture only that region (pixel coordinates).
    """
    region = None
    if left is not None and top is not None and width is not None and height is not None:
        region = (left, top, width, height)  # pyautogui uses (left, top, width, height)
    path = output_path.strip() or None
    return _screenshot_to_path_or_data_url(region=region, output_path=path, as_data_url=as_data_url)


@tool(
    name="click",
    description="Click at the given screen coordinates (x, y in pixels). Optionally specify button and clicks.",
    timeout=5.0,
    require_confirmation=True,
)
def click(x: int, y: int, button: str = "left", clicks: int = 1) -> str:
    """Click at pixel position (x, y). button: 'left', 'right', or 'middle'. clicks: number of clicks (e.g. 2 for double-click)."""
    _check_computer_use_deps()
    import pyautogui
    pyautogui.click(x=x, y=y, button=button, clicks=clicks)
    return f"Clicked at ({x}, {y}) with {button} button, {clicks} click(s)."


@tool(
    name="type_text",
    description="Type a string of text using the keyboard. Use for typing into focused input fields.",
    timeout=30.0,
    require_confirmation=True,
)
def type_text(text: str, interval: float = 0.0) -> str:
    """Type the given text. interval: delay in seconds between each key (0 for no delay)."""
    _check_computer_use_deps()
    import pyautogui
    pyautogui.write(text, interval=interval)
    return f"Typed {len(text)} characters."


@tool(
    name="scroll",
    description="Scroll the mouse wheel. Positive clicks = scroll up/forward, negative = down/back.",
    timeout=2.0,
    require_confirmation=True,
)
def scroll(clicks: int, x: int | None = None, y: int | None = None) -> str:
    """Scroll by clicks. Optionally move to (x, y) before scrolling. clicks: positive up, negative down."""
    _check_computer_use_deps()
    import pyautogui
    if x is not None and y is not None:
        pyautogui.moveTo(x, y)
    pyautogui.scroll(clicks)
    return f"Scrolled {clicks} clicks."


@tool(
    name="move_mouse",
    description="Move the mouse cursor to the given screen coordinates (x, y in pixels).",
    timeout=2.0,
    require_confirmation=False,
)
def move_mouse(x: int, y: int, duration: float = 0.0) -> str:
    """Move mouse to (x, y). duration: seconds for animated move (0 for instant)."""
    _check_computer_use_deps()
    import pyautogui
    pyautogui.moveTo(x, y, duration=duration)
    return f"Moved mouse to ({x}, {y})."


@tool(
    name="key_press",
    description="Press a key or key combination (e.g. 'enter', 'tab', 'ctrl+c').",
    timeout=2.0,
    require_confirmation=True,
)
def key_press(key: str, presses: int = 1, interval: float = 0.0) -> str:
    """Press a key. key: e.g. 'enter', 'tab', 'ctrl', 'alt', 'shift', or combo like 'ctrl+c'. presses: number of times."""
    _check_computer_use_deps()
    import pyautogui
    pyautogui.press(key, presses=presses, interval=interval)
    return f"Pressed '{key}' {presses} time(s)."


@tool(
    name="hotkey",
    description="Press a key combination (e.g. 'ctrl', 'c' for copy). Pass keys as separate arguments.",
    timeout=2.0,
    require_confirmation=True,
)
def hotkey(*keys: str) -> str:
    """Press a key combination. E.g. hotkey('ctrl', 'c') for copy."""
    _check_computer_use_deps()
    import pyautogui
    pyautogui.hotkey(*keys)
    return f"Pressed combination: {'+'.join(keys)}."


@tool(
    name="get_cursor_position",
    description="Get the current mouse cursor position in screen coordinates (x, y).",
    timeout=2.0,
    require_confirmation=False,
)
def get_cursor_position() -> str:
    """Return current (x, y) of the mouse cursor."""
    _check_computer_use_deps()
    import pyautogui
    x, y = pyautogui.position()
    return f"x={x}, y={y}"


@tool(
    name="get_screen_size",
    description="Get the screen size in pixels (width, height). Use for coordinate mapping.",
    timeout=2.0,
    require_confirmation=False,
)
def get_screen_size() -> str:
    """Return screen width and height in pixels."""
    _check_computer_use_deps()
    import pyautogui
    w, h = pyautogui.size()
    return f"width={w}, height={h}"


class ComputerUseToolkit:
    """
    Tools for controlling the computer: screenshot, click, type, scroll, mouse, keyboard.

    Use with: Agent.builder().tools(ComputerUseToolkit().get_tools()).build()

    Requires: pip install curio-agent-sdk[computer-use]
    """

    def get_tools(self) -> list[Tool]:
        """Return all computer use tools."""
        return [
            screenshot,
            click,
            type_text,
            scroll,
            move_mouse,
            key_press,
            hotkey,
            get_cursor_position,
            get_screen_size,
        ]
