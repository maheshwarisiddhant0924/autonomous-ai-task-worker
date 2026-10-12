"""Playwright browser automation tools."""

from __future__ import annotations

import logging
from typing import Any

from playwright.async_api import async_playwright, Page, Browser, BrowserContext

from worker.config import ERP_BASE_URL, ERP_USERNAME, ERP_PASSWORD
from worker.models import ToolResult, ToolError

logger = logging.getLogger(__name__)

# Global singletons for browser sessions
_playwright = None
_browser: Browser | None = None
_context: BrowserContext | None = None
_page: Page | None = None


async def _get_page() -> Page:
    """Get or create the Playwright page singleton."""
    global _playwright, _browser, _context, _page

    if _page is not None and not _page.is_closed():
        return _page

    if _playwright is None:
        _playwright = await async_playwright().start()

    if _browser is None:
        _browser = await _playwright.chromium.launch(headless=True)

    if _context is None:
        _context = await _browser.new_context(
            viewport={"width": 1280, "height": 800},
            base_url=ERP_BASE_URL,
        )

    _page = await _context.new_page()
    return _page


async def close_browser():
    """Close down the Playwright session."""
    global _playwright, _browser, _context, _page
    if _page:
        await _page.close()
        _page = None
    if _context:
        await _context.close()
        _context = None
    if _browser:
        await _browser.close()
        _browser = None
    if _playwright:
        await _playwright.stop()
        _playwright = None


# ── Browser Tools ────────────────────────────────────────────────────────────

async def login() -> ToolResult:
    """Login to the ERP system using credentials from the environment."""
    page = await _get_page()
    try:
        await page.goto("/login")
        await page.fill("input[name='username']", ERP_USERNAME)
        await page.fill("input[name='password']", ERP_PASSWORD)
        await page.click("button[type='submit']")
        await page.wait_for_load_state("networkidle")

        if "/login" in page.url:
            return ToolResult(
                success=False,
                error=ToolError(
                    error_type="auth_failed",
                    message="Login failed (check credentials)",
                    tool_name="login",
                )
            )

        return ToolResult(success=True, message="Successfully logged in")
    except Exception as e:
        logger.error(f"Login failed: {e}")
        return ToolResult(
            success=False,
            error=ToolError(error_type="browser_error", message=str(e), tool_name="login")
        )


async def open_url(url: str) -> ToolResult:
    page = await _get_page()
    try:
        if "localhost" in url:
            url = url.replace("localhost", "127.0.0.1")
        await page.goto(url)
        await page.wait_for_load_state("networkidle")
        snapshot = await page.locator("body").aria_snapshot()
        return ToolResult(
            success=True,
            data={"snapshot": snapshot, "url": page.url},
            message=f"Navigated to {url}",
        )
    except Exception as e:
        return ToolResult(
            success=False,
            error=ToolError(error_type="browser_error", message=str(e), tool_name="open_url", retryable=True)
        )


async def read_page() -> ToolResult:
    page = await _get_page()
    try:
        snapshot = await page.locator("body").aria_snapshot()
        return ToolResult(
            success=True,
            data={"snapshot": snapshot, "url": page.url},
        )
    except Exception as e:
        return ToolResult(
            success=False,
            error=ToolError(error_type="browser_error", message=str(e), tool_name="read_page", retryable=True)
        )


async def click(element_role: str, element_name: str) -> ToolResult:
    """Click an element using ARIA role and name."""
    page = await _get_page()
    try:
        locator = page.get_by_role(element_role, name=element_name, exact=True)
        if await locator.count() == 0:
            # Try fuzzy match
            locator = page.get_by_role(element_role, name=element_name)

        if await locator.count() == 0:
            return ToolResult(
                success=False,
                error=ToolError(
                    error_type="StaleElementError",
                    message=f"Could not find {element_role} named '{element_name}'",
                    tool_name="click",
                    retryable=True,
                )
            )

        await locator.first.click()
        await page.wait_for_load_state("networkidle")
        snapshot = await page.locator("body").aria_snapshot()

        return ToolResult(
            success=True,
            data={"snapshot": snapshot, "url": page.url},
            message=f"Clicked {element_role} '{element_name}'",
        )
    except Exception as e:
        return ToolResult(
            success=False,
            error=ToolError(error_type="browser_error", message=str(e), tool_name="click", retryable=True)
        )
