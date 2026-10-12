"""Form filling and submission tools (ACI layer).

Implements the atomic write tool pattern:
snapshot → key pre-check → pre-submit guard → submit
"""

from __future__ import annotations

import logging
from typing import Any

from worker.models import ToolResult, ToolError, BoundState
from worker.tools.browser import _get_page
from worker.toolkits import Toolkit

logger = logging.getLogger(__name__)


async def fill_form(
    bound: BoundState,
    toolkit: Toolkit,
    form_id: str = "invoice_form"
) -> ToolResult:
    """Fill a form using accessible labels mapped from bound keys.

    Falls back to fuzzy matching if exact label fails.
    Returns FormMapMismatch error if mapping fails.
    """
    page = await _get_page()
    expected_map = toolkit.label_map(form_id)
    if not expected_map:
        return ToolResult(
            success=False,
            error=ToolError(error_type="FormMapMismatch", message=f"No label map for {form_id}", tool_name="fill_form")
        )

    # Resolve labels against live DOM
    snapshot = await page.locator("body").aria_snapshot()
    # (In a real implementation, we'd parse the aria_snapshot string to find available labels.
    # For this prototype, we'll extract label texts directly via JS)
    js_get_labels = """
    () => Array.from(document.querySelectorAll('label')).map(l => l.innerText.trim())
    """
    available_labels = await page.evaluate(js_get_labels)

    actual_map = {}
    for bound_key, expected_label in expected_map.items():
        if expected_label in available_labels:
            actual_map[bound_key] = expected_label
        else:
            # Try fuzzy fallback
            fuzzy_map = toolkit.fuzzy_label_match(form_id, available_labels)
            if fuzzy_map and bound_key in fuzzy_map:
                actual_map[bound_key] = fuzzy_map[bound_key]
            else:
                return ToolResult(
                    success=False,
                    error=ToolError(
                        "FormMapMismatch",
                        f"Expected label '{expected_label}' not found (tried exact & fuzzy)",
                        "fill_form"
                    )
                )

    # Actually fill the form
    filled_fields = []
    for bound_key, label_text in actual_map.items():
        value = bound.target_values.get(bound_key)
        if value is not None:
            try:
                # Playwright exact label matching
                locator = page.get_by_label(label_text, exact=True)
                if await locator.count() == 0:
                    locator = page.get_by_label(label_text)

                if await locator.count() > 0:
                    tag_name = await locator.first.evaluate("el => el.tagName.toLowerCase()")
                    if tag_name == "select":
                        await locator.first.select_option(str(value))
                    else:
                        await locator.first.fill(str(value))
                    filled_fields.append(bound_key)
            except Exception as e:
                logger.warning(f"Failed to fill field {label_text}: {e}")

    new_snapshot = await page.locator("body").aria_snapshot()
    screenshot_path = None
    try:
        from worker.config import AUDIT_LOG
        shots = AUDIT_LOG.parent / "screenshots"
        shots.mkdir(exist_ok=True, parents=True)
        bk = bound.target_values.get("invoice_number", "form")
        screenshot_path = str(shots / f"screenshot_before_{bk}.png")
        await page.screenshot(path=screenshot_path)
    except Exception as e:
        logger.warning(f"Before screenshot failed: {e}")

    return ToolResult(
        success=True,
        data={"snapshot": new_snapshot, "filled": filled_fields, "screenshot_before": screenshot_path},
        screenshot_path=screenshot_path,
        message=f"Filled {len(filled_fields)} fields",
    )


async def submit_form(
    bound: BoundState,
    toolkit: Toolkit,
    business_key_value: str,
    submit_button_name: str = "Create Invoice"
) -> ToolResult:
    """Atomic write tool: snapshot → key pre-check → pre-submit guard → submit."""
    page = await _get_page()

    # 1. Snapshot (done via toolkit interface later, but we trigger it here)
    baseline = toolkit.snapshot(business_key_value)

    # 2. Key pre-check (check if it already exists before clicking)
    ks = toolkit.key_state(business_key_value)
    if ks["status"] == "present":
        # Check if values match
        from worker.verification.verifier import _normalize_for_compare
        mismatch = False
        for k, v in bound.target_values.items():
            if _normalize_for_compare(ks["values"].get(k)) != _normalize_for_compare(v):
                mismatch = True
                break

        if not mismatch:
            # Already done exactly as we wanted, no need to click
            return ToolResult(
                success=True,
                data={"baseline": baseline.model_dump(), "outcome": "ALREADY_PRESENT"},
                message=f"Target record '{business_key_value}' already exists with matching values. Submit skipped (idempotent no-op).",
            )
        else:
            # Someone else created it with different values
            return ToolResult(
                success=False,
                error=ToolError(
                    "PREEXISTING_CONFLICT",
                    "Target key already exists with different values. Escalating.",
                    "submit_form"
                )
            )

    # 3. Pre-submit guard (read live DOM values and diff against bound)
    # The ERP UI formats amounts (e.g. 48500.00 might display differently), so we normalize
    from worker.verification.verifier import _normalize_for_compare
    expected_map = toolkit.label_map("invoice_form")
    guard_mismatch = False
    mismatch_details = []

    for bound_key, label_text in expected_map.items():
        expected_val = bound.target_values.get(bound_key)
        if expected_val is not None:
            locator = page.get_by_label(label_text)
            if await locator.count() > 0:
                actual_val = await locator.first.input_value()
                if _normalize_for_compare(actual_val) != _normalize_for_compare(expected_val):
                    guard_mismatch = True
                    mismatch_details.append(f"{bound_key}: expected {expected_val}, saw {actual_val}")

    if guard_mismatch:
        return ToolResult(
            success=False,
            error=ToolError(
                "pre_submit_mismatch",
                f"UI values do not match bound values: {', '.join(mismatch_details)}",
                "submit_form",
                retryable=True,
            )
        )

    # 4. Click submit
    try:
        # Check for our hook-simulated timeout (Fault injection Scenario 14)
        from worker.mock_erp.fault_injection import check_write_fault, SimulatedTimeoutError
        fault = check_write_fault()

        await page.get_by_role("button", name=submit_button_name).click()

        if fault == "timeout":
            raise SimulatedTimeoutError("Simulated network timeout during form submission")

        await page.wait_for_load_state("networkidle")

        screenshot_path = None
        try:
            from worker.config import AUDIT_LOG
            shots = AUDIT_LOG.parent / "screenshots"
            shots.mkdir(exist_ok=True, parents=True)
            screenshot_path = str(shots / f"screenshot_after_{business_key_value}.png")
            await page.screenshot(path=screenshot_path)
            # Also keep legacy filename for compatibility
            await page.screenshot(path=str(shots / f"screenshot_{business_key_value}.png"))
        except Exception as ss_err:
            logger.warning(f"Screenshot capture failed (non-fatal): {ss_err}")

        return ToolResult(
            success=True,
            data={"baseline": baseline.model_dump(), "url": page.url, "screenshot_after": screenshot_path},
            screenshot_path=screenshot_path,
            message="Form submitted successfully",
        )
    except SimulatedTimeoutError as e:
        return ToolResult(
            success=False,
            error=ToolError(error_type="write_timeout", message=str(e), tool_name="submit_form", retryable=True),
        )
    except Exception as e:
        return ToolResult(
            success=False,
            error=ToolError(error_type="browser_write_error", message=str(e), tool_name="submit_form", retryable=True)
        )
