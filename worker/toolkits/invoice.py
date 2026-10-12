"""Invoice Toolkit — domain-specific logic for invoice processing.

Implements the Toolkit protocol for the invoice use case:
- Discovers PDFs in a sandbox folder
- Extracts via LLM with grounding
- Fills forms in the mock ERP via browser
- Verifies via direct SQLite
"""

from __future__ import annotations

import logging
from typing import Any

from worker.models import (
    StaticContract, ValidationRule, DiscoveryResult, ExtractionResult,
    BoundState, BindFailure, Baseline, VerifyResult, ToolSpec,
    PermissionTier, GroundingResult,
)
from worker.discovery import discover as discover_files
from worker.extraction.converter import convert_pdf
from worker.extraction.extractor import extract_fields
from worker.extraction.grounding import ground_fields
from worker.extraction.checker import independent_check as run_independent_check
from worker.extraction.binder import bind as run_bind
from worker.verification.verifier import verify as run_verify
from worker.mock_erp.database import (
    snapshot_table, key_state as db_key_state,
)

logger = logging.getLogger(__name__)


class InvoiceToolkit:
    """Invoice processing toolkit."""

    name = "invoice"
    description = (
        "Process invoices: find a PDF invoice, extract amount and due date, "
        "enter it into the internal ERP system, and verify the entry."
    )
    required_params = ["company"]

    # Label mapping: bound key → accessible label on the ERP form
    _label_maps = {
        "invoice_form": {
            "invoice_number": "Invoice Number",
            "company": "Company",
            "amount": "Total Amount",
            "currency": "Currency",
            "due_date": "Due Date",
            "subtotal": "Subtotal",
            "tax": "Tax",
            "notes": "Notes",
        }
    }

    tools = {
        "login": ToolSpec(
            name="login", description="Login to ERP system",
            permission_tier=PermissionTier.READ,
        ),
        "open_url": ToolSpec(
            name="open_url", description="Navigate to a URL",
            permission_tier=PermissionTier.READ,
        ),
        "read_page": ToolSpec(
            name="read_page", description="Read current page content",
            permission_tier=PermissionTier.READ,
        ),
        "click": ToolSpec(
            name="click", description="Click an element by ref",
            permission_tier=PermissionTier.READ,
        ),
        "fill_form": ToolSpec(
            name="fill_form", description="Fill form with bound values",
            permission_tier=PermissionTier.IDEMPOTENT_WRITE,
        ),
        "submit_form": ToolSpec(
            name="submit_form", description="Submit form (atomic write)",
            permission_tier=PermissionTier.IDEMPOTENT_WRITE,
        ),
        "update_record": ToolSpec(
            name="update_record", description="Edit existing record",
            permission_tier=PermissionTier.IDEMPOTENT_WRITE,
        ),
        "submit_for_verification": ToolSpec(
            name="submit_for_verification", description="Verify the outcome",
            permission_tier=PermissionTier.READ,
        ),
        "ask_user": ToolSpec(
            name="ask_user", description="Ask user for clarification",
            permission_tier=PermissionTier.READ,
        ),
    }

    def __init__(self, config: dict | None = None):
        self.config = config or {
            "currency": "INR",
            "amount_threshold": 100000.0,
        }

    def static_contract(self, params: dict) -> StaticContract:
        return StaticContract(
            target_system="mock_erp",
            business_key_field="invoice_number",
            required_fields=["invoice_number", "company", "amount", "currency", "due_date"],
            optional_fields=["subtotal", "tax", "notes"],
            rules=[
                ValidationRule(name="positive"),
                ValidationRule(name="iso_date"),
                ValidationRule(name="currency_is", params={"expected": self.config["currency"]}),
                ValidationRule(name="sum_equals"),
            ],
        )

    def discover(self, params: dict) -> DiscoveryResult:
        return discover_files(params)

    def convert(self, source_path: str) -> dict:
        return convert_pdf(source_path)

    def extract(self, document_text: str) -> ExtractionResult:
        return extract_fields(document_text, field_schema="invoice")

    def ground(self, extraction: ExtractionResult, document_text: str) -> list[GroundingResult]:
        return ground_fields(extraction.fields, document_text)

    def independent_check(self, extraction: ExtractionResult, document_text: str) -> dict:
        return run_independent_check(extraction, document_text, is_scanned=extraction.ocr_used)

    def bind(self, extraction: ExtractionResult, grounding: list[GroundingResult],
             contract: StaticContract) -> BoundState | BindFailure:
        return run_bind(extraction, grounding, contract)

    def label_map(self, form_id: str) -> dict[str, str]:
        return self._label_maps.get(form_id, {})

    def fuzzy_label_match(self, form_id: str, available_labels: list[str]) -> dict[str, str] | None:
        """Fuzzy match bound keys to available labels (Levenshtein ≤ 2)."""
        expected_map = self._label_maps.get(form_id, {})
        if not expected_map:
            return None

        result = {}
        for bound_key, expected_label in expected_map.items():
            best_match = None
            best_distance = float("inf")
            for available in available_labels:
                dist = _levenshtein(expected_label.lower(), available.lower())
                if dist < best_distance:
                    best_distance = dist
                    best_match = available
            if best_distance <= 2 and best_match:
                result[bound_key] = best_match
                logger.info(
                    f"Fuzzy label match: '{expected_label}' → '{best_match}' (dist={best_distance})"
                )
            else:
                logger.warning(f"No fuzzy match for '{expected_label}' (best: '{best_match}', dist={best_distance})")
                return None  # fail if any key can't be matched

        return result

    def snapshot(self, business_key: str) -> Baseline:
        return snapshot_table("invoice_number", business_key)

    def key_state(self, business_key: str) -> dict:
        return db_key_state("invoice_number", business_key)

    def verify(self, bound: BoundState, baselines: list[Baseline]) -> VerifyResult:
        bk_value = bound.target_values.get("invoice_number", "")
        return run_verify(bound, baselines, "invoice_number", bk_value)

    def recovery_for(self, error_or_diff: dict) -> dict | None:
        """Look up deterministic recovery. None → LLM reflection."""
        error_type = error_or_diff.get("error_type", "")

        recovery_table = {
            "StaleElementError": {"action": "re_snapshot", "max_retries": 2},
            "read_timeout": {"action": "backoff_retry", "max_retries": 3},
            "write_timeout": {"action": "key_check_then_decide"},
            "pre_submit_mismatch": {"action": "refill_from_bound"},
            "http_409": {"action": "key_check_then_verify"},
            "key_missing": {"action": "resubmit", "max_retries": 2},
            "wrong_values_this_run": {"action": "update_record", "max_retries": 1},
            "browser_session_lost": {"action": "reset_session"},
            "maintenance_page": {"action": "wait_and_retry", "wait_seconds": 5, "max_retries": 2},
            "FormMapMismatch": None,  # → human gate
            "PREEXISTING_CONFLICT": None,  # → human gate
            "duplicate_rows": None,  # → human gate
        }

        return recovery_table.get(error_type)

    def reset_view(self) -> None:
        """Reset navigation to a known state (handled by browser tools)."""
        pass


def _levenshtein(s1: str, s2: str) -> int:
    """Simple Levenshtein distance."""
    if len(s1) < len(s2):
        return _levenshtein(s2, s1)
    if len(s2) == 0:
        return len(s1)
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]
