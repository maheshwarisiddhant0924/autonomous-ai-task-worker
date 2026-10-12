"""Checker — Stage 4.5: Independent extraction check.

For text-layer PDFs: a second extraction with alternate prompt, normalized comparison.
For scanned docs: would use vision model (not implemented in prototype).
"""

from __future__ import annotations

import logging
from worker.models import ExtractedField, ExtractionResult
from worker.extraction.extractor import extract_fields

logger = logging.getLogger(__name__)


def independent_check(
    primary_result: ExtractionResult,
    document_text: str,
    is_scanned: bool = False,
) -> dict:
    """Run independent extraction and compare.

    Returns:
        {
            "passed": bool,
            "mismatches": [{"field": str, "primary": any, "secondary": any}],
            "message": str,
        }
    """
    if is_scanned:
        # For scanned docs, ideally use vision model on rendered image.
        # Prototype limitation: we use the same OCR text with alternate prompt.
        logger.warning(
            "Scanned document: independent check uses same OCR text "
            "(not fully independent — documented limitation)"
        )

    # Second extraction with alternate prompt
    secondary = extract_fields(
        document_text,
        field_schema="invoice",
        alternate_prompt=True,
    )

    # Compare normalized values
    primary_map = {f.field_name: f.value for f in primary_result.fields}
    secondary_map = {f.field_name: f.value for f in secondary.fields}

    mismatches = []
    for field_name, primary_val in primary_map.items():
        if field_name in ("subtotal", "tax"):
            continue  # optional fields, skip for independent check
        secondary_val = secondary_map.get(field_name)
        if secondary_val is None:
            mismatches.append({
                "field": field_name,
                "primary": primary_val,
                "secondary": None,
                "reason": "Missing in secondary extraction",
            })
            continue

        if not _values_match(primary_val, secondary_val, field_name):
            mismatches.append({
                "field": field_name,
                "primary": primary_val,
                "secondary": secondary_val,
                "reason": "Values don't match after normalization",
            })

    passed = len(mismatches) == 0

    if not passed:
        logger.warning(
            f"Independent check failed: {len(mismatches)} mismatch(es): "
            f"{[m['field'] for m in mismatches]}"
        )

    return {
        "passed": passed,
        "mismatches": mismatches,
        "message": "All fields match" if passed else f"{len(mismatches)} field(s) mismatch",
    }


def _values_match(v1, v2, field_name: str) -> bool:
    """Normalize and compare two values."""
    s1 = _normalize_value(str(v1), field_name)
    s2 = _normalize_value(str(v2), field_name)
    return s1 == s2


def _normalize_value(value: str, field_name: str) -> str:
    """Normalize a value for comparison."""
    import re

    value = value.strip()

    if field_name in ("amount", "subtotal", "tax"):
        # Numeric normalization
        cleaned = re.sub(r'[₹$€£,\s]', '', value)
        cleaned = re.sub(r'^(rs\.?|inr|usd|eur)', '', cleaned, flags=re.IGNORECASE)
        try:
            return f"{float(cleaned.strip()):.2f}"
        except ValueError:
            return value.lower()

    if field_name in ("due_date",):
        # Try ISO normalization
        from datetime import datetime
        for fmt in ("%Y-%m-%d", "%d/%m/%Y", "%d-%m-%Y", "%d %b %Y", "%d %B %Y", "%B %d, %Y"):
            try:
                dt = datetime.strptime(value, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return value.lower()

    return value.lower().strip()
