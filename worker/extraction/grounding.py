"""Grounding — Stage 4.4: Quote validation and semantic role checking.

Verifies that:
1. The source_quote actually exists in the document text
2. The value can be derived from the quote
3. The quote appears near semantically relevant keywords (NEW)
4. Ambiguous date formats are rejected
"""

from __future__ import annotations

import re
import logging
from datetime import datetime

from worker.models import ExtractedField, GroundingResult

logger = logging.getLogger(__name__)

# Semantic role keywords per field
SEMANTIC_KEYWORDS = {
    "due_date": ["due", "payable", "payment date", "pay by", "due date", "payment due"],
    "amount": ["total", "amount due", "total due", "grand total", "net payable", "total amount"],
    "tax": ["tax", "gst", "vat", "cgst", "sgst", "igst", "service tax"],
    "invoice_number": ["invoice", "inv", "bill", "receipt", "invoice no", "invoice number"],
    "currency": ["currency", "inr", "usd", "eur", "₹", "$", "€", "rs"],
}

# Context window for semantic check (characters around the quote)
SEMANTIC_CONTEXT_WINDOW = 150


def ground_fields(
    fields: list[ExtractedField],
    document_text: str,
) -> list[GroundingResult]:
    """Ground all extracted fields against the document text.

    Returns a GroundingResult per field.
    """
    results = []
    normalized_text = _normalize_text(document_text)

    for field in fields:
        result = _ground_single_field(field, document_text, normalized_text)
        results.append(result)
        if not result.passed:
            logger.warning(
                f"Grounding failed for '{field.field_name}': {result.reason}"
            )

    return results


def _ground_single_field(
    field: ExtractedField,
    document_text: str,
    normalized_text: str,
) -> GroundingResult:
    """Ground a single field."""

    result = GroundingResult(field_name=field.field_name, passed=False)

    # ── 1. Quote existence ───────────────────────────────────────────────
    quote = field.source_quote.strip()
    if not quote:
        result.reason = "Empty source_quote"
        return result

    normalized_quote = _normalize_text(quote)
    if normalized_quote not in normalized_text:
        # Try fuzzy: remove extra spaces
        compressed_quote = re.sub(r'\s+', ' ', normalized_quote)
        compressed_text = re.sub(r'\s+', ' ', normalized_text)
        if compressed_quote not in compressed_text:
            result.reason = f"Quote not found in document: '{quote[:60]}...'"
            return result

    result.quote_found = True

    # ── 2. Value derivation ──────────────────────────────────────────────
    value_str = str(field.value).strip()
    if not _value_derives_from_quote(value_str, quote, field.field_name):
        result.reason = f"Value '{value_str}' cannot be derived from quote '{quote[:60]}'"
        return result

    result.value_derived = True

    # ── 3. Ambiguous date check ──────────────────────────────────────────
    if field.field_name in ("due_date",):
        ambiguity = _check_date_ambiguity(quote, value_str)
        if ambiguity:
            result.reason = ambiguity
            return result

    # ── 4. Semantic role check (NEW) ─────────────────────────────────────
    keywords = SEMANTIC_KEYWORDS.get(field.field_name, [])
    if keywords:
        semantic_ok = _check_semantic_context(
            document_text, quote, keywords
        )
        if not semantic_ok:
            result.reason = (
                f"Quote for '{field.field_name}' not near expected keywords "
                f"({', '.join(keywords[:3])}). Possible wrong field assignment."
            )
            result.semantic_role_ok = False
            return result

    result.semantic_role_ok = True
    result.passed = True
    return result


def _normalize_text(text: str) -> str:
    """Normalize for comparison."""
    text = text.lower()
    text = re.sub(r'[\u00a0\u200b\u200c\u200d\ufeff]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def _value_derives_from_quote(value: str, quote: str, field_name: str) -> bool:
    """Check if the normalized value can be derived from the quote."""
    # Normalize value
    clean_value = value.strip().lower()
    clean_quote = quote.strip().lower()

    # Direct containment
    if clean_value in clean_quote:
        return True

    # Numeric: normalize both and compare
    if field_name in ("amount", "subtotal", "tax"):
        quote_nums = _extract_numbers(clean_quote)
        value_nums = _extract_numbers(clean_value)
        if quote_nums and value_nums:
            return any(
                any(abs(q - v) < 0.01 for q in quote_nums)
                for v in value_nums
            )
        return False

    # Date: various formats
    if field_name in ("due_date",):
        return _date_matches(clean_value, clean_quote)

    # Currency map (e.g. INR -> Rs.)
    if field_name == "currency":
        currency_map = {
            "inr": ["rs", "rupee", "inr", "₹"],
            "usd": ["usd", "$", "dollar"],
            "eur": ["eur", "€", "euro"]
        }
        symbols = currency_map.get(clean_value, [clean_value])
        return any(sym in clean_quote for sym in symbols)

    # String: fuzzy containment
    value_words = clean_value.split()
    return all(w in clean_quote for w in value_words)


def _extract_numbers(text: str) -> list[float]:
    """Extract all numeric values from text, handling currency symbols and commas."""
    matches = re.finditer(r'[\d,]+\.\d+|[\d,]+', text)
    nums = []
    for match in matches:
        clean = match.group(0).replace(',', '')
        try:
            nums.append(float(clean))
        except ValueError:
            pass
    return nums


def _date_matches(value: str, quote: str) -> bool:
    """Check if a date value matches the quote."""
    # Try parsing the value as ISO date
    try:
        dt = datetime.fromisoformat(value)
        # Check if date components appear in the quote
        day = str(dt.day)
        month_names = [
            dt.strftime("%B").lower(), dt.strftime("%b").lower(),
            str(dt.month).zfill(2), str(dt.month),
        ]
        year = str(dt.year)

        has_day = day in quote
        has_month = any(m in quote for m in month_names)
        has_year = year in quote

        return has_day and has_month and has_year
    except ValueError:
        pass
    return value in quote


def _check_date_ambiguity(quote: str, value: str) -> str | None:
    """Check for ambiguous date formats like 05/10/2026."""
    # Match patterns like DD/MM/YYYY or MM/DD/YYYY
    pattern = r'(\d{1,2})[/\-.](\d{1,2})[/\-.](\d{4})'
    match = re.search(pattern, quote)
    if match:
        first, second, year = int(match.group(1)), int(match.group(2)), int(match.group(3))
        # If both could be day or month, it's ambiguous
        if first <= 12 and second <= 12 and first != second:
            return (
                f"Ambiguous date format in quote: '{match.group(0)}'. "
                f"Could be DD/MM or MM/DD. Requires clarification."
            )
    return None


def _check_semantic_context(
    document_text: str,
    quote: str,
    keywords: list[str],
) -> bool:
    """Check if the quote appears near semantically relevant keywords.

    Searches within ±SEMANTIC_CONTEXT_WINDOW characters of the quote
    for any of the expected keywords.
    """
    text_lower = document_text.lower()
    quote_lower = quote.lower().strip()

    # Find all occurrences of the quote
    pos = 0
    while True:
        idx = text_lower.find(quote_lower, pos)
        if idx == -1:
            break

        # Extract context window
        start = max(0, idx - SEMANTIC_CONTEXT_WINDOW)
        end = min(len(text_lower), idx + len(quote_lower) + SEMANTIC_CONTEXT_WINDOW)
        context = text_lower[start:end]

        # Check for any keyword in context
        for kw in keywords:
            if kw.lower() in context:
                return True

        pos = idx + 1

    return False
