"""Binder — Stage 4.7: Bind extraction results to a validated contract.

Deterministic code that:
1. Checks all required fields are present
2. Runs static contract validation rules
3. Reconciles (subtotal + tax == total)
4. Produces BoundState with checksum
"""

from __future__ import annotations

import logging
import re
from datetime import datetime

from worker.models import (
    ExtractedField, ExtractionResult, BoundState, BindFailure,
    StaticContract, BoundSource, GroundingResult,
)

logger = logging.getLogger(__name__)


# ── Rule registry ────────────────────────────────────────────────────────────

def _rule_positive(value, **kwargs) -> tuple[bool, str]:
    try:
        return float(value) > 0, "Value must be positive"
    except (ValueError, TypeError):
        return False, f"Cannot parse as number: {value}"


def _rule_iso_date(value, **kwargs) -> tuple[bool, str]:
    try:
        datetime.fromisoformat(str(value))
        return True, ""
    except ValueError:
        return False, f"Not a valid ISO date: {value}"


def _rule_currency_is(value, expected=None, **kwargs) -> tuple[bool, str]:
    if expected and str(value).upper() != str(expected).upper():
        return False, f"Expected currency {expected}, got {value}"
    return True, ""


def _rule_sum_equals(
    target_values: dict, **kwargs
) -> tuple[bool, str]:
    subtotal = target_values.get("subtotal")
    tax = target_values.get("tax")
    amount = target_values.get("amount")

    if subtotal is None or tax is None or amount is None:
        return True, ""  # Can't check if fields missing

    try:
        s = float(subtotal)
        t = float(tax)
        a = float(amount)
        if abs((s + t) - a) > 0.01:
            return False, f"subtotal ({s}) + tax ({t}) = {s+t}, but amount = {a}"
        return True, ""
    except (ValueError, TypeError):
        return False, "Cannot parse numeric values for sum check"


RULE_REGISTRY = {
    "positive": _rule_positive,
    "iso_date": _rule_iso_date,
    "currency_is": _rule_currency_is,
    "sum_equals": _rule_sum_equals,
}


# ── Binder ───────────────────────────────────────────────────────────────────

def bind(
    extraction: ExtractionResult,
    grounding_results: list[GroundingResult],
    contract: StaticContract,
) -> BoundState | BindFailure:
    """Bind extraction results to a validated contract.

    1. Check all required fields are present and grounded
    2. Normalize values
    3. Run validation rules
    4. Produce BoundState with checksum
    """
    # ── 1. Check required fields ─────────────────────────────────────────
    field_map = {f.field_name: f for f in extraction.fields}
    grounding_map = {g.field_name: g for g in grounding_results}

    missing_fields = []
    failed_fields = []

    for req_field in contract.required_fields:
        if req_field not in field_map:
            missing_fields.append(req_field)
            continue
        # Check grounding passed
        gr = grounding_map.get(req_field)
        if gr and not gr.passed:
            failed_fields.append(f"{req_field}: grounding failed — {gr.reason}")

    if missing_fields:
        return BindFailure(
            reason=f"Missing required fields: {', '.join(missing_fields)}",
            failed_fields=missing_fields,
        )

    if failed_fields:
        return BindFailure(
            reason=f"Grounding failures: {'; '.join(failed_fields)}",
            failed_fields=[f.split(":")[0] for f in failed_fields],
        )

    # ── 2. Normalize values ──────────────────────────────────────────────
    target_values = {}
    source_quotes = {}
    ocr_confidences = {}

    for field in extraction.fields:
        normalized = _normalize_value(field.value, field.field_name)
        target_values[field.field_name] = normalized
        source_quotes[field.field_name] = field.source_quote
        if extraction.ocr_used:
            ocr_confidences[field.field_name] = field.field_confidence

    # ── 3. Run validation rules ──────────────────────────────────────────
    failed_rules = []

    for rule in contract.rules:
        rule_fn = RULE_REGISTRY.get(rule.name)
        if not rule_fn:
            logger.warning(f"Unknown rule: {rule.name}")
            continue

        if rule.name == "sum_equals":
            passed, msg = rule_fn(target_values=target_values)
        elif rule.name == "currency_is":
            currency_val = target_values.get("currency", "")
            passed, msg = rule_fn(currency_val, **rule.params)
        else:
            # Apply rule to relevant fields
            for field_name, value in target_values.items():
                if _rule_applies(rule.name, field_name):
                    passed, msg = rule_fn(value, **rule.params)
                    if not passed:
                        failed_rules.append(f"{rule.name}({field_name}): {msg}")

    if failed_rules:
        return BindFailure(
            reason=f"Validation failures: {'; '.join(failed_rules)}",
            failed_rules=failed_rules,
        )

    # ── 4. Create BoundState ─────────────────────────────────────────────
    bound = BoundState(
        target_values=target_values,
        source=BoundSource.EXTRACTED,
        source_quotes=source_quotes,
        ocr_confidences=ocr_confidences,
    )
    bound.compute_checksum()

    logger.info(f"Binding successful: {len(target_values)} fields, checksum={bound.checksum}")
    return bound


def _normalize_value(value, field_name: str):
    """Normalize a value based on field type."""
    if value is None:
        return None

    value_str = str(value).strip()

    if field_name in ("amount", "subtotal", "tax"):
        # Numeric normalization: Rs. 48,500 → 48500.00
        cleaned = re.sub(r'[₹$€£,\s]', '', value_str)
        cleaned = re.sub(r'^(rs\.?|inr|usd|eur)', '', cleaned, flags=re.IGNORECASE)
        try:
            return round(float(cleaned.strip()), 2)
        except ValueError:
            return value_str

    if field_name in ("due_date",):
        # Date normalization: "05 Oct 2026" → "2026-10-05"
        for fmt in ("%d %b %Y", "%d %B %Y", "%B %d, %Y", "%d/%m/%Y", "%d-%m-%Y", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(value_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
        return value_str

    if field_name == "currency":
        return value_str.upper()

    return value_str


def _rule_applies(rule_name: str, field_name: str) -> bool:
    """Check if a rule applies to a field."""
    mapping = {
        "positive": {"amount", "subtotal", "tax"},
        "iso_date": {"due_date"},
    }
    applicable = mapping.get(rule_name, set())
    return field_name in applicable
