"""Verifier — Stage 9: Independent database verification.

Deterministic code reading through an independent channel (direct SQLite).
Never asks the LLM whether it succeeded.

Key design from CRITIC paper: models can't reliably self-verify without
external tools, so verification is always code-based.

Scoped verification (not count-based) — fixes the row_count == baseline + 1 bug.
"""

from __future__ import annotations

import logging
from worker.models import (
    BoundState, Baseline, VerifyResult, VerifyDiff, VerifyOutcome, RowSnapshot,
)
from worker.mock_erp.database import snapshot_table, key_state

logger = logging.getLogger(__name__)


def verify(
    bound: BoundState,
    baselines: list[Baseline],
    business_key_field: str,
    business_key_value: str,
    db_path=None,
) -> VerifyResult:
    """Verify the outcome against the database.

    Steps:
    1. Target row exists with correct values
    2. In-scope rows unchanged vs baseline
    3. Determine outcome from first baseline
    """
    if not baselines:
        return VerifyResult(
            passed=False,
            outcome=VerifyOutcome.KEY_MISSING,
            message="No baseline available — cannot verify",
        )

    # ── Check First Baseline for Pre-existing Conflict ───────────────────
    first_baseline = baselines[0]
    first_ks = None
    for rs in first_baseline.table_row_hashes:
        if rs.business_key == business_key_value:
            first_ks = rs
            break

    if first_ks is not None:
        first_values = first_ks.field_values
        first_match = True
        for field_name, expected in bound.target_values.items():
            if _normalize_for_compare(first_values.get(field_name)) != _normalize_for_compare(expected):
                first_match = False
                break
        if not first_match:
            return VerifyResult(
                passed=False,
                outcome=VerifyOutcome.PREEXISTING_CONFLICT,
                message="Key existed before this run with different values — escalate to human",
            )

    # ── Step 1: Check target row in live database ────────────────────────
    ks = key_state(business_key_field, business_key_value, db_path)

    if ks["status"] == "absent":
        return VerifyResult(
            passed=False,
            outcome=VerifyOutcome.KEY_MISSING,
            message=f"Target key '{business_key_value}' not found in database",
        )

    # Compare values
    actual_values = ks["values"]
    diffs: list[VerifyDiff] = []

    for field_name, expected in bound.target_values.items():
        actual = actual_values.get(field_name)
        # Normalize for comparison
        expected_norm = _normalize_for_compare(expected)
        actual_norm = _normalize_for_compare(actual)
        if expected_norm != actual_norm:
            diffs.append(VerifyDiff(
                field=field_name,
                expected=expected,
                actual=actual,
            ))

    if diffs:
        return VerifyResult(
            passed=False,
            outcome=VerifyOutcome.WRONG_VALUES,
            diffs=diffs,
            message=f"Target row exists but {len(diffs)} field(s) don't match",
        )

    # ── Step 2: Check in-scope rows unchanged ────────────────────────────
    last_baseline = baselines[-1]
    current = snapshot_table(business_key_field, business_key_value, db_path)
    warnings: list[str] = []

    baseline_hashes = {
        rs.business_key: rs.content_hash
        for rs in last_baseline.table_row_hashes
    }
    current_hashes = {
        rs.business_key: rs.content_hash
        for rs in current.table_row_hashes
    }

    target_vendor = bound.target_values.get("vendor") or bound.target_values.get("company")

    # Check in-scope rows (sharing key or vendor) are unchanged
    for rs in last_baseline.table_row_hashes:
        bk = rs.business_key
        if bk == business_key_value:
            continue  # skip target row — already verified
        bhash = rs.content_hash
        if bk in current_hashes and current_hashes[bk] != bhash:
            row_vendor = rs.field_values.get("vendor") or rs.field_values.get("company")
            is_in_scope = bool(target_vendor and row_vendor and str(target_vendor).strip().lower() == str(row_vendor).strip().lower())
            if is_in_scope:
                return VerifyResult(
                    passed=False,
                    outcome=VerifyOutcome.WRONG_VALUES,
                    message=f"In-scope row '{bk}' (vendor: {row_vendor}) changed since baseline",
                )
            else:
                warnings.append(f"Unrelated row '{bk}' changed since baseline (concurrent activity)")

    # Check for new rows (concurrent activity) — tolerated but warned
    new_keys = set(current_hashes.keys()) - set(baseline_hashes.keys())
    new_others = new_keys - {business_key_value}
    if new_others:
        warnings.append(
            f"New rows appeared since baseline (concurrent activity): "
            f"{', '.join(sorted(new_others))}"
        )

    # ── Step 3: Determine outcome ────────────────────────────────────────
    if first_ks is None:
        outcome = VerifyOutcome.CREATED
    else:
        outcome = VerifyOutcome.ALREADY_PRESENT

    return VerifyResult(
        passed=True,
        outcome=outcome,
        warnings=warnings,
        message=f"Verification passed: {outcome.value}",
    )


def _normalize_for_compare(value) -> str:
    """Normalize a value for comparison."""
    if value is None:
        return ""
    s = str(value).strip()
    # Normalize numeric values
    try:
        f = float(s.replace(",", "").replace("Rs.", "").replace("₹", "").strip())
        return f"{f:.2f}"
    except (ValueError, TypeError):
        pass
    return s.lower()
