"""Trade Toolkit — domain implementation for processing trade orders and executions.

Demonstrates generalization of the autonomous AI worker pipeline beyond invoices.
Follows the same Toolkit protocol: discovery, parsing, binding, execution, and verification.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from worker.config import DATA_DIR
from worker.mock_erp.database import snapshot_table, key_state
from worker.models import (
    Baseline,
    BoundSource,
    BoundState,
    DiscoveryResult,
    ExtractedField,
    ExtractionResult,
    GroundingResult,
    SourceRef,
    StaticContract,
    ValidationRule,
    VerifyResult,
    ToolSpec,
    PermissionTier,
)
from worker.verification.verifier import verify

logger = logging.getLogger(__name__)


class TradeToolkit:
    """Toolkit for trade order processing and ledger verification."""

    name: str = "trade"
    description: str = "Process trade execution orders (stocks/bonds): parse orders, validate math, enter into trade ledger, and assert in database."
    required_params: list[str] = ["symbol"]
    submit_button_name: str = "Book Trade"

    tools: dict[str, ToolSpec] = {
        "fill_form": ToolSpec(
            name="fill_form",
            description="Fill trade form fields using extracted bound values",
            permission_tier=PermissionTier.IDEMPOTENT_WRITE,
        ),
        "submit_form": ToolSpec(
            name="submit_form",
            description="Submit trade execution order to ledger",
            permission_tier=PermissionTier.IDEMPOTENT_WRITE,
        ),
        "submit_for_verification": ToolSpec(
            name="submit_for_verification",
            description="Verify trade execution against SQLite database",
            permission_tier=PermissionTier.READ,
        ),
        "ask_user": ToolSpec(
            name="ask_user",
            description="Request clarification or approval from user",
            permission_tier=PermissionTier.READ,
        ),
    }

    def static_contract(self, params: dict) -> StaticContract:
        return StaticContract(
            target_system="mock_erp",
            business_key_field="trade_id",
            required_fields=["trade_id", "symbol", "direction", "quantity", "price"],
            optional_fields=["trade_date", "notes"],
            rules=[
                ValidationRule(name="positive", params={"fields": ["quantity", "price"]}),
                ValidationRule(name="allowed_direction", params={"directions": ["buy", "sell"]}),
            ],
        )

    def discover(self, params: dict) -> DiscoveryResult:
        """Find trade confirmation files (.json or .txt) matching symbol in data/trades/ or data/."""
        symbol = params.get("symbol", "").upper().strip()
        search_dirs = [
            DATA_DIR / "trades",
            DATA_DIR,
            Path("data/trades"),
            Path("data"),
        ]

        files: list[Path] = []
        for d in search_dirs:
            if d.exists():
                files.extend(d.glob("*.json"))
                files.extend(d.glob("*.txt"))

        candidates = []
        for f in set(files):
            try:
                content = f.read_text(encoding="utf-8", errors="ignore")
                match_conf = 0.0
                if symbol and symbol in content.upper():
                    match_conf = 1.0
                elif not symbol:
                    match_conf = 0.8

                if match_conf > 0.0:
                    candidates.append(
                        SourceRef(
                            path=str(f),
                            company_match_confidence=match_conf,
                            ordering_basis="file_mtime",
                            ordering_value=str(f.stat().st_mtime),
                            metadata={"filename": f.name, "symbol": symbol},
                        )
                    )
            except Exception as e:
                logger.warning(f"Error inspecting trade file {f}: {e}")

        # Sort newest first
        candidates.sort(key=lambda c: float(c.ordering_value), reverse=True)
        selected = candidates[0] if candidates else None

        return DiscoveryResult(
            candidates=candidates,
            selected=selected,
            ordering_basis="file_mtime",
            no_match=selected is None,
        )

    def convert(self, source_path: str) -> dict:
        """Read trade confirmation text."""
        p = Path(source_path)
        if not p.exists():
            return {"text": "", "ocr_used": False}
        return {"text": p.read_text(encoding="utf-8", errors="ignore"), "ocr_used": False}

    def extract(self, document_text: str) -> ExtractionResult:
        """Parse structured or semi-structured trade file."""
        text = document_text.strip()
        fields: list[ExtractedField] = []

        # 1. Try parsing direct JSON
        if text.startswith("{") and text.endswith("}"):
            try:
                data = json.loads(text)
                for k, v in data.items():
                    fields.append(
                        ExtractedField(
                            field_name=k,
                            value=v,
                            source_quote=f'"{k}": "{v}"' if isinstance(v, str) else f'"{k}": {v}',
                            field_confidence=1.0,
                            context_keywords=[k],
                        )
                    )
                return ExtractionResult(fields=fields, raw_text_length=len(text))
            except Exception:
                pass

        # 2. Key-value regex parsing
        patterns = {
            "trade_id": r"(?:trade[_\s]?id|order[_\s]?id|ref)[:=\s]+([A-Za-z0-9\-_]+)",
            "symbol": r"(?:symbol|ticker|asset)[:=\s]+([A-Za-z0-9]+)",
            "direction": r"(?:direction|side|action)[:=\s]+(buy|sell)",
            "quantity": r"(?:quantity|qty|shares|units)[:=\s]+([\d,]+(?:\.\d+)?)",
            "price": r"(?:price|rate|px)[:=\s]+[\$₹€£]?\s*([\d,]+(?:\.\d+)?)",
            "trade_date": r"(?:trade[_\s]?date|date)[:=\s]+(\d{4}-\d{2}-\d{2})",
        }

        for field_name, pat in patterns.items():
            m = re.search(pat, text, re.IGNORECASE)
            if m:
                val = m.group(1).replace(",", "").strip()
                if field_name in ("quantity", "price"):
                    try:
                        val = float(val)
                    except ValueError:
                        pass
                fields.append(
                    ExtractedField(
                        field_name=field_name,
                        value=val,
                        source_quote=m.group(0),
                        field_confidence=0.95,
                        context_keywords=[field_name],
                    )
                )

        return ExtractionResult(fields=fields, raw_text_length=len(text))

    def ground(self, extraction: ExtractionResult, document_text: str) -> list[GroundingResult]:
        """Ground extracted trade fields against raw document text."""
        results = []
        for f in extraction.fields:
            quote_in = f.source_quote.lower() in document_text.lower() or str(f.value).lower() in document_text.lower()
            results.append(
                GroundingResult(
                    field_name=f.field_name,
                    passed=quote_in,
                    quote_found=quote_in,
                    value_derived=True,
                    semantic_role_ok=True,
                )
            )
        return results

    def independent_check(self, extraction: ExtractionResult, document_text: str) -> dict:
        """Sanity check trade parameters."""
        f_map = {f.field_name: f.value for f in extraction.fields}
        mismatches = []
        if "quantity" in f_map and float(f_map["quantity"]) <= 0:
            mismatches.append({"field": "quantity", "reason": "Trade quantity must be strictly positive"})
        if "price" in f_map and float(f_map["price"]) <= 0:
            mismatches.append({"field": "price", "reason": "Trade price must be strictly positive"})

        passed = len(mismatches) == 0
        return {
            "passed": passed,
            "mismatches": mismatches,
            "message": "Trade integrity verified" if passed else "Invalid trade numerical values",
        }

    def bind(
        self,
        extraction: ExtractionResult,
        groundings: list[GroundingResult],
        contract: StaticContract,
    ) -> BoundState:
        """Bind and freeze validated trade fields."""
        vals = {}
        quotes = {}
        for f in extraction.fields:
            vals[f.field_name] = f.value
            quotes[f.field_name] = f.source_quote

        if "direction" in vals:
            vals["direction"] = str(vals["direction"]).lower()
        if "symbol" in vals:
            vals["symbol"] = str(vals["symbol"]).upper()
        if "trade_date" not in vals:
            vals["trade_date"] = datetime.utcnow().strftime("%Y-%m-%d")

        bound = BoundState(
            target_values=vals,
            source=BoundSource.EXTRACTED,
            source_quotes=quotes,
        )
        bound.compute_checksum()
        return bound

    def label_map(self, form_id: str) -> dict[str, str]:
        """Label mapping for trade execution forms."""
        return {
            "trade_id": "Trade ID",
            "symbol": "Symbol / Ticker",
            "direction": "Side (Buy/Sell)",
            "quantity": "Quantity",
            "price": "Price per Share",
            "trade_date": "Execution Date",
        }

    def fuzzy_label_match(self, form_id: str, available_labels: list[str]) -> dict[str, str]:
        """Levenshtein fallback for slightly altered form labels."""
        import Levenshtein
        mapping = {}
        target_map = self.label_map(form_id)
        for key, expected_label in target_map.items():
            best_match, best_dist = None, 999
            for avail in available_labels:
                dist = Levenshtein.distance(expected_label.lower(), avail.lower())
                if dist < best_dist and dist <= 3:
                    best_match, best_dist = avail, dist
            if best_match:
                mapping[key] = best_match
        return mapping

    def snapshot(self, business_key_value: str | None = None) -> Baseline:
        return snapshot_table("trade_id", business_key_value)

    def key_state(self, business_key_value: str) -> dict:
        return key_state("trade_id", business_key_value)

    def verify(self, bound: BoundState, baselines: list[Baseline]) -> VerifyResult:
        bk_val = str(bound.target_values.get("trade_id", ""))
        return verify(bound, baselines, "trade_id", bk_val)

    def recovery_for(self, error_or_diff: dict) -> dict | None:
        error_type = str(error_or_diff.get("error_type", "")).strip().lower()
        table = {
            "key_missing": {"action": "resubmit", "max_retries": 2, "instruction": "Trade missing from ledger; re-submitting order"},
            "wrong_values": {"action": "refill_from_bound", "max_retries": 2, "instruction": "Trade field discrepancy; re-entering order values"},
            "read_timeout": {"action": "backoff_retry", "max_retries": 3, "instruction": "Timeout reading trade ledger; retrying"},
            "write_timeout": {"action": "key_check_then_decide", "max_retries": 2, "instruction": "Key check after booking timeout"},
            "browser_session_lost": {"action": "reset_session", "max_retries": 2, "instruction": "Session drop detected; re-authenticating"},
        }
        return table.get(error_type)

    def reset_view(self) -> None:
        pass
