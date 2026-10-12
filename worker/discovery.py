"""Discovery — Stage 3: File discovery and selection.

Deterministic file scan with explicit algorithm:
1. SCAN: glob for PDFs in configured directory
2. MATCH: regex on filename + first-page text fallback
3. ORDER: by file mtime or doc_date
4. TIE DETECTION: identical ordering values
5. CONFLICT: mtime vs doc_date disagree
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from pathlib import Path

from worker.config import INVOICE_DIR
from worker.models import SourceRef, DiscoveryResult

logger = logging.getLogger(__name__)


def discover(
    params: dict,
    source_dir: Path | None = None,
) -> DiscoveryResult:
    """Discover and select the best matching source document.

    Args:
        params: Must include "company" key. May include "latest" flag.
        source_dir: Override discovery directory.
    """
    company = params.get("company", "").strip()
    if not company:
        return DiscoveryResult(no_match=True)

    search_dir = source_dir or INVOICE_DIR
    if not search_dir.exists():
        logger.error(f"Source directory does not exist: {search_dir}")
        return DiscoveryResult(no_match=True)

    # ── 1. SCAN ──────────────────────────────────────────────────────────
    pdf_files = sorted(search_dir.glob("*.pdf"))
    if not pdf_files:
        logger.info(f"No PDF files in {search_dir}")
        return DiscoveryResult(no_match=True)

    # ── 2. MATCH COMPANY ─────────────────────────────────────────────────
    company_pattern = re.compile(
        re.escape(company).replace(r"\ ", r"[\s_\-]*"),
        re.IGNORECASE,
    )

    candidates: list[SourceRef] = []
    for pdf in pdf_files:
        confidence = 0.0
        match_source = ""

        # Primary: filename match
        if company_pattern.search(pdf.stem):
            confidence = 0.9
            match_source = "filename"
        else:
            # Fallback: first-page text match
            first_page_text = _extract_first_page_text(pdf)
            if first_page_text and company_pattern.search(first_page_text):
                confidence = 0.7
                match_source = "first_page_text"

        if confidence > 0:
            mtime = os.path.getmtime(pdf)
            doc_date = _extract_doc_date(pdf)

            candidates.append(SourceRef(
                path=str(pdf),
                company_match_confidence=confidence,
                ordering_basis="file_mtime",
                ordering_value=datetime.fromtimestamp(mtime).isoformat(),
                metadata={
                    "filename": pdf.name,
                    "match_source": match_source,
                    "mtime": datetime.fromtimestamp(mtime).isoformat(),
                    "doc_date": doc_date,
                    "size_bytes": pdf.stat().st_size,
                },
            ))

    if not candidates:
        return DiscoveryResult(no_match=True)

    # ── 3. DETERMINE ORDERING ────────────────────────────────────────────
    # Check for ordering conflict: mtime vs doc_date disagree
    ordering_conflict = False
    if len(candidates) > 1:
        has_doc_dates = all(c.metadata.get("doc_date") for c in candidates)
        if has_doc_dates:
            by_mtime = sorted(candidates, key=lambda c: c.metadata["mtime"], reverse=True)
            by_doc = sorted(candidates, key=lambda c: c.metadata["doc_date"], reverse=True)
            if by_mtime[0].path != by_doc[0].path:
                ordering_conflict = True
                logger.warning("Ordering conflict: newest by mtime != newest by doc_date")

    if ordering_conflict:
        return DiscoveryResult(
            candidates=candidates,
            ordering_conflict=True,
            ordering_basis="conflict",
        )

    # ── 4. SORT & TIE DETECTION ──────────────────────────────────────────
    candidates.sort(key=lambda c: c.ordering_value, reverse=True)

    tie = False
    if len(candidates) >= 2:
        if candidates[0].ordering_value == candidates[1].ordering_value:
            tie = True
            logger.warning("Tie detected: top 2 candidates have identical ordering values")

    if tie:
        return DiscoveryResult(
            candidates=candidates,
            tie=True,
            ordering_basis="file_mtime",
        )

    # ── 5. SELECT ────────────────────────────────────────────────────────
    selected = candidates[0]
    logger.info(f"Selected: {selected.metadata.get('filename')} (confidence={selected.company_match_confidence})")

    return DiscoveryResult(
        candidates=candidates,
        selected=selected,
        ordering_basis="file_mtime",
    )


def _extract_first_page_text(pdf_path: Path) -> str | None:
    """Extract text from the first page for company matching."""
    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        if len(doc) == 0:
            doc.close()
            return None
        text = doc[0].get_text()
        doc.close()
        return text[:500]  # first 500 chars is enough for matching
    except Exception as e:
        logger.debug(f"Could not read first page of {pdf_path}: {e}")
        return None


def _extract_doc_date(pdf_path: Path) -> str | None:
    """Try to extract a document date from filename or metadata."""
    # Try filename patterns: invoice_company_2026-10-01.pdf, invoice_20261001.pdf
    patterns = [
        r'(\d{4}[-_]\d{2}[-_]\d{2})',
        r'(\d{8})',
        r'(\d{4})',
    ]
    for pattern in patterns:
        match = re.search(pattern, pdf_path.stem)
        if match:
            date_str = match.group(1).replace("_", "-")
            return date_str

    # Try PDF metadata
    try:
        import pymupdf
        doc = pymupdf.open(str(pdf_path))
        meta = doc.metadata
        doc.close()
        creation_date = meta.get("creationDate", "")
        if creation_date:
            return creation_date[:10]
    except Exception:
        pass

    return None
