"""Converter — Stage 4.1: PDF → markdown text.

Uses PyMuPDF for text-layer extraction. Falls back to Tesseract OCR
if the text layer is empty or too short.
Detects hidden text (white/tiny font) from span properties.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)

# Minimum text length to consider a valid text layer
MIN_TEXT_LAYER_LENGTH = 50


def convert_pdf(pdf_path: str | Path) -> dict:
    """Convert a PDF to markdown text with metadata.

    Returns:
        {
            "text": str,          # extracted markdown text
            "ocr_used": bool,     # whether OCR was used
            "hidden_text_flag": bool,  # suspicious hidden text detected
            "page_count": int,
            "ocr_confidences": dict,  # word-level if OCR used
        }
    """
    import pymupdf

    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    doc = pymupdf.open(str(pdf_path))
    result = {
        "text": "",
        "ocr_used": False,
        "hidden_text_flag": False,
        "page_count": len(doc),
        "ocr_confidences": {},
    }

    all_text_parts = []
    hidden_text_detected = False

    for page_num, page in enumerate(doc):
        # Extract text with formatting info
        blocks = page.get_text("dict", flags=pymupdf.TEXT_PRESERVE_WHITESPACE)

        page_text_parts = []
        for block in blocks.get("blocks", []):
            if block.get("type") != 0:  # text block
                continue
            for line in block.get("lines", []):
                line_text = ""
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text:
                        continue

                    # Detect hidden text (white font on white bg, or tiny font)
                    font_size = span.get("size", 12)
                    color = span.get("color", 0)

                    if font_size < 2 or color == 16777215:  # white = 0xFFFFFF
                        hidden_text_detected = True
                        logger.warning(
                            f"Hidden text detected on page {page_num + 1}: "
                            f"'{text[:50]}...' (size={font_size}, color={color})"
                        )
                        continue  # skip hidden text

                    line_text += text + " "

                if line_text.strip():
                    page_text_parts.append(line_text.strip())

        page_text = "\n".join(page_text_parts)
        all_text_parts.append(page_text)

    full_text = "\n\n".join(all_text_parts).strip()
    result["hidden_text_flag"] = hidden_text_detected

    # Check if text layer is sufficient
    if len(full_text) < MIN_TEXT_LAYER_LENGTH:
        logger.info("Text layer too short, attempting OCR fallback")
        ocr_result = _ocr_fallback(doc)
        if ocr_result:
            result["text"] = ocr_result["text"]
            result["ocr_used"] = True
            result["ocr_confidences"] = ocr_result.get("confidences", {})
            doc.close()
            return result

    result["text"] = full_text
    doc.close()
    return result


def _ocr_fallback(doc) -> dict | None:
    """Use Tesseract OCR as fallback for scanned PDFs.

    Returns text with per-word confidence.
    """
    try:
        import pytesseract
        from PIL import Image
        import io

        all_text = []
        confidences = {}
        word_index = 0

        for page_num, page in enumerate(doc):
            # Render page to image
            pix = page.get_pixmap(dpi=300)
            img = Image.open(io.BytesIO(pix.tobytes("png")))

            # Run Tesseract with confidence data
            data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)

            page_words = []
            for i, word in enumerate(data["text"]):
                word = word.strip()
                if not word:
                    continue
                conf = int(data["conf"][i])
                if conf > 0:  # -1 means no confidence
                    confidences[f"word_{word_index}"] = {
                        "word": word,
                        "confidence": conf / 100.0,
                        "page": page_num,
                    }
                    page_words.append(word)
                    word_index += 1

            all_text.append(" ".join(page_words))

        return {
            "text": "\n\n".join(all_text),
            "confidences": confidences,
        }
    except ImportError:
        logger.warning("pytesseract not installed, OCR fallback unavailable")
        return None
    except Exception as e:
        logger.error(f"OCR failed: {e}")
        return None


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace and Unicode in extracted text."""
    text = re.sub(r'\s+', ' ', text)
    text = text.replace('\u00a0', ' ')  # non-breaking space
    text = text.replace('\u200b', '')   # zero-width space
    return text.strip()
