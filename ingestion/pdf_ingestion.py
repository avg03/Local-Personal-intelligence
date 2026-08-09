"""
pdf_ingestion.py
Extract text from PDF (PyMuPDF primary, pytesseract OCR fallback per page)
and clean the resulting text for downstream chunking/embedding.

extract_text_with_page_map's contract changed from the earlier version:
it now returns PER-PAGE slots + which pages fully failed, instead of a
single concatenated string. That's on purpose - it lets a vision-recovery
step fill in just the failed pages IN PLACE before page_boundaries ever
gets computed, so boundaries are never computed against text that later
gets silently appended to. Call finalize_pages() once, after any recovery
is done, to get the final (full_text, page_boundaries).
"""

import os
import re
import io
import logging
from typing import Optional, List, Tuple, Dict

import fitz  # PyMuPDF
import pytesseract
from PIL import Image

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _extract_single_page_text(page, page_num: int, ocr_dpi: int) -> str:
    """Shared logic: try PyMuPDF text extraction, fall back to OCR if empty."""
    page_text = ""
    try:
        page_text = page.get_text().strip()
    except Exception as e:
        logger.warning(f"PyMuPDF failed on page {page_num + 1}: {e}")
        page_text = ""

    if not page_text:
        try:
            pix = page.get_pixmap(dpi=ocr_dpi)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            page_text = pytesseract.image_to_string(img).strip()
            logger.info(f"Page {page_num + 1}: used OCR fallback.")
        except Exception as e:
            logger.error(f"OCR fallback failed on page {page_num + 1}: {e}")
            page_text = ""

    return page_text


def extract_text_from_pdf(pdf_path: str, save_path: str = None, ocr_dpi: int = 300) -> str:
    """Extract text from a PDF file, page by page, concatenated. Kept for
    callers that don't need page tracking - use extract_text_with_page_map
    + finalize_pages instead if you do."""
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    resume_text = ""
    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF with PyMuPDF: {pdf_path} — {e}")

    for page_num in range(len(doc)):
        page_text = _extract_single_page_text(doc[page_num], page_num, ocr_dpi)
        resume_text += page_text + "\n"

    doc.close()

    if save_path is None:
        save_path = os.path.splitext(pdf_path)[0] + ".txt"
    try:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(resume_text)
    except Exception as e:
        logger.error(f"Failed to save extracted text to {save_path}: {e}")

    return resume_text


def clean_resume_text(text: str) -> str:
    """Clean raw extracted PDF/OCR text for downstream use."""
    if not text:
        return ""

    cleaned_text = text
    cleaned_text = re.sub(r"-\n(?=[a-z])", "", cleaned_text)
    cleaned_text = re.sub(r"[^\x20-\x7E\n\u00A0-\uFFFF]", " ", cleaned_text)
    cleaned_text = re.sub(r"(?im)^\s*(page\s*)?\d+\s*$", "", cleaned_text)
    cleaned_text = re.sub(r"(?m)^\s*[•·▪●‣o]\s*$", "", cleaned_text)
    cleaned_text = re.sub(r"\n\s*\n+", "\n\n", cleaned_text)
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)
    cleaned_text = "\n".join(line.strip() for line in cleaned_text.split("\n"))
    return cleaned_text.strip()


def extract_text_with_page_map(
    pdf_path: str, ocr_dpi: int = 300
) -> Tuple[List[str], List[int]]:
    """
    Extract text per page, WITHOUT finalizing into one string yet.

    Returns:
        (page_texts, failed_pages)
        page_texts: list, one entry per page (cleaned text, or "" if the
                    page fully failed - both PyMuPDF AND OCR came back empty).
        failed_pages: 0-indexed page numbers that need vision recovery.
                      Empty list means every page extracted fine.

    Call finalize_pages(page_texts) after filling in any failed pages to
    get the final (full_text, page_boundaries) - never call it before
    recovery is done, or boundaries will be computed against incomplete text.
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF with PyMuPDF: {pdf_path} — {e}")

    page_texts = []
    failed_pages = []

    for page_num in range(len(doc)):
        raw_page_text = _extract_single_page_text(doc[page_num], page_num, ocr_dpi)
        cleaned_page_text = clean_resume_text(raw_page_text)
        page_texts.append(cleaned_page_text)
        if not cleaned_page_text:
            failed_pages.append(page_num)

    doc.close()
    return page_texts, failed_pages


def finalize_pages(page_texts: List[str]) -> Tuple[str, List[Dict]]:
    """
    Concatenate per-page text into the final string and compute page
    boundaries - the ONLY place boundaries get computed, called once,
    after recovery (if any) has already filled in failed pages.
    """
    full_text = ""
    page_boundaries = []

    for page_num, text in enumerate(page_texts):
        if not text:
            continue  # still-empty page (recovery didn't fill it either) - skip, no zero-width boundary
        start = len(full_text)
        full_text += text + "\n\n"
        end = len(full_text.rstrip())
        page_boundaries.append({
            "page_number": page_num + 1,
            "start_index": start,
            "end_index": end,
        })

    return full_text.strip(), page_boundaries


def find_page_for_chunk(chunk_start_index: int, page_boundaries) -> Optional[int]:
    """Given a chunk's start_index and page_boundaries from finalize_pages(),
    return which page_number that chunk starts in."""
    for pb in page_boundaries:
        if pb["start_index"] <= chunk_start_index <= pb["end_index"]:
            return pb["page_number"]
    return None