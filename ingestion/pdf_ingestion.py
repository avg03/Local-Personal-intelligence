"""
pdf_ingestion.py
Extract text from PDF (PyMuPDF primary, pytesseract OCR fallback per page)
and clean the resulting text for downstream chunking/embedding.
"""

import os
import re
import io
import logging
from typing import Optional

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
    """
    Extract text from a PDF file, page by page.
    Tries PyMuPDF text extraction first. If a page returns empty/unreadable
    text (e.g. scanned image page), falls back to pytesseract OCR on that
    page's rendered image and appends the OCR result.

    Args:
        pdf_path: path to the input PDF.
        save_path: optional path to save the extracted text (.txt).
                   Defaults to same name as pdf_path with .txt extension.
        ocr_dpi: render resolution used for OCR fallback (higher = better
                 OCR accuracy, slower).

    Returns:
        resume_text: full extracted text of the PDF (str).
    """
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
    """
    Clean raw extracted PDF/OCR text for downstream use (chunking, embedding).

    Handles:
    - collapsing repeated whitespace/newlines
    - stripping non-printable/control characters
    - removing common OCR/page artifacts (page numbers, lone bullet symbols)
    - normalizing hyphenated line-break word splits (e.g. "exam-\nple" -> "example")

    Args:
        text: raw extracted text.

    Returns:
        cleaned_text: cleaned string.
    """
    if not text:
        return ""

    cleaned_text = text

    # Fix words split across a line break with a hyphen
    cleaned_text = re.sub(r"-\n(?=[a-z])", "", cleaned_text)

    # Remove non-printable / control characters (keep basic punctuation, unicode letters)
    cleaned_text = re.sub(r"[^\x20-\x7E\n\u00A0-\uFFFF]", " ", cleaned_text)

    # Remove standalone page-number lines (e.g. "12", "Page 3")
    cleaned_text = re.sub(r"(?im)^\s*(page\s*)?\d+\s*$", "", cleaned_text)

    # Remove lone bullet/symbol-only lines left over from OCR noise
    cleaned_text = re.sub(r"(?m)^\s*[•·▪●‣o]\s*$", "", cleaned_text)

    # Collapse multiple blank lines into one
    cleaned_text = re.sub(r"\n\s*\n+", "\n\n", cleaned_text)

    # Collapse repeated spaces/tabs
    cleaned_text = re.sub(r"[ \t]+", " ", cleaned_text)

    # Trim leading/trailing whitespace per line
    cleaned_text = "\n".join(line.strip() for line in cleaned_text.split("\n"))

    return cleaned_text.strip()


def extract_text_with_page_map(pdf_path: str, ocr_dpi: int = 300):
    """
    Like extract_text_from_pdf, but cleans each page individually and
    tracks where each page's text lands in the final concatenated string.
    Use this version when you need page_number for the SQLite `chunks`
    table — offsets computed on raw (uncleaned) text would drift once
    clean_resume_text() collapses whitespace/removes lines.

    Returns:
        (full_text, page_boundaries) where full_text is the cleaned,
        concatenated text and page_boundaries is a list of dicts:
        {"page_number": int, "start_index": int, "end_index": int}
        (page_number is 1-indexed, offsets are into full_text)
    """
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    try:
        doc = fitz.open(pdf_path)
    except Exception as e:
        raise RuntimeError(f"Failed to open PDF with PyMuPDF: {pdf_path} — {e}")

    full_text = ""
    page_boundaries = []

    for page_num in range(len(doc)):
        raw_page_text = _extract_single_page_text(doc[page_num], page_num, ocr_dpi)
        cleaned_page_text = clean_resume_text(raw_page_text)

        if not cleaned_page_text:
            continue  # skip empty pages, don't create a zero-width boundary

        start = len(full_text)
        full_text += cleaned_page_text + "\n\n"
        end = len(full_text.rstrip())

        page_boundaries.append({
            "page_number": page_num + 1,
            "start_index": start,
            "end_index": end,
        })

    doc.close()
    return full_text.strip(), page_boundaries


def find_page_for_chunk(chunk_start_index: int, page_boundaries) -> Optional[int]:
    """
    Given a chunk's start_index (from SemanticChunker output) and the
    page_boundaries list from extract_text_with_page_map(), return which
    page_number that chunk starts in. Returns None if no match (shouldn't
    normally happen if offsets came from the same text).
    """
    for pb in page_boundaries:
        if pb["start_index"] <= chunk_start_index <= pb["end_index"]:
            return pb["page_number"]
    return None


if __name__ == "__main__":
    # quick manual test
    sample_pdf = "sample.pdf"
    if os.path.exists(sample_pdf):
        raw = extract_text_from_pdf(sample_pdf)
        cleaned = clean_resume_text(raw)
        print(cleaned[:500])
    else:
        print("No sample.pdf found for test run.")
