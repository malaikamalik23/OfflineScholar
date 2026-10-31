"""PDF text extraction with PyMuPDF, plus optional Tesseract OCR for scanned pages."""
from __future__ import annotations

import re
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

MIN_PAGE_CHARS = 20  # fewer characters than this -> page treated as image-only


class ExtractionError(Exception):
    """Raised when a document cannot be turned into text. `code` is machine readable."""

    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code
        self.message = message


@dataclass
class PageText:
    page_number: int  # 1-based
    text: str
    ocr: bool = False


@dataclass
class ExtractedDocument:
    file_name: str
    page_count: int
    pages: list[PageText]  # only pages that produced text
    warnings: list[str] = field(default_factory=list)

    @property
    def ocr_pages(self) -> int:
        return sum(1 for p in self.pages if p.ocr)


def tesseract_available() -> bool:
    return shutil.which("tesseract") is not None


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("­", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated at line end
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _ocr_page(page: fitz.Page, language: str, dpi: int) -> str:
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    proc = subprocess.run(
        ["tesseract", "stdin", "stdout", "-l", language],
        input=png, capture_output=True, timeout=180, check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr.decode(errors="ignore").strip()[:200] or "tesseract failed")
    return proc.stdout.decode("utf-8", errors="ignore")


def extract_document(
    file_path: str | Path,
    *,
    ocr_enabled: bool = True,
    ocr_language: str = "eng",
    ocr_dpi: int = 200,
) -> ExtractedDocument:
    """Extract per-page text from a PDF. Page numbers are 1-based.

    Raises ExtractionError (codes: file_not_found, corrupted_pdf, encrypted_pdf, empty_pdf,
    no_extractable_text) for unusable files. Individual bad pages become warnings.
    """
    path = Path(file_path)
    if not path.is_file():
        raise ExtractionError("file_not_found", f"File not found: {path.name}")
    try:
        doc = fitz.open(path)
    except Exception as exc:  # fitz raises several exception types for broken files
        raise ExtractionError("corrupted_pdf", "The file is not a readable PDF.") from exc

    with doc:
        if doc.needs_pass:
            raise ExtractionError("encrypted_pdf", "The PDF is password protected.")
        if doc.page_count == 0:
            raise ExtractionError("empty_pdf", "The PDF has no pages.")

        can_ocr = ocr_enabled and tesseract_available()
        pages: list[PageText] = []
        warnings: list[str] = []
        image_only_without_ocr = 0

        for page in doc:
            number = page.number + 1
            try:
                text = clean_text(page.get_text("text"))
                used_ocr = False
                if len(text) < MIN_PAGE_CHARS:
                    if can_ocr:
                        text = clean_text(_ocr_page(page, ocr_language, ocr_dpi))
                        used_ocr = True
                    elif page.get_images():
                        image_only_without_ocr += 1
                if len(text) >= MIN_PAGE_CHARS or (text and not used_ocr and not page.get_images()):
                    pages.append(PageText(number, text, used_ocr))
            except Exception as exc:
                warnings.append(f"Page {number} skipped: {exc}")

        if image_only_without_ocr:
            warnings.append(
                f"{image_only_without_ocr} scanned page(s) skipped because OCR is unavailable "
                "(install Tesseract and keep OS_OCR_ENABLED=1)."
            )
        if not pages:
            if image_only_without_ocr:
                raise ExtractionError(
                    "no_extractable_text",
                    "This looks like a scanned PDF and OCR is not available. Install Tesseract to enable it.",
                )
            raise ExtractionError("no_extractable_text", "No readable text was found in this PDF.")

        return ExtractedDocument(path.name, doc.page_count, pages, warnings)
