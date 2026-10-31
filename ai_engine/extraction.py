"""PDF text extraction with PyMuPDF, plus optional Tesseract OCR for scanned pages."""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf as fitz  # PyMuPDF

MIN_PAGE_CHARS = 20  # fewer characters than this -> page treated as image-only
# A page with a short text layer (e.g. a slide title) whose images cover a good part of the page is
# OCR'd as well, so content shown only in pictures (tables, formulas, diagrams) becomes searchable.
# Measured on real lecture slides: such pages have ~40-150 chars of text and images on 14-40% of the page.
SPARSE_PAGE_CHARS = 200
MIN_IMAGE_COVERAGE = 0.10


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


def _windows_tesseract_candidates() -> list[Path]:
    # The official Windows installer (UB Mannheim) does not add Tesseract to PATH.
    roots = [os.environ.get(v) for v in ("ProgramFiles", "ProgramFiles(x86)", "LOCALAPPDATA")]
    return [Path(r) / sub / "tesseract.exe" for r in roots if r
            for sub in ("Tesseract-OCR", "Programs/Tesseract-OCR")]


def find_tesseract(cmd: str | None = None) -> str | None:
    """Path of the tesseract binary: explicit `cmd` (OS_TESSERACT_CMD), else PATH, else the
    default Windows install folders. None if it cannot be found."""
    if cmd:
        return shutil.which(cmd) or (cmd if Path(cmd).is_file() else None)
    found = shutil.which("tesseract")
    if found or sys.platform != "win32":
        return found
    return next((str(p) for p in _windows_tesseract_candidates() if p.is_file()), None)


def tesseract_available(cmd: str | None = None) -> bool:
    return find_tesseract(cmd) is not None


def clean_text(text: str) -> str:
    text = text.replace("\x00", "").replace("­", "")
    text = re.sub(r"-\n(?=[a-z])", "", text)  # re-join words hyphenated at line end
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _image_coverage(page: fitz.Page) -> float:
    """Fraction of the page area covered by images (overlaps counted twice; capped at 1)."""
    area = 0.0
    for img in page.get_images():
        try:
            area += sum(abs(r.get_area()) for r in page.get_image_rects(img[0]))
        except Exception:  # noqa: BLE001 - unusual image references: ignore
            continue
    return min(1.0, area / page.rect.get_area()) if page.rect.get_area() else 0.0


def _ocr_page(page: fitz.Page, language: str, dpi: int, binary: str) -> str:
    png = page.get_pixmap(dpi=dpi).tobytes("png")
    proc = subprocess.run(
        [binary, "stdin", "stdout", "-l", language],
        input=png, capture_output=True, timeout=180, check=False,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),  # no console flash on Windows
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
    tesseract_cmd: str | None = None,
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

        binary = find_tesseract(tesseract_cmd) if ocr_enabled else None
        can_ocr = binary is not None
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
                        text = clean_text(_ocr_page(page, ocr_language, ocr_dpi, binary))
                        used_ocr = True
                    elif page.get_images():
                        image_only_without_ocr += 1
                elif can_ocr and len(text) < SPARSE_PAGE_CHARS and _image_coverage(page) >= MIN_IMAGE_COVERAGE:
                    # Slide with just a title in the text layer and its content in a picture.
                    try:
                        ocr_text = clean_text(_ocr_page(page, ocr_language, ocr_dpi, binary))
                        if len(ocr_text) > len(text):
                            text, used_ocr = ocr_text, True
                    except Exception as exc:  # keep the text layer we already have
                        warnings.append(f"Page {number}: OCR of embedded images failed: {exc}")
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
