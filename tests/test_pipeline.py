from __future__ import annotations

import pytest

from ai_engine.chunking import chunk_pages
from ai_engine.extraction import ExtractionError, PageText, extract_document, tesseract_available

from .conftest import BIOLOGY, make_pdf, make_scanned_pdf


def test_extraction_keeps_page_numbers(pdfs):
    doc = extract_document(pdfs["biology"], ocr_enabled=False)
    assert doc.page_count == 2
    assert [p.page_number for p in doc.pages] == [1, 2]
    assert "chloroplasts" in doc.pages[0].text and "prophase" in doc.pages[1].text


def test_corrupted_file(tmp_path):
    bad = tmp_path / "bad.pdf"
    bad.write_bytes(b"%PDF-1.4 this is not really a pdf")
    with pytest.raises(ExtractionError) as e:
        extract_document(bad)
    assert e.value.code == "corrupted_pdf"


def test_blank_pdf_has_no_text(tmp_path):
    p = tmp_path / "blank.pdf"
    make_pdf(p, [""])
    with pytest.raises(ExtractionError) as e:
        extract_document(p, ocr_enabled=False)
    assert e.value.code == "no_extractable_text"


@pytest.mark.skipif(not tesseract_available(), reason="tesseract not installed")
def test_scanned_pdf_uses_ocr(tmp_path):
    p = make_scanned_pdf(tmp_path / "scan.pdf", "Newton second law states force equals mass times acceleration")
    doc = extract_document(p, ocr_enabled=True)
    assert doc.ocr_pages == 1
    assert "force" in doc.pages[0].text.lower()


def test_scanned_pdf_without_ocr_is_reported(tmp_path):
    p = make_scanned_pdf(tmp_path / "scan.pdf", "Some scanned words appear here on the page")
    with pytest.raises(ExtractionError) as e:
        extract_document(p, ocr_enabled=False)
    assert e.value.code == "no_extractable_text" and "scanned" in e.value.message.lower()


def test_chunks_respect_size_pages_and_overlap():
    text = " ".join(f"Sentence number {i} talks about topic {i}." for i in range(60))
    pages = [PageText(1, text), PageText(2, "Second page content that is long enough to keep.")]
    chunks = chunk_pages(pages, chunk_size=300, overlap=80)
    assert all(len(c.text) <= 300 for c in chunks)
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    assert {c.page_number for c in chunks} == {1, 2}
    page1 = [c for c in chunks if c.page_number == 1]
    assert len(page1) > 2
    # overlap: the end of one chunk reappears at the start of the next
    assert page1[0].text.split(". ")[-1][:15] in page1[1].text
    assert all(c.page_number == 2 for c in chunks[len(page1):])


def test_short_page_is_one_chunk():
    chunks = chunk_pages([PageText(7, BIOLOGY[1])])
    assert len(chunks) == 1 and chunks[0].page_number == 7
