"""Run the exam engine on a folder of real PDFs and print a readable report (no database needed).

    python scripts/analyze_folder.py C:\\path\\to\\test-papers [--lexical]

File roles come from the file names: ``*Midterm*`` / ``*Final*`` / ``*Paper*`` / ``*Exam*`` are past papers,
``*Syllabus*`` is the syllabus, everything else is lecture material. By default the real
sentence-transformers model from data/models is used (falls back to TF-IDF); ``--lexical`` forces TF-IDF.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine.extraction import extract_document  # noqa: E402
from exam_engine.bridge import syllabus_topics_from  # noqa: E402
from exam_engine.embeddings import SentenceTransformerEmbedder, TfidfEmbedder  # noqa: E402
from exam_engine.service import ExamEngine  # noqa: E402

PAPER = re.compile(r"midterm|final|paper|exam|quiz", re.I)


def role(name: str) -> str:
    if "syllabus" in name.lower() or "outline" in name.lower():
        return "syllabus"
    return "past_paper" if PAPER.search(name) else "lecture"


def pages_of(path: Path, doc_id: int) -> list[dict]:
    ex = extract_document(str(path), ocr_enabled=False, ocr_language="eng", ocr_dpi=200)
    return [{"document_id": doc_id, "document_name": path.name, "page_number": p.page_number, "text": p.text}
            for p in ex.pages]


def main(argv: list[str]) -> int:
    args = [a for a in argv if not a.startswith("--")]
    if not args:
        print(__doc__)
        return 1
    folder = Path(args[0])
    by_role: dict[str, list[dict]] = {"past_paper": [], "syllabus": [], "lecture": []}
    for i, f in enumerate(sorted(folder.glob("*.pdf")), 1):
        by_role[role(f.name)] += pages_of(f, i)
    print({k: len(v) for k, v in by_role.items()}, "pages per role")

    embedder = TfidfEmbedder()
    if "--lexical" not in argv:
        local = ROOT / "data" / "models" / "all-MiniLM-L6-v2"
        try:
            embedder = SentenceTransformerEmbedder(str(local) if local.is_dir()
                                                   else "sentence-transformers/all-MiniLM-L6-v2")
        except Exception as exc:  # noqa: BLE001
            print(f"(real model unavailable: {type(exc).__name__}; using TF-IDF)")
    engine = ExamEngine(embedder=embedder)
    print("matching:", embedder.kind)

    syllabus = syllabus_topics_from(by_role["syllabus"]) if by_role["syllabus"] else []
    print(f"\n== Syllabus topics read ({len(syllabus)}) ==")
    for t in syllabus:
        print("  -", t["name"], ("  [" + ", ".join(t.get("subtopics", [])) + "]") if t.get("subtopics") else "")

    past = engine.analyze_past_papers(by_role["past_paper"], syllabus_topics=syllabus or None)
    qs = past["questions"]
    print(f"\n== Questions found: {len(qs)} ==")
    for q in qs[:8]:
        print(f"  [{q.get('year')}] p{q.get('page_number')} {q['text'][:110]!r}")
    print(f"\n== Repeated questions: {len(past['repeated_questions'])} ==")
    for g in past["repeated_questions"][:5]:
        print("  -", g["representative"][:100], "| papers:", g["paper_count"], "| terms:", g["shared_key_terms"][:4])
    print("\n== Important topics ==")
    for t in past["important_topics"][:10]:
        print(f"  {t['importance_score']:>5} {t['topic']:<40} papers={t['paper_count']} questions={t['question_count']}")

    if syllabus and by_role["lecture"]:
        cov = engine.syllabus_coverage(syllabus, by_role["lecture"])
        print("\n== Syllabus coverage ==")
        for t in cov["topics"]:
            print(f"  {t['status']:<18} {t['evidence_score']:<5} {t['topic'][:45]:<46} {t.get('document') or ''} {t.get('pages') or ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
