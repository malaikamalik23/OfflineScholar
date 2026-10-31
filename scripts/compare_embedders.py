"""Compare the TF-IDF fallback with the real sentence-transformers model on syllabus coverage.

Needs the model saved by ``python scripts/download_models.py`` (data/models/all-MiniLM-L6-v2).
Run from the project root:   python scripts/compare_embedders.py

It prints, per syllabus topic, the status and evidence score under both embedders, plus a few
paraphrase probes (same idea, different words) where only a semantic model should succeed.
Use the output to calibrate ``exam_engine.embeddings.THRESHOLDS`` on your own material.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from exam_engine import analyze_syllabus_coverage  # noqa: E402
from exam_engine.embeddings import SentenceTransformerEmbedder, TfidfEmbedder  # noqa: E402
from tests.exam_fixtures import LECTURES, SYLLABUS_TOPICS  # noqa: E402

PROBES = [
    # (syllabus topic, lecture text that explains it WITHOUT using the topic's words)
    ("Overfitting", "A model that memorises noise in the training data performs well on seen examples "
                    "but poorly on new, unseen examples."),
    ("Central processing unit", "The part of the computer that fetches, decodes and executes instructions."),
    ("Gradient descent", "We repeatedly nudge the weights a small step in the direction that lowers the loss."),
]


def model_path() -> str:
    local = ROOT / "data" / "models" / "all-MiniLM-L6-v2"
    return str(local) if local.is_dir() else "sentence-transformers/all-MiniLM-L6-v2"


def row(res: dict) -> dict:
    return {t["topic"]: (t["status"], t.get("evidence_score")) for t in res["topics"]}


def main() -> int:
    try:
        dense = SentenceTransformerEmbedder(model_path(), local_files_only=True)
    except Exception as exc:  # noqa: BLE001
        print(f"Could not load the model ({type(exc).__name__}: {exc}).\n"
              "Run:  python scripts/download_models.py   (needs internet once)")
        return 1
    lexical = TfidfEmbedder()

    print("== Fixture course: syllabus coverage ==")
    a = row(analyze_syllabus_coverage(SYLLABUS_TOPICS, LECTURES, embedder=lexical))
    b = row(analyze_syllabus_coverage(SYLLABUS_TOPICS, LECTURES, embedder=dense))
    print(f"{'topic':38} {'tf-idf':24} {'real model':24}")
    for topic in a:
        fa, fb = a[topic], b.get(topic, ("?", None))
        print(f"{topic[:37]:38} {fa[0]:>14} {fa[1]!s:>8}   {fb[0]:>14} {fb[1]!s:>8}")

    print("\n== Paraphrase probes (topic words NOT in the text) ==")
    for topic, text in PROBES:
        chunk = [{"document_id": 1, "document_name": "probe.pdf", "page_number": 1, "text": text}]
        ra = analyze_syllabus_coverage([topic], chunk, embedder=lexical)["topics"][0]
        rb = analyze_syllabus_coverage([topic], chunk, embedder=dense)["topics"][0]
        print(f"{topic:26} tf-idf: {ra['status']:<18} real: {rb['status']:<18} (score {rb.get('evidence_score')})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
