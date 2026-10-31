"""Try quiz generation with the local Ollama model on a folder of lecture PDFs.

    python scripts/test_llm_quiz.py C:\\path\\to\\test-papers "logistic regression" "perceptron" [--model qwen2.5:1.5b]

For each topic it picks the best-matching lecture pages, asks the local model for questions and prints
what survived the checks (every question must quote evidence that exists in those pages) and what was
rejected. If Ollama is not reachable it says so and shows the extractive fallback instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine.extraction import extract_document  # noqa: E402
from ai_engine.llm import OllamaClient  # noqa: E402
from backend.config import get_settings  # noqa: E402
from exam_engine.bridge import OllamaQuizLLM  # noqa: E402
from exam_engine.embeddings import SentenceTransformerEmbedder, TfidfEmbedder, cosine_matrix, embed_all  # noqa: E402
from exam_engine.service import ExamEngine  # noqa: E402
from exam_engine.text_utils import strip_running_lines  # noqa: E402


def lecture_pages(folder: Path) -> list[dict]:
    pages = []
    for i, f in enumerate(sorted(folder.glob("*.pdf")), 1):
        n = f.name.lower()
        if any(k in n for k in ("midterm", "final", "paper", "exam", "syllabus", "outline")):
            continue
        ex = extract_document(str(f), ocr_enabled=False, ocr_language="eng", ocr_dpi=200)
        pages += [{"document_id": i, "document_name": f.name, "page_number": p.page_number, "text": p.text}
                  for p in ex.pages]
    return strip_running_lines(pages)


def main(argv: list[str]) -> int:
    model = "qwen2.5:1.5b"
    if "--model" in argv:
        model = argv[argv.index("--model") + 1]
        argv = [a for a in argv if a not in ("--model", model)]
    if len(argv) < 2:
        print(__doc__)
        return 1
    folder, topics = Path(argv[0]), argv[1:]
    pages = lecture_pages(folder)
    print(f"{len(pages)} lecture pages read")

    local = ROOT / "data" / "models" / "all-MiniLM-L6-v2"
    try:
        embedder = SentenceTransformerEmbedder(str(local))
    except Exception:  # noqa: BLE001
        embedder = TfidfEmbedder()

    s = get_settings()
    client = OllamaClient(s.ollama_url, model, timeout=240.0, temperature=0.2, num_ctx=4096)
    st = client.status()
    print("ollama:", st)
    engine = ExamEngine(embedder=embedder, llm=OllamaQuizLLM(client) if st.get("reachable") else None)

    for topic in topics:
        texts = [p["text"] for p in pages]
        q_vec, p_vec = embed_all(embedder, [topic], texts)
        sims = cosine_matrix(q_vec, p_vec)[0]
        best = sorted(range(len(pages)), key=lambda i: -sims[i])[:4]
        ctx = [pages[i] for i in best]
        quiz = engine.make_quiz(ctx, topic, n_questions=3)
        r = quiz["generation_report"]
        print(f"\n### {topic}  | from LLM: {r['from_llm']}  extractive: {r['from_extractive']}  "
              f"rejected: {r['llm_rejected']} {r['llm_rejection_reasons']}")
        for n in r["notes"]:
            print("   note:", n)
        for q in quiz["questions"]:
            print(f"\n  [{q['generation_method']}] {q['question']}")
            for o in q["options"]:
                print(f"     {'*' if o['key'] == q['correct_option'] else ' '} {o['key']}. {o['text'][:90]}")
            print(f"     source: {q['source']['document_name']} p{q['source']['page_number']}  | {q['explanation'][:160]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
