"""Benchmark local LLMs on the real Offline Scholar pipeline.

    python scripts/benchmark_models.py --models qwen2.5:3b llama3.2:3b phi3:mini
    python scripts/benchmark_models.py --retrieval-only        # no LLM: check retrieval first

Ingests the PDFs in --pdf-dir (layout described in the questions file) into a fresh, separate data folder
using the real extractor, chunker, embedding model and FAISS index, then asks every question to each
model through ai_engine.rag.answer_question, the same function behind POST /api/chat/ask.

Per question it records: grounded or refused, whether the answer contains the required facts, whether a
cited source is a correct page, whether the right page was retrieved at all, and latency.
Results are written as JSON plus a Markdown summary to --out. Nothing here talks to the internet:
the embedding model must already be downloaded (scripts/download_models.py) and Ollama must be running.
"""
from __future__ import annotations

import argparse
import fnmatch
import hashlib
import json
import re
import shutil
import statistics
import sys
import time
import unicodedata
from dataclasses import replace
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ai_engine.ingest import ingest_document  # noqa: E402
from ai_engine.llm import LLMUnavailableError, OllamaClient  # noqa: E402
from ai_engine.rag import answer_question  # noqa: E402
from ai_engine.retrieval import retrieve_context  # noqa: E402
from ai_engine.runtime import Services, build_services  # noqa: E402
from backend.config import Settings  # noqa: E402
from backend.database import configure_database, session_scope  # noqa: E402
from backend.models import Document, Subject  # noqa: E402

DEFAULT_MODELS = ["qwen2.5:3b", "llama3.2:3b", "phi3:mini"]


# ---------------------------------------------------------------- setup


def ingest(spec: dict, pdf_dir: Path, svc: Services) -> dict[str, int]:
    """Load every PDF into its subject. Returns {subject name: subject id}."""
    subject_ids: dict[str, int] = {}
    for subj in spec["subjects"]:
        folder = pdf_dir / subj["folder"]
        files = sorted(folder.glob("*.pdf"))
        if not files:
            sys.exit(f"No PDFs in {folder}")
        with session_scope() as db:
            s = Subject(name=subj["name"])
            db.add(s)
            db.flush()
            subject_ids[subj["name"]] = s.id
        for f in files:
            doc_type = next((t for pat, t in subj["doc_types"].items() if fnmatch.fnmatch(f.name, pat)),
                            subj["default_doc_type"])
            data = f.read_bytes()
            with session_scope() as db:
                doc = Document(subject_id=subject_ids[subj["name"]], filename=f.name, doc_type=doc_type,
                               stored_path=str(f), sha256=hashlib.sha256(data).hexdigest(), size_bytes=len(data))
                db.add(doc)
                db.flush()
                doc_id = doc.id
            t0 = time.perf_counter()
            ingest_document(doc_id, services=svc)
            with session_scope() as db:
                d = db.get(Document, doc_id)
                print(f"  {f.name:<55} {doc_type:<10} {d.status:<6} {d.page_count:>3} pages "
                      f"{d.chunk_count:>4} chunks  {time.perf_counter() - t0:5.1f}s" + (f"  {d.error}" if d.error else ""))
    return subject_ids


# ---------------------------------------------------------------- grading


def norm(text: str) -> str:
    """NFKC turns ligatures such as 'ﬁ' into 'fi' so regexes match what a reader sees."""
    return unicodedata.normalize("NFKC", text)


def facts_ok(answer: str, groups: list[str]) -> bool:
    a = norm(answer)
    return all(re.search(g, a, re.I) for g in groups)


def source_hit(pairs: list[tuple[str, int]], expected: list[list]) -> bool:
    want = {(f, int(p)) for f, p in expected}
    return any((f, int(p)) in want for f, p in pairs)


def check_retrieval(spec: dict, subject_ids: dict[str, int], svc: Services, top_k: int) -> dict[str, dict]:
    """Model-independent: did retrieval find the right page, and did refusal questions retrieve anything?"""
    out = {}
    for q in spec["questions"]:
        hits = retrieve_context(q["question"], subject_ids[q["subject"]], top_k, services=svc)
        pairs = [(h.document_name, h.page_number) for h in hits]
        out[q["id"]] = {
            "retrieved": [[f, p, round(h.score, 3)] for (f, p), h in zip(pairs, hits)],
            "right_page_retrieved": source_hit(pairs, q["sources"]) if q["expect"] == "answer" else None,
        }
    return out


def run_model(model: str, spec: dict, subject_ids: dict[str, int], base: Services, top_k: int,
              retrieval: dict) -> list[dict]:
    s = base.settings
    llm = OllamaClient(s.ollama_url, model, s.ollama_timeout, s.llm_temperature, s.llm_num_ctx)
    svc = replace(base, llm=llm)
    st = llm.status()
    if not st.get("model_installed"):
        print(f"  ! {model} is not available in Ollama ({st.get('detail', 'not installed')}); skipped")
        return []
    llm.chat("Reply with OK.", "OK?")  # load the model into memory; not timed

    rows = []
    for q in spec["questions"]:
        t0 = time.perf_counter()
        try:
            ans = answer_question(q["question"], subject_ids[q["subject"]], top_k, services=svc)
            error = None
        except LLMUnavailableError as exc:
            ans, error = None, str(exc)
        secs = time.perf_counter() - t0
        llm_called = bool(retrieval[q["id"]]["retrieved"])
        row = {"id": q["id"], "expect": q["expect"], "seconds": round(secs, 2), "llm_called": llm_called,
               "error": error}
        if ans is not None:
            cited = [(src.document_name, src.page_number) for src in ans.sources]
            row.update(answer=ans.answer, grounded=ans.grounded, confidence=round(ans.confidence, 3),
                       sources=[list(c) for c in cited])
            if q["expect"] == "answer":
                row["facts_ok"] = ans.grounded and facts_ok(ans.answer, q["must_match"])
                row["citation_ok"] = ans.grounded and source_hit(cited, q["sources"])
            row["behaviour_ok"] = ans.grounded == (q["expect"] == "answer")
        rows.append(row)
        mark = "ok " if row.get("behaviour_ok") and row.get("facts_ok", True) else "XX "
        print(f"  {mark}{q['id']:<6} {secs:5.1f}s  grounded={row.get('grounded')}  "
              f"facts={row.get('facts_ok', '-')}  cite={row.get('citation_ok', '-')}")
    return rows


# ---------------------------------------------------------------- report


def pct(n: int, d: int) -> str:
    return f"{n}/{d} ({100 * n / d:.0f}%)" if d else "n/a"


def summarise(model: str, rows: list[dict]) -> dict:
    ans = [r for r in rows if r["expect"] == "answer" and r["error"] is None]
    ref = [r for r in rows if r["expect"] == "refuse" and r["error"] is None]
    llm_secs = [r["seconds"] for r in rows if r["llm_called"] and r["error"] is None]
    return {
        "model": model,
        "answerable": len(ans),
        "correct": sum(r["facts_ok"] for r in ans),
        "cited_right_page": sum(r["citation_ok"] for r in ans),
        "over_refused": sum(not r["grounded"] for r in ans),
        "unanswerable": len(ref),
        "refused": sum(not r["grounded"] for r in ref),
        "hallucinated": sum(r["grounded"] for r in ref),
        "errors": sum(r["error"] is not None for r in rows),
        "median_s": round(statistics.median(llm_secs), 1) if llm_secs else None,
        "max_s": round(max(llm_secs), 1) if llm_secs else None,
    }


def markdown(summaries: list[dict], retrieval: dict, spec: dict, meta: dict) -> str:
    answerable = [q for q in spec["questions"] if q["expect"] == "answer"]
    refusals = [q for q in spec["questions"] if q["expect"] == "refuse"]
    hit = sum(retrieval[q["id"]]["right_page_retrieved"] for q in answerable)
    no_passage = sum(not retrieval[q["id"]]["retrieved"] for q in refusals)
    lines = [
        f"Run {meta['started']} · top_k={meta['top_k']} · min_score={meta['min_score']} · "
        f"embedding={meta['embedding_model']} · {meta['questions']} questions",
        "",
        f"Retrieval (no LLM): right page in top-{meta['top_k']} for {pct(hit, len(answerable))} answerable questions; "
        f"nothing retrieved for {pct(no_passage, len(refusals))} unanswerable questions (LLM not called).",
        "",
        "| Model | Correct answer | Cited a right page | Over-refused | Correctly refused | Made up an answer | Median s | Max s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for s in summaries:
        lines.append(f"| `{s['model']}` | {pct(s['correct'], s['answerable'])} | {pct(s['cited_right_page'], s['answerable'])} "
                     f"| {s['over_refused']} | {pct(s['refused'], s['unanswerable'])} | {s['hallucinated']} "
                     f"| {s['median_s']} | {s['max_s']} |")
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------- main


def main() -> None:
    # Print progress as it happens, even when piped, and keep "·" readable on a Windows console.
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf-dir", type=Path, default=ROOT / "data" / "benchmark")
    ap.add_argument("--questions", type=Path, default=ROOT / "scripts" / "benchmark_questions.json")
    ap.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "benchmark" / "results")
    ap.add_argument("--retrieval-only", action="store_true", help="check retrieval, skip the LLMs")
    args = ap.parse_args()

    spec = json.loads(args.questions.read_text(encoding="utf-8"))
    work = args.pdf_dir / "_run"  # throwaway database + index, never the app's real data
    shutil.rmtree(work, ignore_errors=True)
    settings = Settings(data_dir=work, offline=True, _env_file=None)
    configure_database(settings)
    base = build_services(settings)

    started = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"Ingesting PDFs from {args.pdf_dir} (embedding model: {settings.resolved_embedding_model})")
    subject_ids = ingest(spec, args.pdf_dir, base)

    retrieval = check_retrieval(spec, subject_ids, base, args.top_k)
    meta = {"started": started, "top_k": args.top_k, "min_score": settings.min_score,
            "embedding_model": settings.embedding_model, "questions": len(spec["questions"])}

    results, summaries = {}, []
    if not args.retrieval_only:
        for model in args.models:
            print(f"\n== {model}")
            rows = run_model(model, spec, subject_ids, base, args.top_k, retrieval)
            if rows:
                results[model] = rows
                summaries.append(summarise(model, rows))

    report = markdown(summaries, retrieval, spec, meta)
    print("\n" + report)
    for q in spec["questions"]:
        r = retrieval[q["id"]]
        if r["right_page_retrieved"] is False:
            print(f"  retrieval miss {q['id']}: wanted {q['sources']}, got {[x[:2] for x in r['retrieved']]}")

    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    (args.out / f"benchmark-{stamp}.json").write_text(json.dumps(
        {"meta": meta, "retrieval": retrieval, "summaries": summaries, "results": results}, indent=2,
        ensure_ascii=False), encoding="utf-8")
    (args.out / f"benchmark-{stamp}.md").write_text(report, encoding="utf-8")
    print(f"Saved results to {args.out}")


if __name__ == "__main__":
    main()
