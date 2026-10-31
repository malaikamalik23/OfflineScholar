"""Measure answer quality and latency of a running Offline Scholar server on your own PDF.

    python scripts/evaluate.py --pdf "Lecture 3.pdf" --questions questions.json [--top-k 4] [--out results.json]

questions.json is a list of {"q": "...", "pages": [3], "kind": "direct"}; kind is one of
direct | para (paraphrased) | ocr (answer only in a slide image) | none (not covered: must be refused).

A question counts as OK when a covered question comes back grounded and cites one of the expected
pages, or an uncovered one comes back not grounded. That does NOT check that the answer text is right:
read the printed answers too (that is how the wrong numbers in docs/EVALUATION.md were found).
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import httpx


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--questions", required=True, type=Path)
    ap.add_argument("--base-url", default="http://127.0.0.1:8000")
    ap.add_argument("--subject", default="Evaluation")
    ap.add_argument("--top-k", type=int, default=4)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()

    api = args.base_url.rstrip("/") + "/api"
    questions = json.loads(args.questions.read_text(encoding="utf-8"))
    c = httpx.Client(timeout=600, trust_env=False)
    health = c.get(f"{api}/health").json()
    print(f"server: {health['status']} | LLM {health['llm']['model']} | embeddings {health['embeddings']['model']}")

    existing = [d for d in c.get(f"{api}/documents").json() if d["filename"] == args.pdf.name]
    if existing:
        doc = existing[0]
        print(f"reusing document {doc['id']} ({doc['status']})")
    else:
        t = time.perf_counter()
        r = c.post(f"{api}/documents/upload", data={"subject_name": args.subject},
                   files={"file": (args.pdf.name, args.pdf.read_bytes(), "application/pdf")})
        r.raise_for_status()
        doc = r.json()
        while (st := c.get(f"{api}/documents/{doc['id']}/status").json())["status"] in ("pending", "processing"):
            time.sleep(0.5)
        print(f"ingest: {time.perf_counter() - t:.1f}s, {st['page_count']} pages, {st['chunk_count']} chunks, "
              f"status {st['status']} {st.get('error') or ''}")

    results = []
    for item in questions:
        q, pages, kind = item["q"], set(item.get("pages", [])), item.get("kind", "direct")
        t = time.perf_counter()
        body = c.post(f"{api}/chat/ask", json={"question": q, "subject_id": doc["subject_id"],
                                                "top_k": args.top_k}).json()
        latency = time.perf_counter() - t
        cited = [s["page_number"] for s in body.get("sources", [])]
        ok = not body.get("grounded") if kind == "none" else bool(body.get("grounded") and set(cited) & pages)
        results.append({**item, "latency": round(latency, 2), "ok": ok, "cited": cited, **body})
        print(f"[{'OK ' if ok else 'BAD'}] {kind:6} {latency:5.1f}s grounded={body.get('grounded')} "
              f"cited={cited} expected={sorted(pages)} :: {q}\n       {' '.join(str(body.get('answer')).split())[:300]}")

    print()
    for kind in dict.fromkeys(r["kind"] for r in results):
        rs = [r for r in results if r["kind"] == kind]
        print(f"{kind:6} {sum(r['ok'] for r in rs)}/{len(rs)}")
    lat = [r["latency"] for r in results]
    print(f"latency: median {statistics.median(lat):.1f}s, max {max(lat):.1f}s")
    if args.out:
        args.out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
