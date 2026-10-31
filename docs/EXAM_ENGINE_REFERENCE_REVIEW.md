# Reference repository review

Reviewed on 2026-10-31 by shallow-cloning each repo and reading its source. AnythingLLM was only sparse-cloned, so just its top-level files (LICENSE, README, package.json) were inspected, not its source tree.
**No code was copied.** Everything in `exam_engine/` is original; the repos informed design decisions only.

| Repo | License as found | What it actually implements | What we took (as an idea) |
|---|---|---|---|
| SmartQP — Question Paper Analysis | **No LICENSE file, no license statement in README** → all rights reserved by default; do not copy | Streamlit app (`app.py`). `question_parser.py` (regex segmentation, marks, unit headers), `analysis.py` (sparse-vector cosine grouping at a fixed threshold 0.62, topic table, study plan), `syllabus.py`. TF-IDF based, not neural. | Regex-based, explainable question segmentation with marks + section detection; an "explain each topic's score" requirement. We use *average-linkage* clustering instead of its pairwise threshold groups. |
| QPredict | **No LICENSE file, no license statement** → do not copy | Streamlit app with `question_extractor.py` (KTU-format specific layouts), `clustering.py`, `syllabus_mapper.py`, `quiz_generator.py` (offline keyword MCQs + optional LLM), `scoring.py`. "Prediction" is frequency scoring, not forecasting. | Idea of mapping clusters to syllabus units, offline-first quiz fallback with LLM option. Its extractor is tuned to one university's layout; ours is format-agnostic with a fallback. |
| ExamRAG | README badge says MIT, but **no LICENSE file in the repo** → treat as unverified | Python CLI (`cli.py`, `main.py`) with an `app/intelligence` package: question classifier, topic extractor, predictor, study planner, practice mode; prediction step calls an LLM (`_generate_predictions_llm`). | Separating pattern analysis from LLM prediction; keeping the LLM optional. Our analysis never needs an LLM, so results are reproducible and explainable. |
| Local RAG QA Handbook | README says MIT; no LICENSE file | ~40-line LangChain script: PyPDFLoader → RecursiveCharacterTextSplitter → FAISS → Ollama. A tutorial, not an exam tool. | Confirms the shape of Member 1's retrieval layer (chunks → embeddings → FAISS → local LLM). We deliberately do **not** duplicate it. |
| AnythingLLM | **MIT** (LICENSE file verified) | Large Node.js project for document chat (per its README/package.json). Not Python; not exam-specific. Source tree not read. | Citing sources (document + page) on every answer/quiz item; workspace-scoped document sets. |

Key finding: none of the repos does what Offline Scholar needs end to end (semantic repeated-question detection **and**
syllabus coverage with page evidence **and** weak-topic revision planning). Two are TF-IDF/regex tools without a license;
two are generic RAG material.
