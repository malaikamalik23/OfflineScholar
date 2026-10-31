# Evaluation with the real models

First verification of Offline Scholar with the real models instead of the test doubles (hashing embedder,
fake LLM). Numbers below drove the defaults in `backend/config.py` and the prompt in `ai_engine/rag.py`.
Re-run on your own material with `scripts/evaluate.py` (see its docstring).

## Setup

| | |
|---|---|
| Machine | Windows 11, AMD Ryzen 5 PRO 5650U (6 cores / 12 threads), 15 GB RAM, integrated GPU only |
| LLM | Ollama 0.40.13, `qwen2.5:3b` (1.9 GB), CPU only |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2`, loaded from `data/models` |
| OCR | Tesseract (UB Mannheim build, `C:\Program Files\Tesseract-OCR`, not on PATH) |
| Document | A real 26-page lecture-slide PDF on supervised classification (decision trees, entropy, Naive Bayes, k-NN, regression). 13 slides have only a title in the text layer; their content (tables, formulas, diagrams) is in images. |
| Questions | 38: 16 answered directly on a slide, 9 paraphrased/borderline, 3 answered only inside slide images, 10 not covered by the lecture (must be refused). Each has the expected page(s). |

"OK" means: covered question → `grounded: true` and a cited page is an expected page; uncovered question →
`grounded: false`. Every answer was also read by hand; wrong content is listed separately below.

## Results (default settings: top_k 4, `OS_MIN_SCORE` 0.25, `OS_MIN_ANSWER_SUPPORT` 0.55)

qwen2.5:3b, model not yet loaded (cold) and Ollama's prompt cache empty, nothing else running:

| | First run (original prompt) * | Final |
|---|---|---|
| Direct | 15/16 | 15/16 |
| Paraphrased | 5/9 | 8/9 |
| Content only in slide images | 2/3 (0/3 before slide OCR: the images were never read) | 2/3 |
| Not covered, correctly refused | 10/10 | **10/10** |
| Answer latency, median / max | 7.5 s / 26.8 s | **6.5 s / 23.0 s** |
| `/api/health` latency | 3.7 s | 0.1 s |

\* already with the slide OCR and HTTP-client fixes, `OS_MIN_SCORE=0` (every question sent to the LLM).

Re-running the same questions back to back gives much lower latency (median 3.8 s) because Ollama reuses the
cached prompt; the cold figures above are what a student asking a new question gets. Answers vary slightly
between runs (temperature 0.1): the one direct miss in the final run was a refusal of a question qwen answered
correctly in other runs.

Wrong content found by reading the answers (qwen2.5:3b, every run):
- "Information gain of the Outlook attribute?" → *0.940* (the training-set entropy; the slide's result is 0.247).
- "Which attribute has the highest gain?" → *Outlook* (correct) but quoted the gain as 0.940.
- "How many positive examples are in the training data?" → *8* (slide says 9); the model counted table rows
  instead of reading the sentence.
Small models are weak at reading numbers out of OCR'd tables and formulas. Citations always pointed to the right
slide, so the student can check.

## Findings and the changes they led to

**Retrieval / `OS_MIN_SCORE`.** Best cosine score per question: covered questions 0.31–0.77 (median ≈0.50),
uncovered ones 0.12–0.40. The ranges overlap, so no threshold can separate them on its own; the LLM's `NOT_FOUND`
and the support guard do the real filtering. 0.25 is kept: no covered question scored below 0.31, and 3/10
uncovered questions fall below it and are answered instantly without calling the LLM.

**top_k.** Correct page within the top k: k=1 22/28, k=2 24/28, k=3 25/28, **k=4 26/28**, k=6 27/28. Going to 6
gains one question for ~50% more prompt tokens (≈ +3–5 s per answer on this CPU), so 4 stays.

**Chunk size/overlap.** 400/80, 600/100, 900/150 and 1200/200 gave the same retrieval quality on slides (each slide
is shorter than a chunk). 900/150 is kept. Not yet measured on dense text (textbook chapters, typed notes).
Fixed: a page's leftover text under 25% of a chunk is merged into the previous chunk; on one slide the final
result line ("= 0.940 − 0.694 = 0.247") had been split into a 203-character fragment.

**OCR of slide images.** OCR used to run only on pages with < 20 characters of text, so the 13 picture slides
were indexed as just their titles. Pages with < 200 characters of text whose images cover ≥ 10% of the page are
now OCR'd as well, and the OCR text is appended to (not substituted for) the exact text layer: replacing it lost
a clean "Outlook 0.247 … Wind 0.151" table and the model then quoted noisy OCR numbers. Cost: ≈1.5–3.3 s per
such page; ingesting this PDF takes ≈20 s (≈45 s for the first upload after a server start, which also loads
the embedding model).

**Prompt.** The original prompt made the model refuse when the slide used different words ("No need for feature
scaling" vs. "normalization", "Sensitive to outliers" under logistic regression). The new prompt accepts
paraphrased evidence, forbids added examples/details, asks to copy numbers exactly and to answer briefly. Brevity
plus `OS_LLM_MAX_TOKENS=400` bound the worst case (some first-run answers ran past 200 tokens); cold latency went
from 7.5 s / 26.8 s to 6.5 s / 23.0 s (median / max). Most of the remaining time is reading the ~1000-token
context, not writing the answer.

**Hallucination guard (`answer_support`, `OS_MIN_ANSWER_SUPPORT`).** With the looser prompt the model once
answered an uncovered question ("How does a random forest reduce overfitting?") from general knowledge, citing a
decision-tree slide, and once invented example symptoms ("fever, cough… influenza") that are not on the slide.
The guard measures which share of the answer's content words appear in the question or the passages it cites,
after cutting passage text the model pasted back (llama3.2 and phi3 do this, and the copied words hid a
general-knowledge answer). A first sentence that refuses in words ("the passages do not contain…") and
`NOTFOUND`/`NOT FOUND` count as abstentions. Across 187 faithful answers from 3 models the lowest support was
0.62; the 5 known hallucinations scored 0.12–0.50; threshold 0.55. Below it the API returns the normal
"couldn't find this" response with `grounded: false`. The guard assumes brief answers: long elaborations score
lower, which is one more reason the prompt asks for brevity. See the model comparison for its limits.

**LLM speed (qwen2.5:3b, this CPU).** Prompt processing 76–208 tokens/s, generation 15–17 tokens/s. Latency ≈
prompt tokens / ~100 + answer tokens / 16, so context size and answer length are the levers.

## Offline verification

The server was started with a wrapper that logs **and blocks** every DNS lookup and socket connection to a
non-loopback address, then a PDF was uploaded (with OCR), three questions were asked and the document was deleted.
With both `OS_OFFLINE=0` and `OS_OFFLINE=1`: everything worked and **zero** outbound attempts were made; the
server process held no remote TCP connections. With `OS_OFFLINE=1` and no `data/models` copy, the app used to
retry Hugging Face for ~13 s and then say "check your internet connection"; it now fails immediately with
instructions (`embedding_model_unavailable`, health `degraded`). Not covered: a run with the network physically
disconnected, and Ollama's own background update checks (the Ollama app, not this code).

## Model comparison

Same PDF, same 38 questions, same settings, each model cold (others unloaded, empty prompt cache). RAM is the
peak working set of Ollama's `llama-server` process at a 4096-token context; the API server itself (FastAPI +
MiniLM + FAISS) peaked at ~0.76 GB on top of that. Accuracy columns are the benchmark runs; the llama and phi
runs used the answer checks from before the echo/prose/`NOTFOUND` fixes, and the right-hand column applies the
final checks to their recorded answers.

| | **qwen2.5:3b** | llama3.2:3b | phi3:mini |
|---|---|---|---|
| Direct (right page) | 15/16 | 16/16 | 14/16 |
| Paraphrased | 8/9 | 6/9 | 7/9 |
| Slide-image content | 2/3 | 2/3 | 2/3 |
| Not covered, refused | 10/10 | 10/10 | 8/10 (10/10 with final checks) |
| Wrong content in answers (read by hand) | 3 wrong numbers | 4 wrong facts/numbers (entropy "9.5", "Temperature 0.971", "3 positive examples", noise attributed to Naive Bayes) + raw OCR text pasted as an answer | wrong numbers + general-knowledge answers (see below) |
| Pastes the prompt's passages back | no | often (headers) | often (whole passages) |
| Median / max latency | **6.5 s / 23.0 s** | 9.0 s / 44.6 s | 24.0 s / 64.9 s |
| RAM (llama-server) | **~2.2–2.5 GB** | ~2.8 GB | ~4.1 GB |
| Download | 1.9 GB | 2.0 GB | 2.2 GB |

How the uncovered questions were stopped (replay of every refused question that reached the LLM, final code):

| | qwen2.5:3b | llama3.2:3b | phi3:mini |
|---|---|---|---|
| Below `OS_MIN_SCORE`, LLM not called | 3 | 3 | 3 |
| Model abstained itself | 6 | 7 (4 of them written `NOTFOUND`/`NOT FOUND`) | 2 (appended `NOT_FOUND` after a general-knowledge answer) |
| Caught by the support guard | 1 (random forest, 0.30) | 0 | 2 (0.17, 0.32) + backprop (0.12) and a prose refusal in the benchmark |
| Got through | 0 | 0 | 1 in the replay (random forest, support 0.60) |

**Recommendation: keep `qwen2.5:3b` as the default.** It is the fastest and smallest of the three on this laptop,
the best on paraphrased questions, follows the answer format (exact `NOT_FOUND`, no pasted passages) and abstains
by itself on uncovered questions, so the guard is a backup rather than the main defence. llama3.2:3b is a
reasonable second choice (accurate on direct questions) but ~40% slower, makes more numeric mistakes and pastes
raw passage text. phi3:mini is not recommended: almost 4× slower, the most RAM, and it answers uncovered questions
from general knowledge often enough that one got past the guard.

Limits of the guard: it is a lexical heuristic. Across 187 faithful answers from all runs the lowest support was
0.62 and the known hallucinations scored 0.12–0.50, but phi3's random-forest answer reached 0.60 because
question words count as supported (removing them would push faithful short answers down to 0.33). It cannot
detect a wrong number copied from the right passage.
