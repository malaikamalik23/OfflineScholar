# Graph Report - offline-scholar  (2026-10-31)

## Corpus Check
- Corpus is ~48,538 words - fits in a single context window. You may not need a graph.

## Summary
- 863 nodes · 2169 edges · 46 communities (39 shown, 7 thin omitted)
- Extraction: 93% EXTRACTED · 7% INFERRED · 0% AMBIGUOUS · INFERRED: 150 edges (avg confidence: 0.9)
- Token cost: 82,630 input · 0 output

## Community Hubs (Navigation)
- PDF Extraction & Chunking
- LLM Client & Errors
- Ingestion & Retrieval
- Exam Similarity Scoring
- API Regression Tests
- SQLite Models & Sessions
- CI Workflow
- Exam Topic Analysis
- Quiz Generation
- Exam Engine Tests
- Exam Engine API Hooks
- Exam DB Bridge
- Exam Embedders (TF-IDF/ST)
- Question Extraction
- API Schemas
- Syllabus Coverage
- RAG Answering & Guard
- Config & Scripts
- App Startup & Windows Tests
- Document Routes & Errors
- FAISS Vector Store
- Embedder Interface
- Weak-Topic Analysis
- Embedding Model Loading
- Exam Engine Integration Bridge
- Test Fixtures
- Settings & Paths
- Evaluation & Model Benchmark
- ExamEngine Facade
- Benchmark & Example Scripts
- App Entry & File Storage
- NLP Text Utilities
- Exam Engine Contracts
- Reference Repo Review
- Folder Analysis Script
- Revision Planning
- Member 1-2 Integration Contract
- Quiz Text Cleaning
- Subject Routes
- API Reference & RAG Pipeline
- Dependencies & Test Doubles
- Concept Embedder Tests
- Embedding & Coverage Thresholds
- DB Lock Regression Test
- Known Limitations

## God Nodes (most connected - your core abstractions)
1. `upload()` - 37 edges
2. `content_terms()` - 29 edges
3. `extract_document()` - 28 edges
4. `create_app()` - 26 edges
5. `build_services()` - 25 edges
6. `get_services()` - 25 edges
7. `Settings` - 24 edges
8. `Services` - 23 edges
9. `group_similar_questions()` - 23 edges
10. `analyze_syllabus_coverage()` - 23 edges

## Surprising Connections (you probably didn't know these)
- `Local RAG QA Handbook` --semantically_similar_to--> `Grounded RAG Pipeline (PDF -> chunks -> MiniLM -> FAISS -> Ollama)`  [INFERRED] [semantically similar]
  docs/EXAM_ENGINE_REFERENCE_REVIEW.md → README.md
- `Grounded Quiz Validation (LLM must quote context evidence)` --semantically_similar_to--> `Hallucination Guard (answer_support / OS_MIN_ANSWER_SUPPORT 0.55)`  [INFERRED] [semantically similar]
  exam_engine/README.md → docs/EVALUATION.md
- `test_missing_model_fails_fast_without_network()` --uses--> `EmbeddingModelUnavailableError`  [INFERRED]
  tests/test_offline.py → ai_engine/embeddings.py
- `test_db_not_locked_while_embedding()` --uses--> `HashingEmbedder`  [INFERRED]
  tests/test_api.py → ai_engine/embeddings.py
- `build_embedder()` --uses--> `Settings`  [INFERRED]
  ai_engine/embeddings.py → backend/config.py

## Import Cycles
- None detected.

## Hyperedges (group relationships)
- **Exam analysis signals feeding the revision plan** — docs_exam_engine_contracts_analyze_past_papers, docs_exam_engine_contracts_syllabus_coverage, docs_exam_engine_contracts_weak_topics, docs_exam_engine_contracts_revision_plan [EXTRACTED 1.00]
- **Layered defence against ungrounded answers** — docs_evaluation_os_min_score, docs_evaluation_prompt_redesign, docs_evaluation_answer_support_guard, docs_api_grounded_answer_response [EXTRACTED 1.00]
- **Local LLM candidates benchmarked** — docs_evaluation_qwen2_5_3b, docs_evaluation_llama3_2_3b, docs_evaluation_phi3_mini [EXTRACTED 1.00]

## Communities (46 total, 7 thin omitted)

### Community 0 - "PDF Extraction & Chunking"
Cohesion: 0.06
Nodes (35): chunk_pages(), _split_page(), TextChunk, _units(), clean_text(), extract_document(), ExtractedDocument, ExtractionError (+27 more)

### Community 1 - "LLM Client & Errors"
Cohesion: 0.06
Nodes (21): EmbeddingModelUnavailableError, LLMClient, LLMUnavailableError, OllamaClient, _prefer_ipv4_loopback(), _body(), register_error_handlers(), _app_error() (+13 more)

### Community 2 - "Ingestion & Retrieval"
Cohesion: 0.14
Nodes (17): ensure_index_consistent(), ingest_document(), _mark_failed(), retrieve_context(), RetrievedChunk, get_services(), Services, set_services() (+9 more)

### Community 3 - "Exam Similarity Scoring"
Cohesion: 0.14
Nodes (17): cosine_matrix(), embed_all(), thresholds_for(), score_matches(), _align(), _key(), _brief(), _cluster() (+9 more)

### Community 4 - "API Regression Tests"
Cohesion: 0.10
Nodes (22): upload(), test_abstention_with_explanation_is_not_grounded(), test_answer_padded_with_outside_knowledge_is_rejected(), test_ask_cites_page_two_for_mitosis(), test_ask_returns_grounded_answer_with_page(), test_bare_lowercase_token_is_an_abstention(), test_delete_removes_document_and_vectors(), test_echoed_page_headers_are_removed_from_answer() (+14 more)

### Community 5 - "SQLite Models & Sessions"
Cohesion: 0.12
Nodes (9): _factory(), get_db(), Base, QuizAnswer, QuizAttempt, TopicPerformance, utcnow(), submit_quiz() (+1 more)

### Community 6 - "CI Workflow"
Cohesion: 0.07
Nodes (29): HF_HUB_OFFLINE, PYTHONUTF8, TRANSFORMERS_OFFLINE, os, python, jobs, pytest, include (+21 more)

### Community 7 - "Exam Topic Analysis"
Cohesion: 0.13
Nodes (13): _as_records(), _paper_key(), _paper_label(), pretty_topic(), calculate_topic_frequency(), _contains(), _cooccurring_keywords(), _discover() (+5 more)

### Community 8 - "Quiz Generation"
Cohesion: 0.13
Nodes (19): build_prompt(), _concept_pool(), _extractive(), _find_evidence(), _finish(), generate_quiz(), grade_quiz(), _normalise_context() (+11 more)

### Community 9 - "Exam Engine Tests"
Cohesion: 0.10
Nodes (20): extract_questions(), parse_syllabus_text(), _check_grounded(), test_average_linkage_does_not_chain_unrelated_questions(), test_brief_example_is_grouped_even_though_wording_differs(), test_course_number_is_not_a_question_number(), test_coverage_statuses_pages_and_explanations(), test_extractive_quiz_is_grounded_and_may_be_short() (+12 more)

### Community 10 - "Exam Engine API Hooks"
Cohesion: 0.11
Nodes (10): RevisionRecommendation, analyze_subject(), _gather_and_analyse(), generate_quiz_for_subject(), recommend_revision(), _subject(), subject(), test_quiz_for_subject_uses_lecture_notes_and_falls_back_without_llm_json() (+2 more)

### Community 11 - "Exam DB Bridge"
Cohesion: 0.11
Nodes (10): build_engine(), _documents(), load_chunks(), load_chunks_for_document(), load_pages(), _merge_overlap(), OllamaQuizLLM, pages_from_chunks() (+2 more)

### Community 12 - "Exam Embedders (TF-IDF/ST)"
Cohesion: 0.13
Nodes (6): _l2(), SentenceTransformerEmbedder, TfidfEmbedder, main(), model_path(), row()

### Community 13 - "Question Extraction"
Cohesion: 0.16
Nodes (15): classify_question(), _doc_year(), _extract_marks(), extract_questions_from_chunks(), _fallback_unnumbered_safe(), _finalise(), _looks_like_question(), _num() (+7 more)

### Community 14 - "API Schemas"
Cohesion: 0.15
Nodes (12): AskRequest, AskResponse, DocType, DocumentOut, DocumentStatus, QuizAnswerIn, QuizResult, QuizSubmit (+4 more)

### Community 15 - "Syllabus Coverage"
Cohesion: 0.16
Nodes (12): get_embedder(), MatchMatrices, analyze_syllabus_coverage(), _best_snippet(), _doc_label(), _empty_row(), _evaluate(), _explain() (+4 more)

### Community 16 - "RAG Answering & Guard"
Cohesion: 0.16
Nodes (11): Answer, answer_question(), answer_support(), build_prompt(), _content_words(), _excerpt(), _is_refusal(), Source (+3 more)

### Community 17 - "Config & Scripts"
Cohesion: 0.16
Nodes (8): get_settings(), strip_running_lines(), download_embedding_model(), main(), pull_ollama_model(), lecture_pages(), main(), test_running_footers_are_removed_from_pages()

### Community 18 - "App Startup & Windows Tests"
Cohesion: 0.16
Nodes (9): build_services(), create_app(), test_index_rebuilt_if_missing(), test_api_reports_missing_model(), _lock_uploads(), test_delete_subject_with_locked_pdf(), test_delete_succeeds_when_pdf_is_locked(), test_startup_sweep_keeps_referenced_uploads() (+1 more)

### Community 19 - "Document Routes & Errors"
Cohesion: 0.21
Nodes (8): AppError, delete_document(), document_status(), get_document(), _get_document(), list_documents(), _resolve_subject(), upload_document()

### Community 21 - "Embedder Interface"
Cohesion: 0.14
Nodes (3): build_embedder(), Embedder, HashingEmbedder

### Community 22 - "Weak-Topic Analysis"
Cohesion: 0.18
Nodes (14): analyze_weak_topics(), _flatten(), add(), _order_with_prerequisites(), visit(), _recommend(), _ts(), _rec() (+6 more)

### Community 23 - "Embedding Model Loading"
Cohesion: 0.16
Nodes (4): SentenceTransformerEmbedder, health(), test_local_model_dir_is_preferred(), test_missing_model_fails_fast_without_network()

### Community 24 - "Exam Engine Integration Bridge"
Cohesion: 0.24
Nodes (6): _call(), get_analysis(), get_revision(), analysis(), _require_subject(), revision()

### Community 25 - "Test Fixtures"
Cohesion: 0.18
Nodes (6): client(), _hermetic_env(), llm(), settings(), no_network(), offline_settings()

### Community 27 - "Evaluation & Model Benchmark"
Cohesion: 0.19
Nodes (9): GET /health, Evaluation with Real Models, llama3.2:3b, Model Comparison (qwen vs llama vs phi3), Offline Verification (socket-blocking wrapper), phi3:mini, qwen2.5:3b (default LLM), OS_* Environment Configuration (+1 more)

### Community 28 - "ExamEngine Facade"
Cohesion: 0.32
Nodes (4): ExamEngine, jsonable(), _analysis(), test_every_engine_output_is_json_serialisable()

### Community 29 - "Benchmark & Example Scripts"
Cohesion: 0.16
Nodes (3): dump(), test_question_file_is_consistent(), test_analysis_endpoint_end_to_end()

### Community 31 - "NLP Text Utilities"
Cohesion: 0.17
Nodes (3): lexical_analyzer(), surface_form(), tokenize()

### Community 32 - "Exam Engine Contracts"
Cohesion: 0.19
Nodes (11): POST /quiz/submit, ExamEngine.analyze_past_papers, ExamEngine.lecture_topics, Chunk/Page Record {document_id, document_name, page_number, text}, parse_syllabus_text, revision_plan, ExamEngine.syllabus_coverage, TopicFrequency / ImportantTopic (+3 more)

### Community 33 - "Reference Repo Review"
Cohesion: 0.20
Nodes (9): exam_engine JSON Contracts, LLM Contract llm(prompt) -> str, make_quiz / grade, SimilarityResult / Group, AnythingLLM, ExamRAG, QPredict, Reference Repository Review (+1 more)

### Community 34 - "Folder Analysis Script"
Cohesion: 0.24
Nodes (6): _clean_syllabus_text(), syllabus_topics_from(), main(), pages_of(), role(), test_web_printed_syllabus_keeps_only_topics()

### Community 35 - "Revision Planning"
Cohesion: 0.25
Nodes (8): _actions(), generate_revision_plan(), _headline(), _reasons(), _schedule(), test_frequent_and_weak_topic_outranks_frequent_but_fine_and_rare_but_uncovered(), test_plan_with_missing_signals_says_so_and_never_crashes(), test_schedule_respects_budget_and_progress_reduces_minutes()

### Community 36 - "Member 1-2 Integration Contract"
Cohesion: 0.29
Nodes (9): GET /analysis/{subject_id} and /revision/{subject_id}, ai_engine.interfaces (extract_document, get_document_chunks, generate_embeddings, retrieve_context), Exam-engine hooks analyze_subject / recommend_revision, Integration Contract (Member 1 <-> Member 2), revision_recommendations table, Shared SQLite Schema (backend/models.py), bridge.py (build_engine, DB readers), exam_engine (Member 2 examination intelligence) (+1 more)

### Community 37 - "Quiz Text Cleaning"
Cohesion: 0.24
Nodes (6): _clean_concept(), _clean_definition(), _clean_sentence(), _clean_term(), _definition_candidates(), test_quiz_skips_fragments_formulas_and_context_dependent_sentences()

### Community 38 - "Subject Routes"
Cohesion: 0.36
Nodes (4): Subject, create_subject(), delete_subject(), list_subjects()

### Community 39 - "API Reference & RAG Pipeline"
Cohesion: 0.22
Nodes (7): API Reference, POST /chat/ask, POST /documents/upload, API Error Codes, Grounded Answer Response (answer, grounded, confidence, sources), Local RAG QA Handbook, Tesseract OCR for image-only pages and slides

### Community 40 - "Dependencies & Test Doubles"
Cohesion: 0.40
Nodes (6): Embedder Contract (encode/embed/callable), TF-IDF Lexical Fallback Matching, Offline Scholar, Deterministic Hashing Embedder and Fake LLM (tests), pytest>=8 (dev dependency), Runtime Dependencies (fastapi, sqlalchemy, pymupdf, sentence-transformers, faiss-cpu, scikit-learn...)

### Community 41 - "Concept Embedder Tests"
Cohesion: 0.33
Nodes (3): ConceptEmbedder, test_dense_embedder_groups_paraphrases_with_no_shared_keywords(), test_dense_semantic_match_counts_without_any_shared_keyword()

## Knowledge Gaps
- **9 isolated node(s):** `Offline Scholar Limitations`, `POST /documents/upload`, `API Error Codes`, `llama3.2:3b`, `phi3:mini` (+4 more)
  These have ≤1 connection - possible missing edges or undocumented components. (Counts symbols only; 290 node(s) total have ≤1 connection when file, concept and rationale nodes are included.)
- **7 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `exam_engine (Member 2 examination intelligence)` connect `Member 1-2 Integration Contract` to `Exam Engine Contracts`, `Reference Repo Review`, `Folder Analysis Script`, `Exam Embedders (TF-IDF/ST)`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Are the 5 inferred relationships involving `create_app()` (e.g. with `Services` and `Settings`) actually correct?**
  _`create_app()` has 5 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Offline Scholar Limitations`, `POST /documents/upload`, `API Error Codes` to the rest of the system?**
  _9 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `PDF Extraction & Chunking` be split into smaller, more focused modules?**
  _Cohesion score 0.05794556628621598 - nodes in this community are weakly interconnected._
- **Why does `Evaluation with Real Models` connect `Evaluation & Model Benchmark` to `LLM Client & Errors`, `Ingestion & Retrieval`, `API Reference & RAG Pipeline`, `Dependencies & Test Doubles`, `Embedding & Coverage Thresholds`?**
  _High betweenness centrality (0.043) - this node is a cross-community bridge._
- **Should `LLM Client & Errors` be split into smaller, more focused modules?**
  _Cohesion score 0.05779220779220779 - nodes in this community are weakly interconnected._
- **Why does `VectorStore` connect `FAISS Vector Store` to `Ingestion & Retrieval`, `App Startup & Windows Tests`, `Embedder Interface`?**
  _High betweenness centrality (0.032) - this node is a cross-community bridge._