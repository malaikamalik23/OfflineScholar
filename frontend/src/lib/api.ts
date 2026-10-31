// Typed client for the Offline Scholar backend (docs/API.md). Same origin in the desktop app;
// the Vite dev server proxies /api to :8000.

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`/api${path}`, init);
  } catch {
    throw new ApiError(0, "backend_unreachable", "Can't reach the Offline Scholar backend. Is it running?");
  }
  if (res.status === 204) return undefined as T;
  const body = await res.json().catch(() => null);
  if (!res.ok) {
    const err = body?.error;
    const detail = Array.isArray(body?.detail) ? body.detail.map((d: { msg: string }) => d.msg).join("; ") : undefined;
    throw new ApiError(res.status, err?.code ?? "http_error", err?.message ?? detail ?? `Request failed (HTTP ${res.status}).`);
  }
  return body as T;
}

const json = (method: string, data: unknown): RequestInit => ({
  method,
  headers: { "content-type": "application/json" },
  body: JSON.stringify(data),
});

// ---------- types ----------

export type DocType = "lecture" | "past_paper" | "syllabus";
export type DocStatus = "pending" | "processing" | "ready" | "failed";

export interface Subject {
  id: number;
  name: string;
}

export interface Doc {
  id: number;
  subject_id: number;
  filename: string;
  doc_type: DocType;
  status: DocStatus;
  error: string | null;
  warnings: string[];
  page_count: number;
  chunk_count: number;
  ocr_pages: number;
  size_bytes: number;
  created_at: string;
}

export interface Source {
  document_name: string;
  page_number: number;
  excerpt: string;
  score?: number | null;
}

export interface AskResponse {
  answer: string;
  grounded: boolean;
  sources: Source[];
  confidence: number;
}

export interface Health {
  status: "ok" | "degraded";
  llm: { reachable: boolean; model: string; model_installed: boolean };
  embeddings: { model: string; available: boolean };
  ocr: { enabled: boolean; tesseract_installed: boolean };
  offline_mode: boolean;
}

export interface TopicFrequency {
  topic: string;
  question_count: number;
  paper_count: number;
  years: number[];
  importance_score: number;
  importance_label: string;
  why: string;
  one_off: boolean;
  example_questions?: unknown[];
}

export interface ExamQuestion {
  question_id: string;
  text: string;
  document_name: string;
  page_number: number;
  year: number | null;
}

export interface RepeatedGroup {
  group_id: string;
  representative: string;
  paper_count: number;
  years: number[];
  match_type: string;
  explanation: string;
  members: (ExamQuestion & { similarity_to_representative?: number })[];
}

export type CoverageStatusText = "Covered" | "Partially covered" | "Not covered";

export interface CoverageTopic {
  topic: string;
  status: CoverageStatusText;
  document: string | null;
  pages: number[];
  evidence_score: number;
  explanation: string;
  closest_material: { document?: string; pages?: number[] } | string | null;
}

export interface Analysis {
  subject: Subject;
  matching: "dense" | "lexical";
  documents: { past_paper_pages: number; syllabus_pages: number; lecture_pages: number; quiz_answers: number };
  past_papers: null | {
    questions: ExamQuestion[];
    repeated_questions: RepeatedGroup[];
    topic_frequency: { total_questions: number; total_papers: number; topics: TopicFrequency[] };
  };
  syllabus_coverage: null | {
    topics: CoverageTopic[];
    summary: { total_topics: number; covered: number; partially_covered: number; not_covered: number; coverage_pct: number };
  };
  weak_topics: null | { topics?: { topic: string; accuracy?: number; is_weak?: boolean }[] };
  notes: string[];
}

export interface PriorityTopic {
  topic: string;
  rank: number;
  priority_score: number;
  priority_label: "High" | "Medium" | "Low" | string;
  signals: { exam_importance: number | null; weakness: number | null; coverage_gap: number | null };
  suggested_minutes: number;
  reasons: string[];
  actions: string[];
  not_seen_in_past_papers: boolean;
}

export interface Revision {
  subject: Subject;
  summary: { topics_in_plan: number; days: number; minutes_per_day: number; signals_used: string[]; headline: string };
  priority_topics: PriorityTopic[];
  schedule: { day: number; minutes: number; sessions: { topic: string; minutes: number }[] }[];
  notes: string[];
}

export interface QuizQuestion {
  question_id: string;
  topic: string;
  question: string;
  options: { key: string; text: string }[];
  correct_option: string;
  correct_answer: string;
  explanation: string;
  difficulty: string;
  source: { document_name: string; page_number: number; evidence: string };
}

export interface GeneratedQuiz {
  topic: string;
  questions: QuizQuestion[];
  generation_report: { requested: number; generated: number; notes: string[] };
}

export interface QuizResult {
  attempt_id: number;
  total: number;
  correct: number;
  score: number;
  topics: { topic: string; attempts: number; correct: number; accuracy: number }[];
}

// ---------- calls ----------

export const api = {
  health: () => request<Health>("/health"),

  subjects: () => request<Subject[]>("/subjects"),
  createSubject: (name: string) => request<Subject>("/subjects", json("POST", { name })),
  deleteSubject: (id: number) => request<void>(`/subjects/${id}`, { method: "DELETE" }),

  documents: (subjectId: number) => request<Doc[]>(`/documents?subject_id=${subjectId}`),
  upload: (file: File, subjectId: number, docType: DocType) => {
    const form = new FormData();
    form.append("file", file);
    form.append("subject_id", String(subjectId));
    form.append("doc_type", docType);
    return request<Doc>("/documents/upload", { method: "POST", body: form });
  },
  deleteDocument: (id: number) => request<void>(`/documents/${id}`, { method: "DELETE" }),

  ask: (question: string, subjectId: number | null) =>
    request<AskResponse>("/chat/ask", json("POST", { question, subject_id: subjectId, top_k: 4 })),

  analysis: (subjectId: number) => request<Analysis>(`/analysis/${subjectId}`),
  revision: (subjectId: number) => request<Revision>(`/revision/${subjectId}`),

  generateQuiz: (subjectId: number, topic: string, n: number) =>
    request<GeneratedQuiz>("/quiz/generate", json("POST", { subject_id: subjectId, topic, n_questions: n })),
  submitQuiz: (subjectId: number, answers: { topic: string; correct: boolean; question_ref?: string }[]) =>
    request<QuizResult>("/quiz/submit", json("POST", { subject_id: subjectId, answers })),
};
