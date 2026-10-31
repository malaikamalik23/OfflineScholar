import { createContext, useCallback, useContext, useEffect, useMemo, useRef, useState, type ReactNode } from "react";
import { api, ApiError, type Analysis, type Doc, type Health, type Revision, type Subject } from "./lib/api";

export interface Settings {
  name: string;
  quizLength: number;
  showSources: boolean;
}

export interface QuizHistoryItem {
  id: string;
  subjectId: number;
  title: string;
  date: string;
  score: number; // 0-100
  questions: number;
}

interface Saved {
  subjectId: number | null;
  settings: Settings;
  quizzes: QuizHistoryItem[];
  doneSessions: string[]; // "subjectId:day:topic" ticked in the revision schedule
}

const KEY = "offline-scholar-ui-v2";
const DEFAULTS: Saved = { subjectId: null, settings: { name: "Student", quizLength: 5, showSources: true }, quizzes: [], doneSessions: [] };

function load(): Saved {
  try {
    const raw = localStorage.getItem(KEY);
    if (raw) {
      const s = JSON.parse(raw) as Partial<Saved>;
      return { ...DEFAULTS, ...s, settings: { ...DEFAULTS.settings, ...s.settings } };
    }
  } catch {
    /* storage unavailable */
  }
  return DEFAULTS;
}

export interface Loadable<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

interface Store {
  health: Health | null;
  backendDown: boolean;
  subjects: Subject[];
  subject: Subject | null;
  selectSubject: (id: number) => void;
  createSubject: (name: string) => Promise<Subject>;
  docs: Doc[];
  refreshDocs: () => Promise<void>;
  /** Bumps whenever documents finish processing, are deleted, or a quiz is submitted. */
  dataVersion: number;
  bumpData: () => void;
  analysis: Loadable<Analysis>;
  revision: Loadable<Revision>;
  loadAnalysis: (force?: boolean) => void;
  loadRevision: (force?: boolean) => void;
  settings: Settings;
  updateSettings: (p: Partial<Settings>) => void;
  quizzes: QuizHistoryItem[];
  addQuiz: (q: QuizHistoryItem) => void;
  doneSessions: string[];
  toggleSession: (key: string) => void;
  clearLocal: () => void;
}

const Ctx = createContext<Store | null>(null);
const msg = (e: unknown) => (e instanceof ApiError || e instanceof Error ? e.message : String(e));

export function StoreProvider({ children }: { children: ReactNode }) {
  const [saved, setSaved] = useState<Saved>(load);
  const [health, setHealth] = useState<Health | null>(null);
  const [backendDown, setBackendDown] = useState(false);
  const [subjects, setSubjects] = useState<Subject[]>([]);
  const [docs, setDocs] = useState<Doc[]>([]);
  const [dataVersion, setDataVersion] = useState(0);
  // `sid` records which subject a result (or a load in progress) belongs to.
  type Slot<T> = Loadable<T> & { key?: string; sid?: number };
  const [analysisState, setAnalysis] = useState<Slot<Analysis>>({ data: null, loading: false, error: null });
  const [revisionState, setRevision] = useState<Slot<Revision>>({ data: null, loading: false, error: null });

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(saved));
    } catch {
      /* ignore */
    }
  }, [saved]);

  // Backend health: drives the "Local model ready" pill.
  useEffect(() => {
    let alive = true;
    const check = () =>
      api
        .health()
        .then((h) => alive && (setHealth(h), setBackendDown(false)))
        .catch(() => alive && setBackendDown(true));
    check();
    const t = setInterval(check, 30000);
    return () => {
      alive = false;
      clearInterval(t);
    };
  }, []);

  useEffect(() => {
    api
      .subjects()
      .then((list) => {
        setSubjects(list);
        setSaved((s) => (s.subjectId && list.some((x) => x.id === s.subjectId) ? s : { ...s, subjectId: list[0]?.id ?? null }));
      })
      .catch(() => setBackendDown(true));
  }, []);

  const subjectId = saved.subjectId;
  const subject = subjects.find((s) => s.id === subjectId) ?? null;

  const docsRef = useRef<Doc[] | null>(null);
  const refreshDocs = useCallback(async () => {
    if (!subjectId) {
      docsRef.current = [];
      return setDocs([]);
    }
    const list = await api.documents(subjectId);
    // A document finishing (or disappearing) changes what analysis/revision can say.
    const ready = (l: Doc[]) => l.filter((d) => d.status === "ready").map((d) => d.id).join();
    if (docsRef.current !== null && ready(docsRef.current) !== ready(list)) setDataVersion((v) => v + 1);
    docsRef.current = list;
    setDocs(list);
  }, [subjectId]);

  useEffect(() => {
    docsRef.current = null;
    setDocs([]);
    refreshDocs().catch(() => setBackendDown(true));
  }, [refreshDocs]);

  // Poll while anything is still processing (the API processes uploads in the background).
  const busy = docs.some((d) => d.status === "pending" || d.status === "processing");
  useEffect(() => {
    if (!busy) return;
    const t = setInterval(() => refreshDocs().catch(() => undefined), 1500);
    return () => clearInterval(t);
  }, [busy, refreshDocs]);

  const cacheKey = `${subjectId}:${dataVersion}`;
  const inflight = useRef<Record<string, boolean>>({});

  const loadAnalysis = useCallback(
    (force = false) => {
      if (!subjectId || (!force && (analysisState.key === cacheKey || inflight.current["a" + cacheKey]))) return;
      inflight.current["a" + cacheKey] = true;
      setAnalysis((a) => ({ ...(a.sid === subjectId ? a : { data: null }), loading: true, error: null, sid: subjectId }));
      api
        .analysis(subjectId)
        .then((data) => setAnalysis({ data, loading: false, error: null, key: cacheKey, sid: subjectId }))
        .catch((e) => setAnalysis({ data: null, loading: false, error: msg(e), key: cacheKey, sid: subjectId }))
        .finally(() => delete inflight.current["a" + cacheKey]);
    },
    [subjectId, cacheKey, analysisState.key],
  );

  const loadRevision = useCallback(
    (force = false) => {
      if (!subjectId || (!force && (revisionState.key === cacheKey || inflight.current["r" + cacheKey]))) return;
      inflight.current["r" + cacheKey] = true;
      setRevision((r) => ({ ...(r.sid === subjectId ? r : { data: null }), loading: true, error: null, sid: subjectId }));
      api
        .revision(subjectId)
        .then((data) => setRevision({ data, loading: false, error: null, key: cacheKey, sid: subjectId }))
        .catch((e) => setRevision({ data: null, loading: false, error: msg(e), key: cacheKey, sid: subjectId }))
        .finally(() => delete inflight.current["r" + cacheKey]);
    },
    [subjectId, cacheKey, revisionState.key],
  );

  // Never show another subject's results (e.g. right after switching subject).
  const empty = { data: null, loading: false, error: null };
  const analysis: Loadable<Analysis> = analysisState.sid === subjectId ? analysisState : empty;
  const revision: Loadable<Revision> = revisionState.sid === subjectId ? revisionState : empty;

  const store = useMemo<Store>(
    () => ({
      health,
      backendDown,
      subjects,
      subject,
      selectSubject: (id) => setSaved((s) => ({ ...s, subjectId: id })),
      createSubject: async (name) => {
        const s = await api.createSubject(name);
        setSubjects((list) => [...list, s]);
        setSaved((x) => ({ ...x, subjectId: s.id }));
        return s;
      },
      docs,
      refreshDocs,
      dataVersion,
      bumpData: () => setDataVersion((v) => v + 1),
      analysis,
      revision,
      loadAnalysis,
      loadRevision,
      settings: saved.settings,
      updateSettings: (p) => setSaved((s) => ({ ...s, settings: { ...s.settings, ...p } })),
      quizzes: saved.quizzes.filter((q) => q.subjectId === subjectId),
      addQuiz: (q) => setSaved((s) => ({ ...s, quizzes: [...s.quizzes, q] })),
      doneSessions: saved.doneSessions,
      toggleSession: (key) =>
        setSaved((s) => ({ ...s, doneSessions: s.doneSessions.includes(key) ? s.doneSessions.filter((k) => k !== key) : [...s.doneSessions, key] })),
      clearLocal: () => setSaved((s) => ({ ...DEFAULTS, subjectId: s.subjectId, settings: s.settings })),
    }),
    [health, backendDown, subjects, subject, docs, refreshDocs, dataVersion, analysis, revision, loadAnalysis, loadRevision, saved, subjectId],
  );

  return <Ctx.Provider value={store}>{children}</Ctx.Provider>;
}

export function useStore() {
  const s = useContext(Ctx);
  if (!s) throw new Error("useStore must be used inside StoreProvider");
  return s;
}

/** Fetches the subject's analysis the first time a page needs it (results are cached until data changes). */
export function useAnalysis() {
  const { analysis, loadAnalysis, subject } = useStore();
  useEffect(() => {
    if (subject) loadAnalysis();
  }, [subject, loadAnalysis]);
  return analysis;
}

export function useRevision() {
  const { revision, loadRevision, subject } = useStore();
  useEffect(() => {
    if (subject) loadRevision();
  }, [subject, loadRevision]);
  return revision;
}
