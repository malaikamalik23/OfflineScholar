import { Fragment, useEffect, useRef, useState, type ReactNode } from "react";
import { BookOpen, ChevronDown, ListFilter, Loader2, Plus, Send } from "lucide-react";
import { Button, cx } from "../components/ui";
import { api, type AskResponse } from "../lib/api";
import { clean } from "../lib/format";
import { useStore } from "../store";
import { takePendingQuestion } from "../router";

type Msg = { id: string; role: "user" | "assistant"; text?: string; answer?: AskResponse; error?: string; pending?: boolean };

// Keeps each subject's conversation while you visit other pages.
const saved = new Map<string, Msg[]>();

/** Renders **bold**, `code` and [n] citation markers in an answer. */
function rich(text: string): ReactNode {
  return clean(text)
    .split(/(\*\*[^*]+\*\*|`[^`]+`|\[\d+\])/g)
    .map((part, i) =>
      part.startsWith("**") ? (
        <strong key={i} className="font-semibold">
          {part.slice(2, -2)}
        </strong>
      ) : part.startsWith("`") ? (
        <code key={i} className="font-mono text-[14px]">
          {part.slice(1, -1)}
        </code>
      ) : /^\[\d+\]$/.test(part) ? (
        <span key={i} className="font-mono text-[13px] text-accent">
          {part}
        </span>
      ) : (
        <Fragment key={i}>{part}</Fragment>
      ),
    );
}

/** The API returns only the cited passages, in citation order; label each with the number the answer used. */
function citationNumbers(answer: string, count: number): number[] {
  const cited = [...new Set([...answer.matchAll(/\[(\d+)\]/g)].map((m) => Number(m[1])))].sort((a, b) => a - b);
  return cited.length === count ? cited : Array.from({ length: count }, (_, i) => i + 1);
}

function SourceList({ sources, answer }: { sources: AskResponse["sources"]; answer: string }) {
  const [open, setOpen] = useState<number | null>(null);
  const numbers = citationNumbers(answer, sources.length);
  return (
    <ol className="mt-4 space-y-1 border-t border-line pt-3 text-[13px]">
      {sources.map((s, i) => (
        <li key={i}>
          <button onClick={() => setOpen(open === i ? null : i)} className="flex w-full items-center gap-2 text-left text-muted hover:text-ink">
            <span className="font-mono">[{numbers[i]}]</span> {s.document_name}, page {s.page_number}
            <ChevronDown className={cx("size-3.5 transition-transform", open === i && "rotate-180")} />
          </button>
          {open === i && <p className="mt-1 mb-2 rounded-md bg-neutral-50 px-3 py-2 text-neutral-700">{clean(s.excerpt)}</p>}
        </li>
      ))}
    </ol>
  );
}

export default function Assistant() {
  const { docs, settings, subject } = useStore();
  const ready = docs.filter((d) => d.status === "ready");
  const key = String(subject?.id);
  const [messages, setMessages] = useState<Msg[]>(() => saved.get(key) ?? []);
  const [draft, setDraft] = useState("");
  const [allSubjects, setAllSubjects] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  useEffect(() => setMessages(saved.get(key) ?? []), [key]);
  useEffect(() => {
    saved.set(key, messages);
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [messages, key]);

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (e.key === "/" && !["TEXTAREA", "INPUT", "SELECT"].includes(document.activeElement?.tagName ?? "")) {
        e.preventDefault();
        box.current?.focus();
      }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, []);

  const busy = messages.some((m) => m.pending);

  async function ask(q: string) {
    const question = q.trim();
    if (question.length < 3 || busy || !subject) return;
    const id = crypto.randomUUID();
    setMessages((m) => [...m, { id: id + "q", role: "user", text: question }, { id, role: "assistant", pending: true }]);
    setDraft("");
    try {
      const answer = await api.ask(question, allSubjects ? null : subject.id);
      setMessages((m) => m.map((x) => (x.id === id ? { id, role: "assistant", answer } : x)));
    } catch (err) {
      setMessages((m) => m.map((x) => (x.id === id ? { id, role: "assistant", error: err instanceof Error ? err.message : String(err) } : x)));
    }
  }

  // A question handed over from another page (e.g. the Revision Plan) is asked right away.
  useEffect(() => {
    const q = takePendingQuestion();
    if (q) ask(q);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  return (
    <div className="flex min-h-0 flex-1 flex-col">
      <div className="flex flex-wrap items-start justify-between gap-3">
        <div>
          <h1 className="text-[26px] font-semibold tracking-tight">Study Assistant</h1>
          <p className="mt-1 text-[15px] text-muted">Answers grounded in your notes, processed entirely on this device.</p>
        </div>
        <Button size="sm" onClick={() => setMessages([])} disabled={busy || messages.length === 0}>
          <Plus className="size-4" /> New chat
        </Button>
      </div>

      <div className="mt-4 min-h-0 flex-1 overflow-y-auto" aria-live="polite">
        {messages.length === 0 && (
          <div className="flex flex-col items-center py-12 text-center">
            <span className="grid size-12 place-items-center rounded-xl bg-accent-soft text-accent">
              <BookOpen className="size-6" />
            </span>
            <p className="mt-4 font-medium">{ready.length ? `Ask anything about your ${ready.length} ready documents` : "Upload documents first, then ask about them"}</p>
            <p className="mt-1 max-w-md text-sm text-muted">Answers only use your uploaded material and cite the document and page. If your notes don't cover something, you'll be told so.</p>
          </div>
        )}

        {messages.map((m) =>
          m.role === "user" ? (
            <div key={m.id} className="flex justify-end py-4">
              <p className="max-w-[80%] rounded-xl bg-neutral-100 px-4 py-3 text-[15px]">{m.text}</p>
            </div>
          ) : (
            <div key={m.id} className="flex gap-3 border-t border-line bg-white px-4 py-5">
              <span className="relative grid size-7 shrink-0 place-items-center rounded-md bg-accent text-white">
                <BookOpen className="size-4" />
              </span>
              <div className="min-w-0 flex-1">
                <p className="font-semibold">Offline Scholar</p>
                {m.pending ? (
                  <p className="mt-2 inline-flex items-center gap-2 text-[15px] text-muted">
                    <Loader2 className="size-4 animate-spin" /> Searching your notes and asking the local model… (can take 10–30 s)
                  </p>
                ) : m.error ? (
                  <p className="mt-2 text-[15px] text-red-700">{m.error}</p>
                ) : m.answer ? (
                  <>
                    <div className={cx("mt-2 text-[15px] leading-7 whitespace-pre-wrap", !m.answer.grounded && "text-muted")}>{rich(m.answer.answer)}</div>
                    {settings.showSources && m.answer.grounded && m.answer.sources.length > 0 && <SourceList sources={m.answer.sources} answer={m.answer.answer} />}
                  </>
                ) : null}
              </div>
            </div>
          ),
        )}
        <div ref={end} />
      </div>

      <div className="mt-3 shrink-0">
        <label className="mb-2 flex items-center gap-2 text-[13px] text-muted">
          <ListFilter className="size-4" /> Searching in:
          <select value={allSubjects ? "all" : "subject"} onChange={(e) => setAllSubjects(e.target.value === "all")} className="rounded-md bg-transparent py-0.5 font-medium text-ink hover:bg-neutral-100">
            <option value="subject">{subject?.name} ({ready.length} ready documents)</option>
            <option value="all">All subjects</option>
          </select>
        </label>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            ask(draft);
          }}
          className="rounded-xl border border-line bg-white p-3 focus-within:border-accent"
        >
          <textarea
            ref={box}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                ask(draft);
              }
            }}
            rows={2}
            placeholder="Ask a question about your course..."
            aria-label="Ask a question about your course"
            className="w-full resize-none bg-transparent text-[15px] outline-none placeholder:text-neutral-500"
          />
          <div className="flex items-center justify-between gap-3">
            <span className="text-[13px] text-muted">Enter to send · Shift+Enter for a new line</span>
            <Button type="submit" variant="primary" size="sm" disabled={draft.trim().length < 3 || busy}>
              <Send className="size-3.5" /> Send
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
}
