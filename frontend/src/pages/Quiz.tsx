import { useEffect, useState } from "react";
import { CheckCircle2, Loader2, ShieldCheck, Target, XCircle } from "lucide-react";
import { Bar, Button, Card, Segmented, cx } from "../components/ui";
import { api, type QuizQuestion, type QuizResult } from "../lib/api";
import { clean, scoreColor, titleCase } from "../lib/format";
import { useRevision, useStore } from "../store";
import { go, takePendingQuizTopic } from "../router";

const COUNTS = [3, 5, 10];

type Phase =
  | { kind: "setup" }
  | { kind: "loading" }
  | { kind: "run"; topic: string; qs: QuizQuestion[]; i: number; picked: string | null; answers: boolean[] }
  | { kind: "done"; topic: string; qs: QuizQuestion[]; answers: boolean[]; saved: QuizResult | null; saveError: string | null };

export default function Quiz() {
  const { subject, settings, addQuiz, bumpData } = useStore();
  const revision = useRevision();
  const [count, setCount] = useState(String(COUNTS.includes(settings.quizLength) ? settings.quizLength : 5));
  const [topic, setTopic] = useState("");
  const [phase, setPhase] = useState<Phase>({ kind: "setup" });
  const [error, setError] = useState<string | null>(null);

  // Arriving from "Quiz me on this" / a revision action pre-fills the topic.
  useEffect(() => {
    const t = takePendingQuizTopic();
    if (t) setTopic(t);
  }, []);

  const suggestions = (revision.data?.priority_topics ?? []).slice(0, 8).map((p) => p.topic);

  async function start() {
    if (!subject || topic.trim().length < 2) return;
    setError(null);
    setPhase({ kind: "loading" });
    try {
      const quiz = await api.generateQuiz(subject.id, topic.trim(), Number(count));
      if (!quiz.questions.length) {
        setError(quiz.generation_report.notes[0] ?? "No questions could be made for this topic from your lecture notes. Try another topic or upload notes that cover it.");
        setPhase({ kind: "setup" });
        return;
      }
      setPhase({ kind: "run", topic: quiz.topic, qs: quiz.questions, i: 0, picked: null, answers: [] });
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
      setPhase({ kind: "setup" });
    }
  }

  function choose(key: string) {
    if (phase.kind !== "run" || phase.picked !== null) return;
    setPhase({ ...phase, picked: key, answers: [...phase.answers, key === phase.qs[phase.i].correct_option] });
  }

  async function next() {
    if (phase.kind !== "run" || !subject) return;
    if (phase.i + 1 < phase.qs.length) return setPhase({ ...phase, i: phase.i + 1, picked: null });
    const { topic: t, qs, answers } = phase;
    const correct = answers.filter(Boolean).length;
    addQuiz({
      id: crypto.randomUUID(),
      subjectId: subject.id,
      title: titleCase(t),
      date: new Date().toLocaleDateString("en-US", { month: "short", day: "numeric" }),
      score: Math.round((correct / qs.length) * 100),
      questions: qs.length,
    });
    setPhase({ kind: "done", topic: t, qs, answers, saved: null, saveError: null });
    try {
      const saved = await api.submitQuiz(subject.id, qs.map((q, i) => ({ topic: t, correct: answers[i], question_ref: q.question_id })));
      bumpData(); // weak topics and the revision plan now include this result
      setPhase((p) => (p.kind === "done" ? { ...p, saved } : p));
    } catch (err) {
      setPhase((p) => (p.kind === "done" ? { ...p, saveError: err instanceof Error ? err.message : String(err) } : p));
    }
  }

  // Number keys 1-4 pick an answer, Enter moves on.
  useEffect(() => {
    if (phase.kind !== "run") return;
    const on = (e: KeyboardEvent) => {
      if (e.ctrlKey || e.altKey || e.metaKey) return;
      const n = Number(e.key);
      const q = phase.qs[phase.i];
      if (n >= 1 && n <= q.options.length) choose(q.options[n - 1].key);
      else if (e.key === "Enter" && phase.picked !== null) next();
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  });

  if (phase.kind === "loading")
    return (
      <Card className="mx-auto mt-10 max-w-xl p-8 text-center">
        <Loader2 className="mx-auto size-8 animate-spin text-accent" />
        <p className="mt-4 font-medium">Writing {count} questions on “{topic}” from your lecture notes…</p>
        <p className="mt-1 text-sm text-muted">The local model takes roughly 10–20 seconds per question on a laptop CPU.</p>
      </Card>
    );

  if (phase.kind === "run") {
    const q = phase.qs[phase.i];
    return (
      <div className="mx-auto max-w-2xl">
        <div className="mb-3 flex items-center justify-between text-[13px] text-muted">
          <span>
            Question {phase.i + 1} of {phase.qs.length} · {titleCase(phase.topic)}
          </span>
          <button onClick={() => setPhase({ kind: "setup" })} className="hover:text-ink">
            End quiz
          </button>
        </div>
        <Bar value={(phase.i / phase.qs.length) * 100} color="var(--color-accent)" />
        <Card className="mt-5 p-6">
          <h1 className="text-xl font-semibold">{clean(q.question)}</h1>
          <div className="mt-5 space-y-2.5" role="radiogroup" aria-label="Answers">
            {q.options.map((o, n) => {
              const chosen = phase.picked === o.key;
              const right = phase.picked !== null && o.key === q.correct_option;
              const wrong = chosen && o.key !== q.correct_option;
              return (
                <button
                  key={o.key}
                  role="radio"
                  aria-checked={chosen}
                  disabled={phase.picked !== null}
                  onClick={() => choose(o.key)}
                  className={cx(
                    "flex w-full items-center gap-3 rounded-lg border px-4 py-3 text-left text-[15px] transition-colors disabled:cursor-default",
                    right ? "border-green-300 bg-green-50" : wrong ? "border-red-300 bg-red-50" : "border-line hover:bg-neutral-50",
                  )}
                >
                  <span className="grid size-6 shrink-0 place-items-center rounded-md bg-neutral-100 font-mono text-[13px]">{n + 1}</span>
                  <span className="flex-1">{clean(o.text)}</span>
                  {right && <CheckCircle2 className="size-5 text-good" />}
                  {wrong && <XCircle className="size-5 text-bad" />}
                </button>
              );
            })}
          </div>
          {phase.picked !== null && (
            <div className="mt-5">
              <p className="text-sm">
                <span className={phase.picked === q.correct_option ? "font-medium text-green-700" : "font-medium text-red-700"}>{phase.picked === q.correct_option ? "Correct." : `Not quite: the answer is ${q.correct_option}.`}</span>{" "}
                <span className="text-muted">{clean(q.explanation)}</span>
              </p>
              <p className="mt-1 text-[13px] text-muted">
                Source: {q.source.document_name}, page {q.source.page_number}
              </p>
              <div className="mt-4 flex justify-end">
                <Button variant="primary" onClick={next} autoFocus>
                  {phase.i + 1 < phase.qs.length ? "Next question" : "See results"}
                </Button>
              </div>
            </div>
          )}
        </Card>
      </div>
    );
  }

  if (phase.kind === "done") {
    const correct = phase.answers.filter(Boolean).length;
    const score = Math.round((correct / phase.qs.length) * 100);
    return (
      <div className="mx-auto max-w-2xl">
        <Card className="p-6 text-center">
          <p className="text-[13px] text-muted">Quiz complete · {titleCase(phase.topic)}</p>
          <p className="mt-2 text-5xl font-medium">{score}%</p>
          <p className="mt-2 text-muted">
            {correct} of {phase.qs.length} correct.{" "}
            {phase.saved ? "Saved: your revision plan now includes this result." : phase.saveError ? "" : "Saving…"}
          </p>
          {phase.saveError && <p className="mt-2 text-sm text-red-700">Couldn't save the result: {phase.saveError}</p>}
          <div className="mt-5 flex justify-center gap-3">
            <Button onClick={() => setPhase({ kind: "setup" })}>New quiz</Button>
            <Button variant="primary" onClick={() => go("revision")}>
              View revision plan
            </Button>
          </div>
        </Card>
        <Card className="mt-5">
          <ul className="divide-y divide-line">
            {phase.qs.map((q, i) => (
              <li key={q.question_id} className="flex items-start gap-3 px-5 py-3 text-sm">
                {phase.answers[i] ? <CheckCircle2 className="mt-0.5 size-4 shrink-0 text-good" /> : <XCircle className="mt-0.5 size-4 shrink-0 text-bad" />}
                <span className="flex-1">
                  {clean(q.question)}
                  <span className="block text-[13px] text-muted">
                    {clean(q.correct_answer)} · {q.source.document_name}, p. {q.source.page_number}
                  </span>
                </span>
              </li>
            ))}
          </ul>
        </Card>
        {phase.saved && (
          <Card className="mt-5 p-5">
            <p className="text-sm font-medium">Topic accuracy so far</p>
            {phase.saved.topics.map((t) => (
              <div key={t.topic} className="mt-2 flex items-center gap-3 text-sm">
                <span className="flex-1">{titleCase(t.topic)}</span>
                <span className="font-mono text-[13px] text-muted">
                  {t.correct}/{t.attempts}
                </span>
                <Bar value={t.accuracy * 100} color={scoreColor(t.accuracy * 100)} className="w-24" />
              </div>
            ))}
          </Card>
        )}
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-2xl pt-6">
      <div className="text-center">
        <span className="mx-auto grid size-11 place-items-center rounded-xl bg-accent-soft text-accent">
          <Target className="size-6" />
        </span>
        <h1 className="mt-5 text-[26px] font-semibold">Practice quiz</h1>
        <p className="mt-2 text-[15px] text-muted">Build a focused quiz from your locally stored course materials.</p>
      </div>
      <Card className="mt-7 p-6">
        <label htmlFor="quiz-topic" className="font-semibold">
          Topic
        </label>
        <input
          id="quiz-topic"
          value={topic}
          onChange={(e) => setTopic(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && start()}
          placeholder="e.g. Support vector machines"
          className="mt-2 h-10 w-full rounded-lg border border-line px-3 text-[15px] outline-none focus:border-accent"
        />
        {suggestions.length > 0 && (
          <div className="mt-3 flex flex-wrap gap-2">
            <span className="text-[13px] text-muted">From your revision plan:</span>
            {suggestions.map((s) => (
              <button key={s} onClick={() => setTopic(s)} className={cx("rounded-full border px-3 py-1 text-[13px]", topic === s ? "border-accent bg-accent-soft text-accent" : "border-line bg-white hover:bg-neutral-50")}>
                {titleCase(s)}
              </button>
            ))}
          </div>
        )}
        <hr className="my-6 border-line" />
        <h2 className="mb-3 font-semibold">Number of questions</h2>
        <Segmented
          label="Number of questions"
          stretch
          value={count}
          onChange={setCount}
          options={COUNTS.map((c) => ({
            value: String(c),
            label: (
              <>
                <span className="font-mono">{c}</span> Questions
              </>
            ),
          }))}
        />
        <p className="mt-5 flex items-start gap-2.5 rounded-lg border border-line bg-neutral-50 px-4 py-3 text-[13px] leading-5 text-muted">
          <ShieldCheck className="mt-0.5 size-4 shrink-0 text-good" /> Questions are written by the local model from your lecture notes only, each with its source page, and processed entirely on this device.
        </p>
        {error && <p className="mt-4 text-sm text-red-700">{error}</p>}
        <Button variant="primary" className="mt-5 h-11 w-full" onClick={start} disabled={topic.trim().length < 2}>
          Start quiz
        </Button>
      </Card>
    </div>
  );
}
