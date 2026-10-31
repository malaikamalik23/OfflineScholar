import { useState } from "react";
import { BarChart3, Check, ChevronRight, List, MessageSquareText, Target } from "lucide-react";
import { Bar, Button, Card, CardHeader, Empty, Notes, PageTitle, Pending, Pill, Segmented, cx } from "../components/ui";
import type { PriorityTopic } from "../lib/api";
import { priorityTone, scoreColor, titleCase } from "../lib/format";
import { useRevision, useStore } from "../store";
import { askAssistant, startFocusedQuiz } from "../router";

/** 0-100: how ready the student is on a topic (quiz results if any, otherwise how well notes cover it). */
const readiness = (t: PriorityTopic) => Math.round(100 * (1 - (t.signals.weakness ?? t.signals.coverage_gap ?? 0)));
const importance = (t: PriorityTopic) => Math.round(100 * (t.signals.exam_importance ?? 0));

export default function Revision() {
  const { loadRevision } = useStore();
  const { data, loading, error } = useRevision();
  const [view, setView] = useState<"matrix" | "list">("matrix");
  const topics = data?.priority_topics ?? [];
  const high = topics.filter((t) => t.priority_label === "High");

  return (
    <>
      <PageTitle
        eyebrow={<span className="mb-3 inline-block rounded-md border border-blue-200 bg-accent-soft px-2 py-1 text-[13px] text-accent">Personalized from your quizzes, notes and past-paper analysis</span>}
        title="Adaptive Revision Plan"
        subtitle={data ? `${high.length} topic${high.length === 1 ? " needs" : "s need"} attention before your exam. ${data.summary.headline}` : undefined}
      />
      {!data && <Pending loading={loading} error={error} onRetry={() => loadRevision(true)} what="your revision plan" />}
      {data && topics.length === 0 && (
        <Card>
          <Empty>Nothing to plan yet. Upload past papers, a syllabus and lecture notes, or take a quiz.</Empty>
        </Card>
      )}
      {data && <Notes notes={data.notes} />}

      {topics.length > 0 && (
        <>
          <Card className="mt-6">
            <CardHeader
              title="Where to focus"
              subtitle="Exam importance compared with how ready you are (quiz results, or notes coverage until you take quizzes)."
              right={
                <Segmented
                  label="View"
                  value={view}
                  onChange={setView}
                  options={[
                    { value: "matrix", label: <><BarChart3 className="size-4" /> Matrix</> },
                    { value: "list", label: <><List className="size-4" /> List</> },
                  ]}
                />
              }
            />
            {view === "matrix" ? <Matrix topics={topics} /> : <PlanList topics={topics} />}
          </Card>
          <Priorities topics={topics.slice(0, 4)} />
          <Schedule />
        </>
      )}
    </>
  );
}

const W = 720, H = 460, L = 70, R = 16, T = 30, B = 50;
const x = (v: number) => L + (v / 100) * (W - L - R);
const y = (v: number) => T + (1 - v / 100) * (H - T - B);

function Matrix({ topics }: { topics: PriorityTopic[] }) {
  const [hover, setHover] = useState<string | null>(null);
  return (
    <div className="overflow-x-auto px-2 py-4">
      <svg viewBox={`0 0 ${W} ${H}`} className="min-w-[560px]" role="img" aria-label="Topics plotted by exam importance and readiness">
        <rect x={x(50)} y={y(50)} width={x(100) - x(50)} height={y(0) - y(50)} fill="var(--color-accent)" opacity={0.09} stroke="var(--color-accent)" strokeOpacity={0.25} />
        <text x={x(100) - 6} y={y(0) - 8} textAnchor="end" fontSize="13" fill="var(--color-accent)">
          High priority
        </text>
        {[0, 20, 40, 60, 80, 100].map((v) => (
          <g key={v}>
            <line x1={L} x2={W - R} y1={y(v)} y2={y(v)} stroke="#ddd" strokeDasharray="4 4" />
            <text x={L - 10} y={y(v) + 4} textAnchor="end" fontSize="12" fontFamily="var(--font-mono)" fill="#666">
              {v}%
            </text>
            <line x1={x(v)} x2={x(v)} y1={T} y2={H - B} stroke="#ddd" strokeDasharray="4 4" />
            <text x={x(v)} y={H - B + 18} textAnchor="middle" fontSize="12" fontFamily="var(--font-mono)" fill="#666">
              {v}
            </text>
          </g>
        ))}
        <text x={(L + W - R) / 2} y={H - 6} textAnchor="middle" fontSize="13" fill="#666">
          Exam importance (past papers)
        </text>
        <text transform={`translate(18 ${(T + H - B) / 2}) rotate(-90)`} textAnchor="middle" fontSize="13" fill="#666">
          Readiness
        </text>
        {topics.map((t) => {
          const hi = t.priority_label === "High";
          const cx_ = x(importance(t)), cy = y(readiness(t));
          const show = hi || hover === t.topic;
          const words = titleCase(t.topic).split(" ");
          return (
            <g key={t.topic} onMouseEnter={() => setHover(t.topic)} onMouseLeave={() => setHover(null)}>
              <title>{`${titleCase(t.topic)}: importance ${importance(t)}, readiness ${readiness(t)}%`}</title>
              <circle cx={cx_} cy={cy} r={hover === t.topic ? 7 : 5} fill={hi ? "var(--color-accent)" : "#888"} />
              {show &&
                words.slice(0, 3).map((w, i) => (
                  <text key={i} x={cx_} y={cy - 12 - (Math.min(words.length, 3) - 1 - i) * 12} textAnchor="middle" fontSize="11" fontWeight={hi ? 600 : 400} fill="#111">
                    {w}
                  </text>
                ))}
            </g>
          );
        })}
      </svg>
    </div>
  );
}

function PlanList({ topics }: { topics: PriorityTopic[] }) {
  return (
    <ol className="divide-y divide-line">
      {topics.map((t) => (
        <li key={t.topic} className="flex flex-wrap items-center gap-x-4 gap-y-2 px-5 py-3.5">
          <span className="grid size-7 shrink-0 place-items-center rounded-full bg-neutral-100 text-[13px] text-muted">{t.rank}</span>
          <div className="min-w-0 flex-1">
            <p className="flex flex-wrap items-center gap-2 font-medium">
              {titleCase(t.topic)} <Pill tone={priorityTone(t.priority_label)}>{t.priority_label}</Pill>
            </p>
            <p className="text-[13px] text-muted">{t.reasons.join(" · ")}</p>
          </div>
          <Bar value={readiness(t)} color={scoreColor(readiness(t))} className="w-20" />
          <span className="w-16 text-right font-mono text-[13px]">{t.suggested_minutes} min</span>
        </li>
      ))}
    </ol>
  );
}

function Tag({ children }: { children: string }) {
  return <span className="rounded-md bg-neutral-100 px-2.5 py-1 font-mono text-[13px] text-neutral-600">{children}</span>;
}

function Priorities({ topics }: { topics: PriorityTopic[] }) {
  return (
    <section className="mt-10">
      <h2 className="text-xl font-semibold">Revision priorities</h2>
      <p className="mt-1 text-[15px] text-muted">Ranked recommendations from your quizzes, notes and past-paper analysis.</p>
      <ol className="mt-4 space-y-4">
        {topics.map((t) => {
          const studied = (t.signals.coverage_gap ?? 1) < 1;
          return (
            <li key={t.topic}>
              <Card className="flex gap-4 p-5 sm:gap-6 sm:p-7">
                <span className="grid size-11 shrink-0 place-items-center rounded-full bg-accent-soft text-accent">{t.rank}</span>
                <div className="min-w-0">
                  <h3 className="text-lg font-semibold">{titleCase(t.topic)}</h3>
                  <div className="mt-2 flex flex-wrap gap-2">
                    <Tag>{`Priority ${t.priority_label}`}</Tag>
                    {t.signals.exam_importance !== null && <Tag>{`Exam importance ${importance(t)}%`}</Tag>}
                    <Tag>{t.signals.weakness !== null ? `Mastery ${readiness(t)}%` : t.signals.coverage_gap === null ? "Notes not checked" : t.signals.coverage_gap >= 1 ? "No notes found" : t.signals.coverage_gap > 0 ? "Partial notes" : "Notes covered"}</Tag>
                    <Tag>{`~${t.suggested_minutes} min`}</Tag>
                  </div>
                  <p className="mt-4 text-[15px] text-neutral-700">{t.reasons.join(". ")}.</p>
                  {t.actions[0] && <p className="mt-1 text-[15px] text-muted">{t.actions[0]}</p>}
                  <div className="mt-4 flex flex-wrap gap-2">
                    <Button variant="primary" className="h-11 px-5" onClick={() => askAssistant(`Explain ${t.topic}`)}>
                      <MessageSquareText className="size-4" /> Study with the assistant <ChevronRight className="size-4" />
                    </Button>
                    {studied && (
                      <Button className="h-11 px-5" onClick={() => startFocusedQuiz(t.topic)}>
                        <Target className="size-4" /> Take a focused quiz
                      </Button>
                    )}
                  </div>
                </div>
              </Card>
            </li>
          );
        })}
      </ol>
    </section>
  );
}

function Schedule() {
  const { revision, subject, doneSessions, toggleSession } = useStore();
  const plan = revision.data;
  if (!plan || !subject) return null;
  const keys = plan.schedule.flatMap((d) => d.sessions.map((s) => `${subject.id}:${d.day}:${s.topic}`));
  const done = keys.filter((k) => doneSessions.includes(k)).length;
  return (
    <Card className="mt-8">
      <div className="flex flex-wrap items-start justify-between gap-4 border-b border-line px-5 py-5 sm:px-7">
        <div>
          <h2 className="text-xl font-semibold">Study schedule</h2>
          <p className="mt-1 text-[15px] text-muted">
            {plan.summary.days} days, about {plan.summary.minutes_per_day} minutes a day.
          </p>
        </div>
        <div className="w-full sm:w-72">
          <div className="flex justify-between text-[15px]">
            <span className="text-muted">Plan progress</span>
            <span className="font-mono font-medium">
              {done} of {keys.length} done
            </span>
          </div>
          <Bar value={keys.length ? (done / keys.length) * 100 : 0} color="var(--color-accent)" className="mt-2 h-2" />
        </div>
      </div>
      {plan.schedule.map((d) => (
        <div key={d.day} className="border-b border-line px-5 py-5 last:border-b-0 sm:px-7">
          <h3 className="text-[13px] font-medium tracking-wide text-muted uppercase">
            Day {d.day} · {d.minutes} min
          </h3>
          <ul className="mt-3 space-y-2">
            {d.sessions.map((s) => {
              const key = `${subject.id}:${d.day}:${s.topic}`;
              const isDone = doneSessions.includes(key);
              return (
                <li key={key} className={cx("flex flex-wrap items-center gap-3 rounded-xl border border-line bg-neutral-50 px-4 py-2.5", isDone && "opacity-70")}>
                  <button
                    role="checkbox"
                    aria-checked={isDone}
                    aria-label={`Mark ${s.topic} on day ${d.day} as ${isDone ? "not done" : "done"}`}
                    onClick={() => toggleSession(key)}
                    className={cx("grid size-8 shrink-0 place-items-center rounded-lg border", isDone ? "border-accent bg-accent text-white" : "border-line bg-white hover:border-neutral-400")}
                  >
                    {isDone && <Check className="size-4" strokeWidth={3} />}
                  </button>
                  <span className={cx("min-w-0 flex-1 text-[15px]", isDone && "text-muted line-through")}>{titleCase(s.topic)}</span>
                  <span className="font-mono text-[13px] text-muted">{s.minutes} min</span>
                  <Button size="sm" onClick={() => askAssistant(`Explain ${s.topic}`)}>
                    Study <ChevronRight className="size-3.5" />
                  </Button>
                </li>
              );
            })}
          </ul>
        </div>
      ))}
    </Card>
  );
}
