import type { ReactNode } from "react";
import { AlertTriangle, BarChart3, ChevronRight, Files, Library, Loader2, MessageSquareText, Target, TrendingDown, TrendingUp } from "lucide-react";
import { Bar, Button, Card, CardHeader, Empty, IconTile, Notes, PageTitle, Pill } from "../components/ui";
import { priorityTone, scoreColor, titleCase } from "../lib/format";
import { useAnalysis, useRevision, useStore } from "../store";
import { go } from "../router";

function greeting() {
  const h = new Date().getHours();
  return h < 12 ? "Good morning" : h < 18 ? "Good afternoon" : "Good evening";
}

function Stat({ label, value, icon, foot, onClick }: { label: string; value: ReactNode; icon: ReactNode; foot: ReactNode; onClick: () => void }) {
  return (
    <button onClick={onClick} className="rounded-xl border border-line bg-white p-5 text-left transition-shadow hover:shadow-sm">
      <div className="flex items-start justify-between gap-3">
        <div>
          <p className="text-[15px] text-muted">{label}</p>
          <p className="mt-2 text-[26px] leading-none font-medium">{value}</p>
        </div>
        <IconTile>{icon}</IconTile>
      </div>
      <div className="mt-4 text-[13px]">{foot}</div>
    </button>
  );
}

const spinner = <Loader2 className="size-6 animate-spin text-muted" />;

export default function Dashboard() {
  const { docs, quizzes, subject, settings } = useStore();
  const analysis = useAnalysis();
  const revision = useRevision();
  const busy = docs.filter((d) => d.status === "pending" || d.status === "processing").length;
  const failed = docs.filter((d) => d.status === "failed").length;
  const cov = analysis.data?.syllabus_coverage?.summary;
  const avg = quizzes.length ? Math.round(quizzes.reduce((s, q) => s + q.score, 0) / quizzes.length) : null;
  const trend = quizzes.length > 1 ? quizzes[quizzes.length - 1].score - quizzes[0].score : null;
  const priorities = revision.data?.priority_topics ?? [];
  const high = priorities.filter((p) => p.priority_label === "High");

  return (
    <>
      <PageTitle
        title={`${greeting()}${settings.name && settings.name !== "Student" ? `, ${settings.name.split(" ")[0]}` : ""}`}
        subtitle={subject?.name}
        right={
          <Button variant="primary" onClick={() => go("assistant")}>
            <MessageSquareText className="size-4" /> AI Study Assistant
          </Button>
        }
      />

      <div className="grid gap-4 sm:grid-cols-2">
        <Stat
          label="Documents"
          value={docs.length}
          icon={<Files className="size-5" />}
          onClick={() => go("documents")}
          foot={
            busy || failed ? (
              <span className="inline-flex items-center gap-1.5 text-amber-800">
                <AlertTriangle className="size-3.5" />
                {[busy && `${busy} processing`, failed && `${failed} failed`].filter(Boolean).join(", ")}
              </span>
            ) : docs.length ? (
              <span className="text-green-700">All documents ready</span>
            ) : (
              <span className="text-muted">Upload notes, past papers and a syllabus</span>
            )
          }
        />
        <Stat
          label="Syllabus coverage"
          value={analysis.loading && !cov ? spinner : cov ? `${Math.round(cov.coverage_pct)}%` : "—"}
          icon={<Library className="size-5" />}
          onClick={() => go("coverage")}
          foot={<span className="text-muted">{cov ? `${cov.covered} covered · ${cov.partially_covered} partial · ${cov.not_covered} missing` : "Needs a syllabus and lecture notes"}</span>}
        />
        <Stat
          label="Average quiz score"
          value={avg === null ? "—" : `${avg}%`}
          icon={<BarChart3 className="size-5" />}
          onClick={() => go("quiz")}
          foot={
            trend === null ? (
              <span className="text-muted">{quizzes.length ? "1 quiz taken" : "No quizzes yet"}</span>
            ) : (
              <span className={`inline-flex items-center gap-1.5 ${trend >= 0 ? "text-green-700" : "text-red-700"}`}>
                {trend >= 0 ? <TrendingUp className="size-3.5" /> : <TrendingDown className="size-3.5" />}
                {`${trend >= 0 ? "Up" : "Down"} ${Math.abs(trend)} points across ${quizzes.length} quizzes`}
              </span>
            )
          }
        />
        <Stat
          label="Topics needing attention"
          value={revision.loading && !revision.data ? spinner : revision.data ? high.length : "—"}
          icon={<Target className="size-5" />}
          onClick={() => go("revision")}
          foot={<span className="text-muted">High priority in your revision plan</span>}
        />
      </div>

      <Card className="mt-6">
        <CardHeader title="Priority topics" subtitle={revision.data?.summary.headline ?? "Frequently examined topics with the most room to improve."} />
        {revision.loading && !revision.data ? (
          <Empty>Building your revision plan…</Empty>
        ) : priorities.length === 0 ? (
          <Empty>{revision.error ?? "Upload past papers and a syllabus to see which topics matter most."}</Empty>
        ) : (
          <ol className="divide-y divide-line">
            {priorities.slice(0, 5).map((t) => (
              <li key={t.topic}>
                <button onClick={() => go("revision")} className="flex w-full items-center gap-4 px-5 py-3.5 text-left hover:bg-neutral-50 sm:px-9">
                  <span className="grid size-7 shrink-0 place-items-center rounded-full bg-neutral-100 text-[13px] text-muted">{t.rank}</span>
                  <span className="min-w-0 flex-1">
                    <span className="flex flex-wrap items-center gap-2 font-medium">
                      {titleCase(t.topic)} <Pill tone={priorityTone(t.priority_label)}>{t.priority_label}</Pill>
                    </span>
                    <span className="mt-0.5 block truncate text-[13px] text-muted">{t.reasons[0]}</span>
                  </span>
                  <Bar value={100 - t.priority_score} color={scoreColor(100 - t.priority_score)} className="hidden w-28 sm:block" />
                  <span className="w-10 text-right font-mono text-[13px]">{Math.round(t.priority_score)}</span>
                  <ChevronRight className="size-4 text-muted" />
                </button>
              </li>
            ))}
          </ol>
        )}
      </Card>

      {analysis.data && <Notes notes={analysis.data.notes} />}

      <Card className="mt-6">
        <CardHeader
          title="Recent quizzes"
          subtitle="Your last three practice sessions on this device."
          right={
            <Button size="sm" onClick={() => go("quiz")}>
              Start a quiz
            </Button>
          }
        />
        {quizzes.length === 0 ? (
          <Empty>No quizzes yet. Quizzes are built from your own lecture notes.</Empty>
        ) : (
          <ul className="divide-y divide-line">
            {quizzes
              .slice(-3)
              .reverse()
              .map((q) => (
                <li key={q.id} className="flex items-center gap-4 px-5 py-3">
                  <span className={`grid size-10 place-items-center rounded-lg font-mono text-sm ${q.score >= 70 ? "bg-green-50 text-green-700" : q.score >= 50 ? "bg-amber-50 text-amber-700" : "bg-red-50 text-red-700"}`}>{q.score}</span>
                  <span className="flex-1">
                    <span className="block font-medium">{q.title}</span>
                    <span className="text-[13px] text-muted">
                      {q.date} · {q.questions} questions
                    </span>
                  </span>
                </li>
              ))}
          </ul>
        )}
      </Card>
    </>
  );
}
