import { useState } from "react";
import { Repeat2 } from "lucide-react";
import { Card, CardHeader, Empty, Notes, PageTitle, Pending, Pill, cx } from "../components/ui";
import { clean, titleCase } from "../lib/format";
import { useAnalysis, useStore } from "../store";

export default function PastPapers() {
  const { loadAnalysis } = useStore();
  const { data, loading, error } = useAnalysis();
  const [tab, setTab] = useState<"overview" | "repeated">("overview");
  const past = data?.past_papers;
  const freq = past?.topic_frequency;
  const topics = [...(freq?.topics ?? [])].filter((t) => t.question_count > 0).sort((a, b) => b.question_count - a.question_count);
  const max = Math.max(1, ...topics.map((t) => t.question_count));
  const byYear = Object.entries(
    (past?.questions ?? []).reduce<Record<string, number>>((acc, q) => {
      if (q.year) acc[q.year] = (acc[q.year] ?? 0) + 1;
      return acc;
    }, {}),
  ).sort(([a], [b]) => Number(a) - Number(b));
  const maxYear = Math.max(1, ...byYear.map(([, n]) => n));
  const repeated = past?.repeated_questions ?? [];

  return (
    <>
      <PageTitle
        title="Past-Paper Analysis"
        subtitle="See what examiners actually ask across your analyzed papers."
        right={freq ? <Pill>Based on {freq.total_papers} past paper{freq.total_papers === 1 ? "" : "s"}</Pill> : undefined}
      />

      {!data && <Pending loading={loading} error={error} onRetry={() => loadAnalysis(true)} what="past-paper patterns" />}
      {data && !past && (
        <Card>
          <Empty>No past papers analysed yet. Upload past exam papers as “Past paper” in Documents.</Empty>
        </Card>
      )}
      {data && <Notes notes={data.notes.filter((n) => /past paper|question/i.test(n))} />}

      {past && freq && (
        <>
          <div role="tablist" className="mb-6 flex gap-2 border-b border-line">
            {(
              [
                ["overview", "Overview"],
                ["repeated", `Repeated questions (${repeated.length})`],
              ] as const
            ).map(([id, label]) => (
              <button
                key={id}
                role="tab"
                aria-selected={tab === id}
                onClick={() => setTab(id)}
                className={cx("-mb-px border-b-2 px-4 py-2.5 text-[15px]", tab === id ? "border-accent font-medium text-ink" : "border-transparent text-muted hover:text-ink")}
              >
                {label}
              </button>
            ))}
          </div>

          {tab === "overview" ? (
            <>
              <Card className="grid grid-cols-2">
                {[
                  ["Papers analyzed", freq.total_papers],
                  ["Questions extracted", freq.total_questions],
                  ["Topics found", topics.length],
                  ["Repeated question groups", repeated.length],
                ].map(([label, value], i) => (
                  <div key={label} className={cx("p-5", i % 2 === 0 && "border-r border-line", i < 2 && "border-b border-line")}>
                    <p className="text-[15px] text-muted">{label}</p>
                    <p className="mt-2 text-[26px] font-medium">{value}</p>
                  </div>
                ))}
              </Card>

              <Card className="mt-6">
                <CardHeader title="Most examined topics" subtitle={`Questions per topic across the ${freq.total_papers} analyzed papers. Hover a bar for the reason.`} />
                {topics.length === 0 ? (
                  <Empty>No topics could be matched to the questions.</Empty>
                ) : (
                  <ul className="space-y-3 px-5 py-6">
                    {topics.slice(0, 12).map((t) => (
                      <li key={t.topic} className="flex items-center gap-3" title={t.why}>
                        <span className="w-32 shrink-0 text-right text-[13px] leading-tight sm:w-48">{titleCase(t.topic)}</span>
                        <div className="flex flex-1 items-center gap-2">
                          <div className="h-6 rounded-sm bg-accent transition-[width] duration-500" style={{ width: `${(t.question_count / max) * 100}%` }} />
                          <span className="font-mono text-[13px]">{t.question_count}</span>
                          {t.importance_label === "High" && <Pill tone="bad">High</Pill>}
                        </div>
                      </li>
                    ))}
                  </ul>
                )}
              </Card>

              {byYear.length > 0 && (
                <Card className="mt-6">
                  <CardHeader title="Questions by year" subtitle="Questions extracted from each year's papers." />
                  <div className="flex h-56 items-end gap-4 px-6 pt-6 pb-4">
                    {byYear.map(([y, n]) => (
                      <div key={y} className="flex h-full flex-1 flex-col items-center justify-end gap-2">
                        <span className="font-mono text-[13px]">{n}</span>
                        <div className="w-full max-w-14 rounded-t-sm bg-accent" style={{ height: `${(n / maxYear) * 75}%` }} />
                        <span className="font-mono text-[13px] text-muted">{y}</span>
                      </div>
                    ))}
                  </div>
                </Card>
              )}
            </>
          ) : (
            <Card>
              <CardHeader title="Repeated questions" subtitle="Questions that appeared, in the same or different words, in two or more papers." />
              {repeated.length === 0 ? (
                <Empty>No question appeared in more than one paper.</Empty>
              ) : (
                <ul className="divide-y divide-line">
                  {repeated.map((g) => (
                    <li key={g.group_id} className="flex gap-4 px-5 py-4">
                      <span className="grid size-9 shrink-0 place-items-center rounded-lg bg-accent-soft text-accent">
                        <Repeat2 className="size-4" />
                      </span>
                      <div className="min-w-0 flex-1">
                        <p className="text-[15px]">{clean(g.representative)}</p>
                        <p className="mt-1 text-[13px] text-muted">{g.explanation}</p>
                        <ul className="mt-2 space-y-0.5 text-[13px] text-muted">
                          {g.members.map((m) => (
                            <li key={m.question_id}>
                              <span className="font-mono">{m.year ?? "—"}</span> · {m.document_name}, page {m.page_number}
                            </li>
                          ))}
                        </ul>
                      </div>
                      <span className="font-mono text-[13px] whitespace-nowrap text-muted">{g.paper_count}×</span>
                    </li>
                  ))}
                </ul>
              )}
            </Card>
          )}
        </>
      )}
    </>
  );
}
