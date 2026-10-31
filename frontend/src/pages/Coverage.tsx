import { useEffect, useRef, useState } from "react";
import { CheckCircle2, ChevronDown, CircleDashed, Search, XCircle } from "lucide-react";
import { Bar, Card, Empty, Notes, PageTitle, Pending, StatusBadge, StatusIcon, cx } from "../components/ui";
import { clean, coverageStatus, titleCase, type CoverageStatus } from "../lib/format";
import { useAnalysis, useStore } from "../store";
import { askAssistant, startFocusedQuiz } from "../router";

type Filter = "all" | CoverageStatus;

export default function Coverage() {
  const { subject, loadAnalysis } = useStore();
  const { data, loading, error } = useAnalysis();
  const [filter, setFilter] = useState<Filter>("all");
  const [query, setQuery] = useState("");
  const [openTopic, setOpenTopic] = useState<string | null>(null);
  const search = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (e.key === "/" && document.activeElement !== search.current) {
        e.preventDefault();
        search.current?.focus();
      }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, []);

  const cov = data?.syllabus_coverage;
  const topics = (cov?.topics ?? []).map((t) => ({ ...t, s: coverageStatus(t.status) }));
  const counts = { covered: cov?.summary.covered ?? 0, partial: cov?.summary.partially_covered ?? 0, missing: cov?.summary.not_covered ?? 0 };
  const total = Math.max(1, cov?.summary.total_topics ?? 0);
  const visible = topics.filter((t) => (filter === "all" || t.s === filter) && t.topic.toLowerCase().includes(query.trim().toLowerCase()));

  const chips: { id: Filter; label: string; n: number }[] = [
    { id: "all", label: "All", n: topics.length },
    { id: "covered", label: "Covered", n: counts.covered },
    { id: "partial", label: "Partial", n: counts.partial },
    { id: "missing", label: "Missing", n: counts.missing },
  ];

  return (
    <>
      <PageTitle title="Syllabus Coverage" subtitle={`See how well your uploaded notes cover the ${subject?.name} syllabus.`} />
      {!data && <Pending loading={loading} error={error} onRetry={() => loadAnalysis(true)} what="syllabus coverage" />}
      {data && !cov && (
        <>
          <Card>
            <Empty>Coverage needs a syllabus and lecture notes. Upload both in Documents.</Empty>
          </Card>
          <Notes notes={data.notes.filter((n) => /syllabus|lecture/i.test(n))} />
        </>
      )}

      {cov && (
        <>
          <Card className="p-6">
            <p className="text-[40px] leading-none font-medium">{Math.round(cov.summary.coverage_pct)}%</p>
            <p className="mt-3 font-medium">Overall syllabus coverage</p>
            <p className="mt-1 text-[13px] text-muted">
              {cov.summary.total_topics} syllabus topics compared with your lecture notes · {data?.matching === "dense" ? "semantic matching" : "word matching"}
            </p>
            <div className="mt-6 flex h-2.5 overflow-hidden rounded-full">
              <div className="bg-good" style={{ width: `${(counts.covered / total) * 100}%` }} />
              <div className="bg-warn" style={{ width: `${(counts.partial / total) * 100}%` }} />
              <div className="bg-bad" style={{ width: `${(counts.missing / total) * 100}%` }} />
            </div>
            <div className="mt-3 grid gap-2 text-[15px] sm:grid-cols-3">
              <span className="flex items-center gap-2">
                <CheckCircle2 className="size-4 text-good" /> Covered <b className="ml-auto pr-4 font-medium">{counts.covered}</b>
              </span>
              <span className="flex items-center gap-2">
                <CircleDashed className="size-4 text-warn" /> Partial <b className="ml-auto pr-4 font-medium">{counts.partial}</b>
              </span>
              <span className="flex items-center gap-2">
                <XCircle className="size-4 text-bad" /> Missing <b className="ml-auto font-medium">{counts.missing}</b>
              </span>
            </div>
          </Card>

          <div className="mt-6 flex flex-wrap gap-2" role="radiogroup" aria-label="Filter by status">
            {chips.map((c) => (
              <button
                key={c.id}
                role="radio"
                aria-checked={filter === c.id}
                onClick={() => setFilter(c.id)}
                className={cx("rounded-full px-3.5 py-2 text-[13px] font-medium", filter === c.id ? "bg-neutral-900 text-white" : "bg-white text-ink hover:bg-neutral-100")}
              >
                {c.label} {c.n}
              </button>
            ))}
          </div>

          <label className="mt-4 flex h-10 items-center gap-2 rounded-lg border border-line bg-white px-3 focus-within:border-accent">
            <Search className="size-4 text-muted" />
            <input ref={search} value={query} onChange={(e) => setQuery(e.target.value)} placeholder="Search syllabus topics" className="flex-1 bg-transparent text-[15px] outline-none" />
          </label>

          <Card className="mt-4">
            {visible.length === 0 ? (
              <Empty>No topics match. Clear the search or pick another filter.</Empty>
            ) : (
              <ul className="divide-y divide-line">
                {visible.map((t) => {
                  const expanded = openTopic === t.topic;
                  const value = t.s === "covered" ? 100 : t.s === "partial" ? 50 : 0;
                  const closest = typeof t.closest_material === "string" ? t.closest_material : t.closest_material?.document ? `${t.closest_material.document}${t.closest_material.pages?.length ? `, page ${t.closest_material.pages.join(", ")}` : ""}` : null;
                  return (
                    <li key={t.topic}>
                      <button aria-expanded={expanded} onClick={() => setOpenTopic(expanded ? null : t.topic)} className="flex w-full items-center gap-3 px-5 py-4 text-left hover:bg-neutral-50">
                        <StatusIcon status={t.s} />
                        <span className="min-w-0 flex-1">
                          <span className="block font-medium">{titleCase(t.topic)}</span>
                          <span className="mt-1 flex flex-wrap items-center gap-3">
                            <StatusBadge status={t.s} />
                            {t.document && (
                              <span className="truncate text-[13px] text-muted">
                                {t.document}
                                {t.pages.length ? `, p. ${t.pages.slice(0, 5).join(", ")}` : ""}
                              </span>
                            )}
                          </span>
                        </span>
                        <Bar value={value} color={t.s === "covered" ? "var(--color-good)" : t.s === "partial" ? "var(--color-warn)" : "var(--color-bad)"} className="hidden w-20 sm:block" />
                        <ChevronDown className={cx("size-4 transition-transform", expanded && "rotate-180")} />
                      </button>
                      {expanded && (
                        <div className="space-y-2 bg-neutral-50 px-5 py-4 text-sm sm:pl-16">
                          <p>{clean(t.explanation)}</p>
                          {closest && t.s !== "covered" && (
                            <p>
                              <span className="text-muted">Closest material:</span> {closest}
                            </p>
                          )}
                          <div className="flex flex-wrap gap-4 pt-1">
                            {t.s !== "covered" && (
                              <a href="#documents" className="font-medium text-accent hover:underline">
                                Upload notes for this topic
                              </a>
                            )}
                            <button onClick={() => askAssistant(`Explain ${t.topic}`)} className="font-medium text-accent hover:underline">
                              Ask the assistant
                            </button>
                            {t.s !== "missing" && (
                              <button onClick={() => startFocusedQuiz(t.topic)} className="font-medium text-accent hover:underline">
                                Quiz me on this
                              </button>
                            )}
                          </div>
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
            )}
          </Card>
        </>
      )}
    </>
  );
}
