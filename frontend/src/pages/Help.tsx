import { useEffect, useRef, useState } from "react";
import { ArrowLeft, ChevronDown, CircleHelp, ClipboardList, Files, MessageSquareText, Rocket, Search } from "lucide-react";
import { Card, CardHeader, Empty, PageTitle, cx } from "../components/ui";
import { HELP_AREAS, SHORTCUTS, type HelpArea } from "../helpData";

const ICONS = { rocket: Rocket, files: Files, chat: MessageSquareText, clipboard: ClipboardList };

export default function Help() {
  const [query, setQuery] = useState("");
  const [area, setArea] = useState<HelpArea | null>(null);
  const [open, setOpen] = useState<string | null>(null);
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

  const q = query.trim().toLowerCase();
  const results = q
    ? HELP_AREAS.flatMap((a) => a.articles.filter((x) => (x.title + " " + x.body).toLowerCase().includes(q)).map((x) => ({ ...x, area: a.title })))
    : [];
  const shortcutHits = q ? SHORTCUTS.filter((s) => (s.action + " " + s.keys.join(" ")).toLowerCase().includes(q)) : SHORTCUTS;

  const Article = ({ id, title, body, sub }: { id: string; title: string; body: string; sub?: string }) => (
    <li>
      <button aria-expanded={open === id} onClick={() => setOpen(open === id ? null : id)} className="flex w-full items-center gap-3 px-5 py-3.5 text-left hover:bg-neutral-50">
        <span className="flex-1">
          <span className="block font-medium">{title}</span>
          {sub && <span className="text-[13px] text-muted">{sub}</span>}
        </span>
        <ChevronDown className={cx("size-4 transition-transform", open === id && "rotate-180")} />
      </button>
      {open === id && <p className="px-5 pb-4 text-[15px] leading-6 text-muted">{body}</p>}
    </li>
  );

  return (
    <>
      <PageTitle title="Help & Shortcuts" subtitle="Find answers, explore features, and learn shortcuts to make your study experience easier." />

      <label className="flex h-11 items-center gap-3 rounded-lg border border-line bg-white px-4 focus-within:border-accent">
        <Search className="size-4 text-muted" />
        <input
          ref={search}
          value={query}
          onChange={(e) => {
            setQuery(e.target.value);
            setArea(null);
          }}
          placeholder="Search help articles, features, or shortcuts..."
          className="flex-1 bg-transparent text-[15px] outline-none"
        />
        <CircleHelp className="size-4 text-muted" />
      </label>

      {q ? (
        <Card className="mt-6">
          <CardHeader title={`Results for “${query.trim()}”`} subtitle={`${results.length} article${results.length === 1 ? "" : "s"}`} />
          {results.length ? (
            <ul className="divide-y divide-line">
              {results.map((r) => (
                <Article key={r.id} id={r.id} title={r.title} body={r.body} sub={r.area} />
              ))}
            </ul>
          ) : (
            <Empty>No articles match. Try a shorter word like “quiz” or “upload”.</Empty>
          )}
        </Card>
      ) : area ? (
        <Card className="mt-6">
          <div className="border-b border-line px-5 py-4">
            <button onClick={() => setArea(null)} className="mb-2 inline-flex items-center gap-1 text-[13px] text-muted hover:text-ink">
              <ArrowLeft className="size-3.5" /> Quick help
            </button>
            <h2 className="text-[17px] font-semibold">{area.title}</h2>
            <p className="mt-1 text-sm text-muted">{area.blurb}</p>
          </div>
          <ul className="divide-y divide-line">
            {area.articles.map((a) => (
              <Article key={a.id} {...a} />
            ))}
          </ul>
        </Card>
      ) : (
        <>
          <h2 className="mt-8 text-xl font-semibold">Quick help</h2>
          <p className="mt-1 text-[15px] text-muted">Explore guidance by feature area.</p>
          <div className="mt-4 grid gap-4 sm:grid-cols-2">
            {HELP_AREAS.map((a) => {
              const Icon = ICONS[a.icon];
              return (
                <button key={a.id} onClick={() => setArea(a)} className="rounded-xl border border-line bg-white p-5 text-left transition-shadow hover:shadow-sm">
                  <span className="grid size-10 place-items-center rounded-lg bg-neutral-100">
                    <Icon className="size-5" />
                  </span>
                  <h3 className="mt-6 text-[17px] font-semibold">{a.title}</h3>
                  <p className="mt-2 text-[15px] leading-6 text-muted">{a.blurb}</p>
                  <p className="mt-5 font-mono text-[13px] text-muted">{a.articles.length} articles</p>
                </button>
              );
            })}
          </div>
        </>
      )}

      {shortcutHits.length > 0 && (
        <Card className="mt-6">
          <CardHeader title="Keyboard shortcuts" />
          <ul className="divide-y divide-line">
            {shortcutHits.map((s) => (
              <li key={s.action} className="flex items-center gap-4 px-5 py-3 text-[15px]">
                <span className="flex-1">{s.action}</span>
                <span className="flex gap-1">
                  {s.keys.map((k) => (
                    <kbd key={k} className="rounded-md border border-line bg-neutral-50 px-2 py-0.5 font-mono text-[12px]">
                      {k}
                    </kbd>
                  ))}
                </span>
              </li>
            ))}
          </ul>
        </Card>
      )}
    </>
  );
}
