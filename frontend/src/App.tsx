import { useEffect, useRef, useState, type FormEvent } from "react";
import { BookOpen, Check, ChevronDown, Plus, WifiOff } from "lucide-react";
import { StoreProvider, useStore } from "./store";
import { ROUTES, go, useRoute, type Route } from "./router";
import { Button, Card, cx } from "./components/ui";
import Dashboard from "./pages/Dashboard";
import Documents from "./pages/Documents";
import Assistant from "./pages/Assistant";
import PastPapers from "./pages/PastPapers";
import Coverage from "./pages/Coverage";
import Quiz from "./pages/Quiz";
import Revision from "./pages/Revision";
import SettingsPage from "./pages/Settings";
import Help from "./pages/Help";

const PAGES: Record<Route, () => JSX.Element> = {
  dashboard: Dashboard,
  documents: Documents,
  assistant: Assistant,
  papers: PastPapers,
  coverage: Coverage,
  quiz: Quiz,
  revision: Revision,
  settings: SettingsPage,
  help: Help,
};

export default function App() {
  return (
    <StoreProvider>
      <Shell />
    </StoreProvider>
  );
}

function Shell() {
  const route = useRoute();
  const { subject, subjects, backendDown } = useStore();
  const Page = PAGES[route];
  const main = useRef<HTMLElement>(null);

  useEffect(() => {
    main.current?.scrollTo(0, 0);
    document.title = `${ROUTES.find((r) => r.id === route)!.title} · Offline Scholar`;
  }, [route]);

  // Alt+1..7 jumps between main sections, Alt+H opens help.
  useEffect(() => {
    const on = (e: KeyboardEvent) => {
      if (!e.altKey || e.ctrlKey || e.metaKey) return;
      const n = Number(e.key);
      if (n >= 1 && n <= 7) {
        e.preventDefault();
        go(ROUTES[n - 1].id);
      } else if (e.key.toLowerCase() === "h") {
        e.preventDefault();
        go("help");
      }
    };
    window.addEventListener("keydown", on);
    return () => window.removeEventListener("keydown", on);
  }, []);

  const needsSubject = !subject && route !== "help" && route !== "settings";

  return (
    <div className="flex h-full">
      <Rail route={route} />
      <div className="flex min-w-0 flex-1 flex-col">
        <Header title={ROUTES.find((r) => r.id === route)!.title} />
        {backendDown && (
          <div role="alert" className="flex items-center gap-2 border-b border-red-200 bg-red-50 px-6 py-2 text-sm text-red-800">
            <WifiOff className="size-4" /> Can't reach the Offline Scholar backend. Start it and this page will reconnect.
          </div>
        )}
        <main ref={main} className="flex-1 overflow-y-auto">
          <div className={cx("mx-auto w-full px-4 py-8 sm:px-6", route === "assistant" && !needsSubject ? "flex h-full max-w-5xl flex-col py-6" : "max-w-5xl")}>
            {needsSubject ? <FirstSubject hasSubjects={subjects.length > 0} /> : <Page />}
          </div>
        </main>
      </div>
    </div>
  );
}

function FirstSubject({ hasSubjects }: { hasSubjects: boolean }) {
  const { createSubject } = useStore();
  const [name, setName] = useState("");
  const [error, setError] = useState<string | null>(null);
  async function submit(e: FormEvent) {
    e.preventDefault();
    try {
      await createSubject(name.trim());
      go("documents");
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }
  return (
    <Card className="mx-auto mt-10 max-w-lg p-8 text-center">
      <span className="mx-auto grid size-12 place-items-center rounded-xl bg-accent-soft text-accent">
        <BookOpen className="size-6" />
      </span>
      <h1 className="mt-4 text-2xl font-semibold">{hasSubjects ? "Choose a subject" : "Welcome to Offline Scholar"}</h1>
      <p className="mt-2 text-muted">
        {hasSubjects ? "Pick one from the menu at the top, or create a new one." : "Create your first subject, then upload its lecture notes, past papers and syllabus."}
      </p>
      <form onSubmit={submit} className="mt-6 flex gap-2">
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="e.g. Machine Learning" aria-label="Subject name" className="h-10 flex-1 rounded-lg border border-line px-3 outline-none focus:border-accent" />
        <Button variant="primary" type="submit" disabled={name.trim().length < 1}>
          Create subject
        </Button>
      </form>
      {error && <p className="mt-3 text-sm text-red-700">{error}</p>}
    </Card>
  );
}

function Rail({ route }: { route: Route }) {
  const top = ROUTES.slice(0, 7);
  const bottom = ROUTES.slice(7);
  const Item = ({ r }: { r: (typeof ROUTES)[number] }) => (
    <a
      href={`#${r.id}`}
      title={r.label}
      aria-label={r.label}
      aria-current={route === r.id ? "page" : undefined}
      className={cx("grid size-11 place-items-center rounded-lg transition-colors", route === r.id ? "bg-neutral-100 text-ink" : "text-neutral-600 hover:bg-neutral-50 hover:text-ink")}
    >
      <r.icon className="size-5" strokeWidth={1.75} />
    </a>
  );
  return (
    <nav className="flex w-[60px] shrink-0 flex-col items-center border-r border-line bg-white sm:w-[72px]" aria-label="Main">
      <a href="#dashboard" aria-label="Offline Scholar home" className="flex h-16 w-full items-center justify-center border-b border-line">
        <span className="relative grid size-8 place-items-center rounded-lg bg-accent text-white">
          <BookOpen className="size-[18px]" strokeWidth={2} />
          <span className="absolute -right-0.5 -bottom-0.5 size-2 rounded-full border-2 border-accent bg-white" />
        </span>
      </a>
      <div className="flex flex-1 flex-col items-center gap-0.5 pt-3">
        {top.map((r) => (
          <Item key={r.id} r={r} />
        ))}
      </div>
      <div className="flex w-full flex-col items-center gap-0.5 border-t border-line py-3">
        {bottom.map((r) => (
          <Item key={r.id} r={r} />
        ))}
      </div>
    </nav>
  );
}

function ModelPill() {
  const { health, backendDown } = useStore();
  const ok = !backendDown && health?.llm.reachable && health.llm.model_installed && health.embeddings.available;
  const label = backendDown ? "Backend offline" : !health ? "Checking…" : ok ? "Local model ready" : !health.llm.reachable ? "Ollama not running" : "Model not installed";
  return (
    <span
      title={health ? `${health.llm.model}${health.offline_mode ? " · offline mode" : ""}` : undefined}
      className={cx(
        "hidden items-center gap-1.5 rounded-md border px-2 py-1 text-[13px] font-medium sm:inline-flex",
        ok ? "border-green-200 bg-green-50 text-green-800" : "border-amber-200 bg-amber-50 text-amber-800",
      )}
    >
      <span className={cx("size-2 rounded-full", ok ? "bg-green-600" : "bg-amber-500")} /> {label}
    </span>
  );
}

function Header({ title }: { title: string }) {
  const { settings } = useStore();
  const initials = settings.name.split(/\s+/).map((w) => w[0]).join("").slice(0, 2).toUpperCase();
  return (
    <header className="flex h-16 shrink-0 items-center gap-3 border-b border-line bg-white px-4 sm:px-6">
      <p className="hidden min-w-[150px] text-[17px] font-semibold md:block">{title}</p>
      <SubjectMenu />
      <div className="ml-auto flex items-center gap-3">
        <ModelPill />
        <a href="#settings" title={`${settings.name} · Settings`} className="grid size-8 place-items-center rounded-full bg-neutral-900 text-[13px] font-semibold text-white">
          {initials || "?"}
        </a>
      </div>
    </header>
  );
}

function SubjectMenu() {
  const { subjects, subject, selectSubject, createSubject } = useStore();
  const [open, setOpen] = useState(false);
  const [adding, setAdding] = useState("");
  const [error, setError] = useState<string | null>(null);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const close = (e: MouseEvent) => !ref.current?.contains(e.target as Node) && setOpen(false);
    const esc = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
    document.addEventListener("mousedown", close);
    document.addEventListener("keydown", esc);
    return () => {
      document.removeEventListener("mousedown", close);
      document.removeEventListener("keydown", esc);
    };
  }, [open]);
  async function add(e: FormEvent) {
    e.preventDefault();
    try {
      await createSubject(adding.trim());
      setAdding("");
      setOpen(false);
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    }
  }
  return (
    <div ref={ref} className="relative min-w-0 md:mx-auto">
      <button
        onClick={() => setOpen((o) => !o)}
        aria-haspopup="listbox"
        aria-expanded={open}
        className="flex h-10 max-w-[60vw] min-w-0 items-center gap-2 rounded-lg border border-line bg-white px-4 text-[15px] shadow-sm hover:bg-neutral-50"
      >
        <span className="truncate">{subject?.name ?? "No subject yet"}</span>
        <ChevronDown className="size-4 shrink-0 text-muted" />
      </button>
      {open && (
        <div className="absolute top-12 left-0 z-20 w-80 rounded-lg border border-line bg-white p-1 shadow-lg">
          <div role="listbox">
            {subjects.map((s) => (
              <button
                key={s.id}
                role="option"
                aria-selected={s.id === subject?.id}
                onClick={() => {
                  selectSubject(s.id);
                  setOpen(false);
                }}
                className="flex w-full items-center gap-2 rounded-md px-3 py-2 text-left text-sm hover:bg-neutral-50"
              >
                <span className="truncate">{s.name}</span>
                {s.id === subject?.id && <Check className="ml-auto size-4 text-accent" />}
              </button>
            ))}
          </div>
          <form onSubmit={add} className="mt-1 flex gap-1 border-t border-line p-2">
            <input value={adding} onChange={(e) => setAdding(e.target.value)} placeholder="New subject name" aria-label="New subject name" className="h-8 flex-1 rounded-md border border-line px-2 text-sm outline-none focus:border-accent" />
            <Button size="sm" variant="primary" type="submit" disabled={!adding.trim()} aria-label="Create subject">
              <Plus className="size-4" />
            </Button>
          </form>
          {error && <p className="px-3 pb-2 text-[13px] text-red-700">{error}</p>}
        </div>
      )}
    </div>
  );
}
