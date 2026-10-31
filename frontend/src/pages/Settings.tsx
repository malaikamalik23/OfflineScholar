import { useState } from "react";
import { Cpu, FileSearch, ScanText, ShieldCheck } from "lucide-react";
import { Button, Card, CardHeader, PageTitle, Pill } from "../components/ui";
import { useStore } from "../store";

export default function SettingsPage() {
  const { settings, updateSettings, clearLocal, health, backendDown } = useStore();
  const [confirm, setConfirm] = useState(false);
  const field = "mt-1.5 h-10 w-full rounded-lg border border-line bg-white px-3 text-[15px] outline-none focus:border-accent";
  const ok = !backendDown && health?.status === "ok";

  return (
    <>
      <PageTitle title="Settings" subtitle="Your profile, study preferences and the local AI setup." />

      <Card>
        <CardHeader title="Profile & study" />
        <div className="grid gap-5 p-5 sm:grid-cols-2">
          <label className="text-sm font-medium">
            Your name
            <input className={field} value={settings.name} onChange={(e) => updateSettings({ name: e.target.value })} />
          </label>
          <label className="text-sm font-medium">
            Default quiz length
            <select className={field} value={settings.quizLength} onChange={(e) => updateSettings({ quizLength: Number(e.target.value) })}>
              {[3, 5, 10].map((n) => (
                <option key={n} value={n}>
                  {n} questions
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-3 text-sm font-medium">
            <input type="checkbox" className="size-4 accent-[var(--color-accent)]" checked={settings.showSources} onChange={(e) => updateSettings({ showSources: e.target.checked })} />
            Show sources under assistant answers
          </label>
        </div>
      </Card>

      <Card className="mt-6">
        <CardHeader
          title="Local AI"
          subtitle="Answers, quizzes and document processing run on this device."
          right={<Pill tone={ok ? "good" : "warn"}>{backendDown ? "Backend offline" : !health ? "Checking…" : ok ? "Ready" : "Needs attention"}</Pill>}
        />
        {health && (
          <dl className="grid gap-4 p-5 text-sm sm:grid-cols-2">
            <div className="flex gap-3">
              <Cpu className="size-5 text-muted" />
              <div>
                <dt className="text-muted">Language model (Ollama)</dt>
                <dd className="font-medium">
                  {health.llm.model} · {!health.llm.reachable ? "Ollama not running" : health.llm.model_installed ? "installed" : `not installed: run ollama pull ${health.llm.model}`}
                </dd>
              </div>
            </div>
            <div className="flex gap-3">
              <FileSearch className="size-5 text-muted" />
              <div>
                <dt className="text-muted">Search model</dt>
                <dd className="font-medium break-all">
                  {health.embeddings.model.split(/[\\/]/).pop()} · {health.embeddings.available ? "available" : "missing: run scripts/download_models.py"}
                </dd>
              </div>
            </div>
            <div className="flex gap-3">
              <ScanText className="size-5 text-muted" />
              <div>
                <dt className="text-muted">OCR for scanned pages</dt>
                <dd className="font-medium">{!health.ocr.enabled ? "Turned off" : health.ocr.tesseract_installed ? "Tesseract installed" : "Tesseract not found (scanned pages will be skipped)"}</dd>
              </div>
            </div>
            <div className="flex gap-3">
              <ShieldCheck className="size-5 text-good" />
              <div>
                <dt className="text-muted">Network use</dt>
                <dd className="font-medium">{health.offline_mode ? "Offline mode: no network access" : "Offline mode off (set OS_OFFLINE=1)"}</dd>
              </div>
            </div>
          </dl>
        )}
      </Card>

      <Card className="mt-6">
        <CardHeader title="Clear history on this device" subtitle="Removes quiz history and schedule ticks stored by this app. Your documents and saved quiz results in the backend stay." />
        <div className="flex flex-wrap items-center gap-3 p-5">
          {confirm ? (
            <>
              <span className="text-sm">Clear local history?</span>
              <Button
                className="border-red-300 text-red-700 hover:bg-red-50"
                onClick={() => {
                  clearLocal();
                  setConfirm(false);
                }}
              >
                Yes, clear
              </Button>
              <Button variant="ghost" onClick={() => setConfirm(false)}>
                Cancel
              </Button>
            </>
          ) : (
            <Button onClick={() => setConfirm(true)}>Clear local history</Button>
          )}
        </div>
      </Card>
    </>
  );
}
