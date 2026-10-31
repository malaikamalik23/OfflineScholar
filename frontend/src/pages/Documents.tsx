import { useRef, useState, type DragEvent } from "react";
import { AlertTriangle, CheckCircle2, FileText, Loader2, Trash2, Upload } from "lucide-react";
import { Alert, Button, Card, CardHeader, Empty, PageTitle, Pill, Segmented, cx } from "../components/ui";
import { api, type Doc, type DocType } from "../lib/api";
import { DOC_TYPE_LABEL, mb } from "../lib/format";
import { useStore } from "../store";

const MAX_MB = 50;

interface UploadError {
  id: number;
  title: string;
  message: string;
}

export default function Documents() {
  const { subject, docs, refreshDocs, bumpData } = useStore();
  const [docType, setDocType] = useState<DocType>("lecture");
  const [filter, setFilter] = useState<DocType | "all">("all");
  const [errors, setErrors] = useState<UploadError[]>([]);
  const [uploading, setUploading] = useState(0);
  const [dragging, setDragging] = useState(false);
  const input = useRef<HTMLInputElement>(null);

  async function accept(files: FileList | File[]) {
    if (!subject) return;
    const add = (title: string, message: string) => setErrors((e) => [{ id: Math.random(), title, message }, ...e]);
    for (const f of Array.from(files)) {
      const size = f.size / (1024 * 1024);
      if (!(f.type === "application/pdf" || f.name.toLowerCase().endsWith(".pdf"))) {
        add("Wrong file type.", `“${f.name}” is not a PDF. Choose a PDF file and try again.`);
        continue;
      }
      if (size > MAX_MB) {
        add("File too large.", `“${f.name}” is ${Math.round(size)} MB. The maximum file size is ${MAX_MB} MB.`);
        continue;
      }
      setUploading((n) => n + 1);
      try {
        await api.upload(f, subject.id, docType);
      } catch (err) {
        add("Upload failed.", `“${f.name}”: ${err instanceof Error ? err.message : String(err)}`);
      } finally {
        setUploading((n) => n - 1);
      }
      await refreshDocs().catch(() => undefined);
    }
  }

  async function remove(d: Doc) {
    try {
      await api.deleteDocument(d.id);
      await refreshDocs();
      bumpData();
    } catch (err) {
      setErrors((e) => [{ id: Math.random(), title: "Couldn't remove.", message: err instanceof Error ? err.message : String(err) }, ...e]);
    }
  }

  function onDrop(e: DragEvent) {
    e.preventDefault();
    setDragging(false);
    accept(e.dataTransfer.files);
  }

  const library = docs.filter((d) => filter === "all" || d.doc_type === filter);
  const ready = docs.filter((d) => d.status === "ready").length;

  return (
    <>
      <PageTitle title="Documents" subtitle={`Notes, past papers and syllabus for ${subject?.name}. Everything is processed on this device.`} />

      <Card className="p-5">
        <p className="mb-2 text-[15px] font-medium">
          Document category <span className="text-red-600">*</span>
        </p>
        <Segmented label="Document category" value={docType} onChange={setDocType} options={(Object.keys(DOC_TYPE_LABEL) as DocType[]).map((c) => ({ value: c, label: DOC_TYPE_LABEL[c] }))} />

        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={onDrop}
          className={cx("mt-5 flex flex-col items-center rounded-xl border-2 border-dashed px-4 py-8 text-center transition-colors", dragging ? "border-accent bg-accent-soft" : "border-neutral-300")}
        >
          <div className="grid size-10 place-items-center rounded-lg bg-neutral-100 text-neutral-600">
            {uploading ? <Loader2 className="size-5 animate-spin" /> : <Upload className="size-5" />}
          </div>
          <p className="mt-3 font-medium">{uploading ? `Uploading ${uploading} file${uploading > 1 ? "s" : ""}…` : "Drag PDFs here or browse"}</p>
          <p className="mt-1 text-[13px] text-muted">
            PDF only, up to {MAX_MB} MB · will be added as <b className="font-medium">{DOC_TYPE_LABEL[docType].toLowerCase()}</b>
          </p>
          <Button className="mt-4" onClick={() => input.current?.click()}>
            Browse files
          </Button>
          <input
            ref={input}
            type="file"
            multiple
            accept="application/pdf,.pdf"
            className="hidden"
            onChange={(e) => {
              if (e.target.files) accept(e.target.files);
              e.target.value = "";
            }}
          />
        </div>

        {errors.length > 0 && (
          <div className="mt-4 space-y-3">
            {errors.map((e) => (
              <Alert key={e.id} title={e.title} onClose={() => setErrors((all) => all.filter((x) => x.id !== e.id))}>
                {e.message}
              </Alert>
            ))}
          </div>
        )}
      </Card>

      <Card className="mt-6">
        <CardHeader
          title="Course library"
          subtitle={`${ready} of ${docs.length} documents ready to search`}
          right={
            <Segmented
              label="Filter documents"
              value={filter}
              onChange={setFilter}
              options={[{ value: "all", label: "All" }, ...(Object.keys(DOC_TYPE_LABEL) as DocType[]).map((c) => ({ value: c, label: DOC_TYPE_LABEL[c] }))]}
            />
          }
        />
        {library.length === 0 ? (
          <Empty>{docs.length ? "No documents in this category yet." : "No documents yet. Upload your lecture notes, past papers and syllabus above."}</Empty>
        ) : (
          <ul className="divide-y divide-line">
            {library.map((d) => (
              <DocRow key={d.id} d={d} onRemove={() => remove(d)} />
            ))}
          </ul>
        )}
      </Card>
    </>
  );
}

function DocRow({ d, onRemove }: { d: Doc; onRemove: () => void }) {
  const busy = d.status === "pending" || d.status === "processing";
  return (
    <li className="flex flex-wrap items-center gap-x-4 gap-y-2 px-5 py-3.5">
      <FileText className="size-5 shrink-0 text-neutral-500" />
      <div className="min-w-0 flex-1">
        <p className="truncate font-medium">{d.filename}</p>
        <p className="text-[13px] text-muted">
          {DOC_TYPE_LABEL[d.doc_type]} · {mb(d.size_bytes)}
          {d.status === "ready" && ` · ${d.page_count} pages, ${d.chunk_count} passages${d.ocr_pages ? `, ${d.ocr_pages} read with OCR` : ""}`}
        </p>
        {d.error && <p className="mt-1 text-[13px] text-red-700">{d.error}</p>}
        {d.warnings?.map((w) => (
          <p key={w} className="mt-1 text-[13px] text-amber-800">
            {w}
          </p>
        ))}
      </div>
      {d.status === "ready" && (
        <Pill tone="good">
          <CheckCircle2 className="size-3.5" /> Ready
        </Pill>
      )}
      {busy && (
        <Pill tone="accent">
          <Loader2 className="size-3.5 animate-spin" /> {d.status === "pending" ? "Queued" : "Extracting & indexing"}
        </Pill>
      )}
      {d.status === "failed" && (
        <Pill tone="bad">
          <AlertTriangle className="size-3.5" /> Failed
        </Pill>
      )}
      <button onClick={onRemove} disabled={busy} aria-label={`Remove ${d.filename}`} title={busy ? "Wait for processing to finish" : "Remove"} className="rounded-md p-1.5 text-muted hover:bg-neutral-100 hover:text-red-700 disabled:opacity-40">
        <Trash2 className="size-4" />
      </button>
    </li>
  );
}
