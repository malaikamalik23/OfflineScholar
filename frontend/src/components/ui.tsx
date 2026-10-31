import type { ReactNode, ButtonHTMLAttributes } from "react";
import { CheckCircle2, CircleAlert, CircleDashed, XCircle } from "lucide-react";
import type { CoverageStatus } from "../lib/format";

export const cx = (...c: (string | false | null | undefined)[]) => c.filter(Boolean).join(" ");

export function Card({ children, className }: { children: ReactNode; className?: string }) {
  return <div className={cx("rounded-xl border border-line bg-white", className)}>{children}</div>;
}

export function CardHeader({ title, subtitle, right }: { title: string; subtitle?: string; right?: ReactNode }) {
  return (
    <div className="flex flex-wrap items-start justify-between gap-3 border-b border-line px-5 py-4">
      <div>
        <h2 className="text-[17px] font-semibold">{title}</h2>
        {subtitle && <p className="mt-1 text-sm text-muted">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

export function PageTitle({ title, subtitle, right, eyebrow }: { title: string; subtitle?: ReactNode; right?: ReactNode; eyebrow?: ReactNode }) {
  return (
    <div className="mb-6 flex flex-wrap items-start justify-between gap-4">
      <div className="min-w-0">
        {eyebrow}
        <h1 className="text-[26px] font-semibold tracking-tight">{title}</h1>
        {subtitle && <p className="mt-1 max-w-xl text-[15px] text-muted">{subtitle}</p>}
      </div>
      {right}
    </div>
  );
}

type BtnProps = ButtonHTMLAttributes<HTMLButtonElement> & { variant?: "primary" | "secondary" | "ghost"; size?: "sm" | "md" };
export function Button({ variant = "secondary", size = "md", className, ...rest }: BtnProps) {
  return (
    <button
      {...rest}
      className={cx(
        "inline-flex items-center justify-center gap-2 rounded-lg font-medium transition-colors disabled:opacity-50",
        size === "sm" ? "h-8 px-3 text-[13px]" : "h-10 px-4 text-sm",
        variant === "primary" && "bg-accent text-white hover:bg-accent-hover",
        variant === "secondary" && "border border-line bg-white hover:bg-neutral-50",
        variant === "ghost" && "text-ink hover:bg-neutral-100",
        className,
      )}
    />
  );
}

export function IconTile({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "accent" }) {
  return (
    <div className={cx("grid size-10 shrink-0 place-items-center rounded-lg", tone === "accent" ? "bg-accent-soft text-accent" : "bg-neutral-100 text-neutral-600")}>
      {children}
    </div>
  );
}

export function Bar({ value, color, className }: { value: number; color: string; className?: string }) {
  return (
    <div className={cx("h-1.5 overflow-hidden rounded-full bg-neutral-200", className)} role="progressbar" aria-valuenow={value} aria-valuemin={0} aria-valuemax={100}>
      <div className="h-full rounded-full transition-[width] duration-500" style={{ width: `${Math.max(0, Math.min(100, value))}%`, background: color }} />
    </div>
  );
}

export function Segmented<T extends string>({ options, value, onChange, label, stretch }: { options: { value: T; label: ReactNode }[]; value: T; onChange: (v: T) => void; label: string; stretch?: boolean }) {
  return (
    <div role="radiogroup" aria-label={label} className={cx("flex-wrap gap-1 rounded-lg bg-neutral-100 p-1", stretch ? "flex" : "inline-flex")}>
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            "inline-flex items-center justify-center gap-1.5 rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors",
            stretch && "flex-1 py-2",
            value === o.value ? "bg-white text-ink shadow-sm" : "text-muted hover:text-ink",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}

const STATUS = {
  covered: { label: "Covered", Icon: CheckCircle2, cls: "border-green-200 bg-green-50 text-green-700" },
  partial: { label: "Partial", Icon: CircleDashed, cls: "border-amber-200 bg-amber-50 text-amber-700" },
  missing: { label: "Missing", Icon: XCircle, cls: "border-red-200 bg-red-50 text-red-700" },
} as const;

export function StatusBadge({ status }: { status: CoverageStatus }) {
  const { label, Icon, cls } = STATUS[status];
  return (
    <span className={cx("inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[13px] font-medium", cls)}>
      <Icon className="size-3.5" /> {label}
    </span>
  );
}

export function StatusIcon({ status }: { status: CoverageStatus }) {
  const { Icon } = STATUS[status];
  const tone = status === "covered" ? "bg-green-50 text-green-600" : status === "partial" ? "bg-amber-50 text-amber-600" : "bg-red-50 text-red-600";
  return (
    <div className={cx("grid size-8 shrink-0 place-items-center rounded-lg", tone)}>
      <Icon className="size-5" />
    </div>
  );
}

export function Pill({ children, tone = "neutral" }: { children: ReactNode; tone?: "neutral" | "accent" | "good" | "warn" | "bad" }) {
  const cls = {
    neutral: "border-line bg-neutral-100 text-neutral-700",
    accent: "border-blue-200 bg-accent-soft text-accent",
    good: "border-green-200 bg-green-50 text-green-700",
    warn: "border-amber-200 bg-amber-50 text-amber-700",
    bad: "border-red-200 bg-red-50 text-red-700",
  }[tone];
  return <span className={cx("inline-flex items-center gap-1 rounded-md border px-2 py-0.5 text-[13px] font-medium", cls)}>{children}</span>;
}

export function Alert({ title, children, onClose }: { title: string; children: ReactNode; onClose?: () => void }) {
  return (
    <div role="alert" className="flex items-start gap-3 rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-800">
      <CircleAlert className="mt-0.5 size-4 shrink-0" />
      <p className="flex-1">
        <strong className="font-semibold">{title}</strong> {children}
      </p>
      {onClose && (
        <button onClick={onClose} aria-label="Dismiss" className="rounded px-1 text-red-700 hover:bg-red-100">
          ×
        </button>
      )}
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="px-5 py-10 text-center text-sm text-muted">{children}</p>;
}

/** Loading / error / "what's missing" block for pages that wait on the exam engine. */
export function Pending({ loading, error, onRetry, what }: { loading: boolean; error: string | null; onRetry: () => void; what: string }) {
  if (loading)
    return (
      <Card className="flex items-center gap-3 p-6 text-[15px] text-muted">
        <span className="size-4 animate-spin rounded-full border-2 border-accent border-t-transparent" /> Analysing your documents for {what}… this can take up to a minute on first load.
      </Card>
    );
  if (error)
    return (
      <Card className="p-6">
        <p className="text-red-700">{error}</p>
        <Button className="mt-3" size="sm" onClick={onRetry}>
          Try again
        </Button>
      </Card>
    );
  return null;
}

export function Notes({ notes }: { notes: string[] }) {
  if (!notes.length) return null;
  return (
    <ul className="mt-4 space-y-1.5 rounded-lg border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-900">
      {notes.map((n) => (
        <li key={n}>• {n}</li>
      ))}
    </ul>
  );
}
