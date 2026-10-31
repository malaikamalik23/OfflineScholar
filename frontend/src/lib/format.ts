import type { CoverageStatusText, DocType } from "./api";

export type CoverageStatus = "covered" | "partial" | "missing";

export const coverageStatus = (s: CoverageStatusText): CoverageStatus =>
  s === "Covered" ? "covered" : s === "Partially covered" ? "partial" : "missing";

export const DOC_TYPE_LABEL: Record<DocType, string> = {
  lecture: "Lecture notes",
  past_paper: "Past paper",
  syllabus: "Syllabus",
};

/** 0-100 → bar colour (red = weak, green = strong). */
export const scoreColor = (m: number) =>
  m < 40 ? "var(--color-bad)" : m < 60 ? "var(--color-warn)" : m < 75 ? "var(--color-accent)" : "var(--color-good)";

export const priorityTone = (label: string): "bad" | "warn" | "neutral" =>
  label === "High" ? "bad" : label === "Medium" ? "warn" : "neutral";

export const titleCase = (s: string) => s.replace(/\b\w/g, (c) => c.toUpperCase());

export const mb = (bytes: number) => {
  const v = bytes / (1024 * 1024);
  return v < 0.1 ? "<0.1 MB" : `${v.toFixed(1)} MB`;
};

/** PDFs often keep typographic ligatures (ﬁ, ﬂ); show them as normal letters. */
// eslint-disable-next-line no-control-regex
export const clean = (s: string) => s.normalize("NFKC").replace(/[\u0000-\u0008\u000b\u000c\u000e-\u001f]/g, " ");
