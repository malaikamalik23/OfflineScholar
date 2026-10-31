import { useEffect, useState } from "react";
import { BarChart3, CircleHelp, Files, FileText, Gauge, Library, MessageSquareText, Settings, Target, type LucideIcon } from "lucide-react";

export type Route = "dashboard" | "documents" | "assistant" | "papers" | "coverage" | "quiz" | "revision" | "settings" | "help";

export const ROUTES: { id: Route; label: string; title: string; icon: LucideIcon }[] = [
  { id: "dashboard", label: "Dashboard", title: "Dashboard", icon: Gauge },
  { id: "documents", label: "Documents", title: "Documents", icon: Files },
  { id: "assistant", label: "Study Assistant", title: "Study Assistant", icon: MessageSquareText },
  { id: "papers", label: "Past Papers", title: "Past Papers", icon: FileText },
  { id: "coverage", label: "Syllabus Coverage", title: "Syllabus Coverage", icon: Library },
  { id: "quiz", label: "Quiz", title: "Quiz", icon: Target },
  { id: "revision", label: "Revision Plan", title: "Revision Plan", icon: BarChart3 },
  { id: "settings", label: "Settings", title: "Settings", icon: Settings },
  { id: "help", label: "Help & Shortcuts", title: "Help & Shortcuts", icon: CircleHelp },
];

const parse = (): Route => {
  const h = window.location.hash.replace(/^#\/?/, "") as Route;
  return ROUTES.some((r) => r.id === h) ? h : "dashboard";
};

export function useRoute() {
  const [route, setRoute] = useState<Route>(parse);
  useEffect(() => {
    const on = () => setRoute(parse());
    window.addEventListener("hashchange", on);
    return () => window.removeEventListener("hashchange", on);
  }, []);
  return route;
}

export const go = (r: Route) => {
  window.location.hash = r;
};

// One-shot hand-offs between pages, read once by the page that opens next.
let pendingQuestion: string | null = null;
let pendingQuizTopic: string | null = null;

export function askAssistant(question: string) {
  pendingQuestion = question;
  go("assistant");
}
export function takePendingQuestion() {
  const q = pendingQuestion;
  pendingQuestion = null;
  return q;
}

export function startFocusedQuiz(topic: string) {
  pendingQuizTopic = topic;
  go("quiz");
}
export function takePendingQuizTopic() {
  const t = pendingQuizTopic;
  pendingQuizTopic = null;
  return t;
}
