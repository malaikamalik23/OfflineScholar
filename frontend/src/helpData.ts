export interface HelpArticle {
  id: string;
  title: string;
  body: string;
}

export interface HelpArea {
  id: string;
  title: string;
  blurb: string;
  icon: "rocket" | "files" | "chat" | "clipboard";
  articles: HelpArticle[];
}

export const HELP_AREAS: HelpArea[] = [
  {
    id: "start",
    title: "Getting Started",
    blurb: "Set up your workspace and learn the core study flow.",
    icon: "rocket",
    articles: [
      { id: "s1", title: "How Offline Scholar works", body: "Upload your notes, past papers and syllabus. They are read on this device, then used to answer questions, measure syllabus coverage and build quizzes. Nothing is uploaded to a server." },
      { id: "s2", title: "The study loop", body: "Check the Dashboard for priority topics, ask the Study Assistant about them, take a practice quiz, then follow the Revision Plan. Quiz results update your mastery scores." },
      { id: "s3", title: "Switching courses", body: "Use the subject menu at the top of every page. Create a new subject from the same menu; each subject has its own documents, analysis and quizzes." },
    ],
  },
  {
    id: "materials",
    title: "Study Materials",
    blurb: "Upload, organize, and troubleshoot your course files.",
    icon: "files",
    articles: [
      { id: "m1", title: "Supported files", body: "PDF files up to 50 MB. Choose a category first: lecture notes, past paper or syllabus." },
      { id: "m2", title: "Why did a file fail?", body: "Scanned PDFs often have no text layer. Export the file again with text, or run it through OCR before uploading." },
      { id: "m3", title: "Indexing takes a while", body: "Large files are split into passages and indexed locally. You can keep studying while it runs." },
      { id: "m4", title: "Removing a document", body: "Open Documents and choose the bin icon next to the file. Its passages are removed from search straight away." },
    ],
  },
  {
    id: "assistant",
    title: "AI Study Assistant",
    blurb: "Ask grounded questions and verify every cited answer.",
    icon: "chat",
    articles: [
      { id: "a1", title: "Asking good questions", body: "Name the topic and what you want: a definition, a comparison, or a running time. For example, “What is the difference between BFS and DFS?”" },
      { id: "a2", title: "Checking citations", body: "Claims in an answer end with a numbered source like [1]. The list under the answer shows the document and page; click one to see the passage." },
      { id: "a3", title: "Limiting the search", body: "Use “Searching in” above the question box to search this subject or all subjects." },
      { id: "a4", title: "When there's no answer", body: "If your notes don't mention a topic, the assistant says so rather than guessing. Upload notes for that topic and ask again." },
    ],
  },
  {
    id: "exam",
    title: "Exam Preparation",
    blurb: "Understand past-paper patterns using measured evidence.",
    icon: "clipboard",
    articles: [
      { id: "e1", title: "Reading past-paper analysis", body: "Most examined topics counts how many questions match each syllabus topic across your past papers. Hover a bar to see why it ranks there." },
      { id: "e2", title: "Repeated questions", body: "Questions that appeared in two or more years are grouped together, so you can practise the pattern rather than one paper." },
      { id: "e3", title: "The focus matrix", body: "The Revision Plan plots exam importance against your readiness (quiz results, or notes coverage before you take quizzes). Topics in the shaded area are examined often and need work." },
      { id: "e4", title: "Practice quizzes", body: "Quizzes are written by the local model from your lecture notes, with a source page for each question. Results update your weak topics and revision plan." },
    ],
  },
];

export const SHORTCUTS: { keys: string[]; action: string }[] = [
  { keys: ["Alt", "1–7"], action: "Go to Dashboard, Documents, Assistant, Past Papers, Coverage, Quiz or Revision" },
  { keys: ["Alt", "H"], action: "Open Help & Shortcuts" },
  { keys: ["/"], action: "Focus the search box on this page" },
  { keys: ["Enter"], action: "Send a question to the Study Assistant" },
  { keys: ["Shift", "Enter"], action: "New line in the question box" },
  { keys: ["1–4"], action: "Pick an answer during a quiz" },
];
