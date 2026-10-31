"""Shared, dependency-light NLP helpers (tokenising, stemming, key phrases).

Everything here is deterministic and runs offline. The stemmer is deliberately
simple and conservative: its only job is to make "networks"/"network",
"defined"/"define", "learning"/"learn" compare equal so that lexical evidence
(shared key terms) can be shown to the student in plain language.
"""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from typing import Iterable, Sequence

STOPWORDS = frozenset(
    """a an the of in on at to for from by with without within into onto over under
    between among about above below and or nor but if then else than so as is are was
    were be been being am do does did done doing have has had having this that these
    those it its their there here which who whom whose what when where why how can
    could may might must shall should will would not no yes also such any all each
    every both either neither some most more less many much few using use used based
    via per etc eg ie one two three four five six give given following your you we
    our us he she they them his her i me my""".split()
)

# Words that tell the student *what to do* in an exam, not what the topic is.
COMMAND_WORDS = frozenset(
    """explain describe discuss define write note notes short briefly brief
    differentiate distinguish compare contrast list state mention enumerate illustrate
    elaborate outline summarize summarise derive prove show demonstrate justify
    evaluate analyse analyze example examples draw diagram diagrams question questions
    marks mark answer answers attempt detail details work working works identify
    name label sketch trace calculate compute find determine solve critically
    comment suitable appropriate relevant""".split()
)

# Words that are legitimate English but make poor *topic names* on their own.
GENERIC_TERMS = frozenset(
    """system systems data type types method methods technique techniques concept
    concepts importance advantage advantages disadvantage disadvantages application
    applications role need needs various different basic basics introduction overview
    term terms principle principles feature features characteristic characteristics
    process way ways approach approaches issue issues problem problems case cases
    section part unit chapter topic topics level levels difference differences
    between kind kinds form forms model models""".split()
)

_TOKEN_RE = re.compile(r"(?<![a-z0-9])[a-z]-[a-z][a-z0-9]*|[a-z0-9]+")
_SEGMENT_SPLIT_RE = re.compile(r"[.;:,!?()\[\]{}\n\r\t/\"“”]|\s[-–—]\s")


def stem(word: str) -> str:
    """Tiny suffix stripper. Consistent rather than linguistically perfect."""
    w = word.lower()
    if len(w) <= 3 or w.isdigit():
        return w
    if w.endswith("ies") and len(w) > 4:
        w = w[:-3] + "y"
    elif w.endswith(("sses", "xes", "ches", "shes")):
        w = w[:-2]
    elif w.endswith("s") and not w.endswith(("ss", "us", "is")) and len(w) > 3:
        w = w[:-1]
    if w.endswith("ing") and len(w) > 5:
        base = w[:-3]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "lsz":
            base = base[:-1]
        w = base
    elif w.endswith("ed") and len(w) > 4:
        base = w[:-2]
        if len(base) >= 3 and base[-1] == base[-2] and base[-1] not in "lsz":
            base = base[:-1]
        w = base
    elif w.endswith("ly") and len(w) > 5:
        w = w[:-2]
    if w.endswith("e") and len(w) > 4:
        w = w[:-1]
    return w


def tokenize(text: str) -> list[str]:
    """Lower-cased alphanumeric tokens (hyphens/underscores split)."""
    text = unicodedata.normalize("NFKC", text or "").lower().replace("_", " ")  # ligatures: "ﬁ" -> "fi"
    return _TOKEN_RE.findall(text)


def is_content_token(tok: str) -> bool:
    return (
        len(tok) >= 2
        and not tok.isdigit()
        and tok not in STOPWORDS
        and tok not in COMMAND_WORDS
    )


def content_terms(text: str, keep_generic: bool = True) -> list[str]:
    """Stemmed content words (no stopwords / exam command words)."""
    out = []
    for tok in tokenize(text):
        if not is_content_token(tok):
            continue
        if not keep_generic and tok in GENERIC_TERMS:
            continue
        out.append(stem(tok))
    return out


def lexical_analyzer(text: str) -> list[str]:
    """Analyzer for TF-IDF: stems + adjacent-stem bigrams (for phrase evidence)."""
    terms = content_terms(text)
    bigrams = [f"{a}_{b}" for a, b in zip(terms, terms[1:])]
    return terms + bigrams


def normalise_text(text: str) -> str:
    """Canonical form used for exact / near-exact duplicate checks."""
    return " ".join(content_terms(text))


def candidate_phrases(text: str, max_n: int = 3) -> list[tuple[tuple[str, ...], str]]:
    """Noun-phrase-like n-grams.

    Text is split at punctuation, stopwords and exam command words; every maximal
    run of content tokens yields its 1..max_n grams. A phrase may not start or
    end with a generic word ("advantages", "types") and may not consist only of
    generic words. Returns (stem-tuple, surface-string) pairs.
    """
    phrases: list[tuple[tuple[str, ...], str]] = []
    for segment in _SEGMENT_SPLIT_RE.split((text or "").lower()):
        run: list[str] = []
        runs: list[list[str]] = []
        for tok in tokenize(segment):
            if is_content_token(tok):
                run.append(tok)
            else:
                if run:
                    runs.append(run)
                run = []
        if run:
            runs.append(run)
        for r in runs:
            for n in range(1, max_n + 1):
                for i in range(len(r) - n + 1):
                    gram = r[i : i + n]
                    if gram[0] in GENERIC_TERMS or gram[-1] in GENERIC_TERMS:
                        continue
                    if all(g in GENERIC_TERMS for g in gram):
                        continue
                    phrases.append((tuple(stem(g) for g in gram), " ".join(gram)))
    return phrases


def surface_form(counter: Counter) -> str:
    """Most common surface spelling for a stem-tuple."""
    return counter.most_common(1)[0][0] if counter else ""


def pretty_topic(name: str) -> str:
    """Title-case a machine-found topic but keep acronyms (TCP, SQL, CNN ...).
    Names the user wrote themselves (any capital letter present) are left untouched."""
    if any(ch.isupper() for ch in name):
        return name
    words = []
    for w in name.split():
        words.append(w.upper() if len(w) <= 3 and w.isalpha() and w.lower() in _ACRONYM_HINTS else w.capitalize())
    return " ".join(words)


_ACRONYM_HINTS = frozenset(
    "tcp udp ip http dns sql cnn rnn lstm nlp ai ml os cpu gpu ram rom api xml json html css dbms rdbms osi lan wan vpn ssl tls ids dos ddos svm knn pca".split()
)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def first_sentences(text: str, limit: int = 220) -> str:
    text = re.sub(r"\s+", " ", text or "").strip()
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text or "").strip()
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"“(])", text)
    return [p.strip() for p in parts if p.strip()]


def safe_div(a: float, b: float) -> float:
    return a / b if b else 0.0


def ensure_sequence(x: Sequence | None) -> list:
    return list(x) if x else []


def strip_running_lines(records: Sequence[dict], min_pages: int = 3, share: float = 0.4) -> list[dict]:
    """Remove running headers/footers ("Cite as: ...", "Page 3 of 9", a course title on every page).

    A line (digits ignored) that shows up on at least ``min_pages`` pages AND on at least ``share`` of the
    pages of its document is page furniture, not content. Documents with fewer pages are left alone.
    Records are {document_id, page_number, text, ...}; new dicts are returned.
    """
    def key(line: str) -> str:
        return re.sub(r"\d+", "#", re.sub(r"\s+", " ", unicodedata.normalize("NFKC", line)).strip().lower())

    pages_of_doc: dict = {}
    seen: dict = {}
    for r in records:
        doc = r.get("document_id")
        pages_of_doc.setdefault(doc, set()).add(r.get("page_number"))
        for ln in (r.get("text") or "").splitlines():
            k = key(ln)
            if len(k) >= 8:
                seen.setdefault((doc, k), set()).add(r.get("page_number"))
    furniture = set()
    for (doc, k), pgs in seen.items():
        n = len(pages_of_doc[doc])
        if n >= min_pages and len(pgs) >= min_pages and len(pgs) >= share * n:
            furniture.add((doc, k))
    if not furniture:
        return [dict(r) for r in records]
    out = []
    for r in records:
        doc = r.get("document_id")
        kept = [ln for ln in (r.get("text") or "").splitlines() if (doc, key(ln)) not in furniture]
        out.append({**r, "text": "\n".join(kept)})
    return out
