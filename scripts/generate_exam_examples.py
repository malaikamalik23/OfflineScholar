"""Run the engine on the SYNTHETIC fixtures and write real response samples to docs/exam_examples/.

These files show Member 3 the exact JSON shapes. They come from running the code, are not
hand-written, and describe a made-up course — never present them as real exam statistics.

    python scripts/generate_exam_examples.py
"""
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exam_engine import ExamEngine  # noqa: E402
from tests.exam_fixtures import LECTURES, PAPERS, SYLLABUS_TOPICS  # noqa: E402

out = ROOT / "docs" / "exam_examples"
out.mkdir(parents=True, exist_ok=True)


def dump(name, obj):
    (out / f"{name}.json").write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


eng = ExamEngine()
past = eng.analyze_past_papers(PAPERS, syllabus_topics=SYLLABUS_TOPICS)
dump("past_paper_analysis", past)
dump("important_topics", past["important_topics"])
cov = eng.syllabus_coverage(SYLLABUS_TOPICS, LECTURES)
dump("syllabus_coverage", cov)
quiz = eng.make_quiz(LECTURES, "Neural networks", 3)
dump("quiz_questions", quiz)

rng = random.Random(7)
attempts = []
for topic in ["Neural networks", "Decision trees", "Overfitting and regularization"]:
    qz = eng.make_quiz(LECTURES, topic, 3)
    for _ in range(3):
        bad = 0.8 if topic == "Neural networks" else 0.2
        ans = {q["question_id"]: (q["correct_option"] if rng.random() > bad else next(k for k in "ABCD" if k != q["correct_option"]))
               for q in qz["questions"]}
        attempts.append(eng.grade(qz, ans))
dump("quiz_results", attempts[0])
weak = eng.weak_topics(attempts)
dump("weak_topic_analysis", weak)
plan = eng.revision_plan({"topic_frequency": past["topic_frequency"], "syllabus_coverage": cov, "weak_topics": weak},
                         days=5, minutes_per_day=90)
dump("revision_plan", plan)
print("wrote", sorted(p.name for p in out.glob("*.json")))
