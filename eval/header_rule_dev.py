"""
Check the bridge's header rule (exam_grader.vision_finetune.is_heading) on development exams only.

The clean held-out run showed that one header line ("Data Science") above the first question sends a whole exam to
review. The rule that drops such lines is checked here on the 45 Mendeley exams that are NOT in the clean held-out
set (eval/heldout/manifest.csv), so the held-out exams stay unseen by the fix. Each exam is read once; the same read
goes through the bridge with and without the rule, and is graded locally (multiple choice against the key; nothing is
sent to Gemini, so written answers are referred in both arms and do not count here).

Only counts are written (eval/header_rule_dev.json): the text of dropped lines can hold a student's name.

    uv run --no-dev python eval/header_rule_dev.py
"""

import csv
import json
import sys
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exam_grader import vision_finetune as bridge  # noqa: E402
from exam_grader.grading import Policy, grade_exam  # noqa: E402
from exam_grader.page_confidence import read_exam as read_pages  # noqa: E402
from exam_grader.reference_marks import reference_mark, reference_tables  # noqa: E402

PDFS = Path.home() / ".cache" / "daiist-vision" / "mendeley" / "pdfs"
TEACHER_CSV = Path.home() / ".cache" / "daiist-vision" / "mendeley" / "Teacher_manual_marks_Anonymized.csv"


def grade(raw, rule: bool) -> dict:
    original = bridge.is_heading
    if not rule:
        bridge.is_heading = lambda row, first_page: False
    try:
        payload = bridge.adapt(raw)
    finally:
        bridge.is_heading = original
    teacher = grade_exam(payload, index=None, policy=Policy(written_review_at_or_below=1))["teacher"]
    return {"loose": sum(a["question_id"] is None for a in payload["answers"]), "teacher": teacher}


def main() -> None:
    held = {int(r["source_paper"]) for r in csv.DictReader((ROOT / "eval" / "heldout" / "manifest.csv").open())}
    try:
        exams, _ = reference_tables(TEACHER_CSV if TEACHER_CSV.is_file() else None)
    except ValueError:
        exams, _ = reference_tables()
    rows = []
    for n in sorted(int(p.stem.split("_")[1]) for p in PDFS.glob("Student_*.pdf")):
        if n in held:
            continue
        raw = read_pages(PDFS / f"Student_{n}.pdf")  # read in memory only: nothing with text is saved
        dropped = [len(str(r.get("text", "")).split()) for r in raw["unassigned"]
                   if bridge.is_heading(r, min([q["page"] for q in raw["questions"] if type(q.get("page")) is int] or [1]))]
        row = {"exam": f"Student_{n}", "dropped_lines": len(dropped), "dropped_max_words": max(dropped, default=0)}
        for arm, rule in (("before", False), ("after", True)):
            try:
                result = grade(raw, rule)
            except ValueError as error:  # the bridge rejects a read it cannot use (e.g. a looping reader)
                row[arm] = {"loose_lines": None, "blocked": True, "mcq_proposed": 0, "mcq_exact": 0, "review": 35,
                            "reader_failed": type(error).__name__}
                continue
            answers = result["teacher"]["answers"]
            mcq = [a for a in answers if a["question_id"] <= 20]
            truth = {q: reference_mark(q, exams[row["exam"]][q]) for q in range(1, 21)}
            proposed = [a for a in mcq if a["mark"] is not None and truth[a["question_id"]] is not None]
            row[arm] = {"loose_lines": result["loose"], "blocked": all(a["reason"] == "unassigned_answer" for a in answers),
                        "mcq_proposed": len(proposed),
                        "mcq_exact": sum(a["mark"] == truth[a["question_id"]] for a in proposed),
                        "review": result["teacher"]["review_count"]}
        rows.append(row)
        print(row["exam"], row["dropped_lines"], row["before"]["blocked"], "->", row["after"]["blocked"],
              row["before"]["mcq_proposed"], "->", row["after"]["mcq_proposed"], flush=True)
    summary = {"exams": len(rows)}
    for arm in ("before", "after"):
        summary[arm] = {"blocked_exams": sum(r[arm]["blocked"] for r in rows),
                        "mcq_proposed": sum(r[arm]["mcq_proposed"] for r in rows),
                        "mcq_exact": sum(r[arm]["mcq_exact"] for r in rows),
                        "median_review_per_exam": median(r[arm]["review"] for r in rows)}
    summary["dropped_lines"] = sum(r["dropped_lines"] for r in rows)
    summary["dropped_line_max_words"] = max(r["dropped_max_words"] for r in rows)
    (ROOT / "eval" / "header_rule_dev.json").write_text(json.dumps({"summary": summary, "exams": rows}, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
