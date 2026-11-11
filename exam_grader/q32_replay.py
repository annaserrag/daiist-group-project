"""Compare the current Q32 rule with the frozen 46-paper run (no OCR or LLM)."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from .grading import grade_exam
from .hyperparameters import VERSION
from .reference_marks import SHA256, reference_mark, teacher_data

ROOT = Path(__file__).resolve().parents[1]


def metrics(rows, mark_key):
    proposed = [r for r in rows if r[mark_key] is not None]
    errors = [abs(r[mark_key] - r["teacher"]) for r in proposed if r["teacher"] is not None]
    return {"papers": len(rows), "proposals": len(proposed), "review": len(rows) - len(proposed),
            "paired_marks": len(errors), "exact": errors.count(0),
            "within_half": sum(e <= .5 for e in errors), "mae": mean(errors) if errors else None}


def replay(folder):
    if folder.resolve().is_relative_to(ROOT):
        raise ValueError("Use the private folder containing the frozen readings.")
    baseline = json.loads((ROOT / "eval/q32_fresh_count.json").read_text(encoding="utf-8"))
    raw = (folder / "q32-fresh-blind.json").read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != baseline["frozen_input_and_proposals_sha256"]:
        raise ValueError("The readings or original decisions have changed.")
    rows = []
    for old in json.loads(raw)["rows"]:
        state = old["state"]
        answers = [] if state == "not_located" else [
            dict(question_id=32, text=old["text"], state=state,
                 confidence=float(state == "read"), id_confidence=1.0)]
        new = grade_exam(dict(schema_version="vision.exam.v1", exam_status="in_scope",
                              answers=answers))["teacher"]["answers"][31]
        rows.append({"paper": old["paper"], "text": old["text"], "previous": old["proposed"],
                     "current": new["mark"], "reason": new["reason"], "feedback": new["feedback"],
                     "previous_credits": old["credited"], "credits": new.get("credited_examples", [])})
    refs = teacher_data()
    for row in rows:
        row["teacher"] = reference_mark(32, refs[f"Student_{row['paper']}"][32])
    (folder / f"q32-replay-{VERSION}.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    return {"dataset": baseline["dataset"], "question_id": 32, "rule": VERSION,
            "scope": "Development replay on the same 46 frozen readings; no fresh validation.",
            "frozen_input_and_proposals_sha256": digest, "teacher_csv_sha256": SHA256,
            "numeric_teacher_marks": sum(r["teacher"] is not None for r in rows),
            "before": metrics(rows, "previous"), "after": metrics(rows, "current"),
            "left_review": sum(r["previous"] is None and r["current"] is not None for r in rows),
            "entered_review": sum(r["previous"] is not None and r["current"] is None for r in rows),
            "still_review_with_new_credits": sum(r["current"] is None and r["credits"] != r["previous_credits"] for r in rows),
            "review_by_reason": dict(Counter(r["reason"] for r in rows if r["current"] is None))}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_folder", type=Path)
    args = parser.parse_args()
    aggregate = replay(args.private_folder)
    (ROOT / "eval" / f"q32_replay_{VERSION}.json").write_text(json.dumps(aggregate, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(aggregate, indent=2))
