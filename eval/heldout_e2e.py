"""
The held-out end-to-end test: the whole product, run once, on five Mendeley exams no earlier step used (four clean).

Why these five. Every other Mendeley exam was used somewhere during development: the 30-case packet (8 source
papers), the drift reference crops (30 papers), or the grader's written-answer work (the papers named in eval/ and
exam_grader/). A search of every file in eval/, exam_grader/, monitoring/ and tests/ for each exam's ID and file
hash leaves Students 43, 45, 46, 47 and 50. One exception applies to all 50 exams: the Q32 rule was revised on all
of them (eval/q32_all50_*.json), so Q32 is reported apart and left out of the clean written score.

Student_45 is no longer clean either. The first run of this test (during a Gemini outage) showed a header line on
Student_45 sending the whole exam to review, and the bridge's header rule was written in response
(eval/header_rule_dev.py). The team kept the rule and discloses it: Student_45 is still run, but scored apart, and
the clean result is the other four exams.

The inputs. eval/heldout/inputs/h<n>.pdf are the full exams with every faint pixel whitened (grey level above 165):
the student's name and ID number are written faintly in a header and show through from the back of the sheet, and
this removes them before anything reaches the reader or Gemini, while the answers, in dark pen, stay. Every page's
top was checked by eye. eval/heldout/manifest.csv has the source and output hashes.

The product runs as the App would, with its fixed default policy (reading threshold 0.80, written marks at or below
1/2 sent to review; the bandit is not involved), and the five files are approved for Gemini for this run only.

The references are the teacher's published marks (multiple choice: the letter recorded, scored with the key).
Two baselines: always full credit, and a typical-mark baseline (each written question's mean teacher mark over the
other 45 exams, rounded to a half point).

    uv run python eval/heldout_e2e.py      (needs GEMINI_API_KEY; run it once, after everything is frozen)

Writes eval/heldout/results.json (every question of every exam) and eval/heldout/summary.json.
"""

import csv
import json
import os
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import env  # noqa: E402

env.load_env()  # GEMINI_API_KEY from .env

TEACHER_CSV = Path.home() / ".cache" / "daiist-vision" / "mendeley" / "Teacher_manual_marks_Anonymized.csv"
HELDOUT = ROOT / "eval" / "heldout"
Q32 = 32
NOT_CLEAN = {"Student_45"}  # the header rule was written after seeing this exam fail (see above)


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY: without it every written answer would be referred.")
    manifest = list(csv.DictReader((HELDOUT / "manifest.csv").open(encoding="utf-8")))
    # Approve exactly these five files for Gemini, for this process only (pipeline.resources reads it once).
    os.environ["GRADE_APPROVED_HASHES"] = " ".join(r["sha256"] for r in manifest)
    import pipeline
    from exam_grader.integration import file_sha256
    from exam_grader.reference_marks import fit_written_baseline, reference_mark, reference_tables

    try:
        exams, _ = reference_tables(TEACHER_CSV if TEACHER_CSV.is_file() else None)
    except ValueError:  # a local copy with the name column dropped no longer matches the published hash
        exams, _ = reference_tables()
    held = [f"Student_{r['source_paper']}" for r in manifest]
    baseline = fit_written_baseline(exams, [e for e in exams if e not in held], held)

    rows, papers = [], []
    for r in manifest:
        path, exam_id = ROOT / r["input"], f"Student_{r['source_paper']}"
        if file_sha256(path) != r["sha256"]:
            raise SystemExit(f"{path.name} differs from the manifest.")
        result = pipeline.run_views(path)
        teacher, record = result["teacher"], result["record"]
        papers.append({"exam": exam_id, "status": teacher["exam_status"], "latency_s": record["latency_s"],
                       "reader_s": record["features"].get("vision_latency_s"), "tokens": record["tokens"],
                       "cost_eur": record["cost_eur"], "review_count": teacher["review_count"]})
        for item in teacher["answers"]:
            q = item["question_id"]
            reference = reference_mark(q, exams[exam_id][q])
            rows.append({"exam": exam_id, "q": q, "type": "mcq" if q <= 20 else "written", "proposed": item["mark"],
                         "reason": item["reason"], "reference": reference, "max": item["max_mark"],
                         "reading_confidence": item.get("reading_confidence"),
                         "typical_mark": baseline.get(q) if q > 20 else None})
        print(exam_id, teacher["exam_status"], f"{record['latency_s']:.0f} s", f"{teacher['review_count']} to review",
              flush=True)

    def score(part: list[dict]) -> dict:
        referenced = [r for r in part if r["reference"] is not None]
        proposed = [r for r in referenced if r["proposed"] is not None]
        errors = [abs(r["proposed"] - r["reference"]) for r in proposed]
        return {"slots": len(part), "with_reference": len(referenced), "proposed": len(proposed),
                "review_share": round(1 - len(proposed) / len(referenced), 3) if referenced else None,
                "exact": sum(e == 0 for e in errors), "within_half": sum(e <= 0.5 for e in errors),
                "mean_abs_error": round(mean(errors), 3) if errors else None,
                "full_credit_baseline_exact": sum(r["reference"] == r["max"] for r in referenced)}

    clean = [r for r in rows if r["exam"] not in NOT_CLEAN]
    written = [r for r in clean if r["type"] == "written" and r["q"] != Q32]
    typical = [r for r in written if r["reference"] is not None and r["typical_mark"] is not None]
    summary = {
        "exams": len(papers), "clean": "Students 43, 46, 47, 50; used by no earlier step",
        "mcq": score([r for r in clean if r["type"] == "mcq"]),
        "written_without_q32": score(written) | {
            "typical_mark_baseline_exact": sum(r["typical_mark"] == r["reference"] for r in typical),
            "typical_mark_baseline_mae": round(mean(abs(r["typical_mark"] - r["reference"]) for r in typical), 3)},
        "q32_not_clean": score([r for r in rows if r["q"] == Q32]),
        "student_45_not_clean": {"mcq": score([r for r in rows if r["exam"] in NOT_CLEAN and r["type"] == "mcq"]),
                                 "written": score([r for r in rows if r["exam"] in NOT_CLEAN and r["type"] == "written"])},
        "per_exam": papers,
        "median_latency_s": sorted(p["latency_s"] for p in papers)[len(papers) // 2],
        "total_tokens": sum(p["tokens"] for p in papers), "total_cost_eur": round(sum(p["cost_eur"] for p in papers), 6),
    }
    (HELDOUT / "results.json").write_text(json.dumps(rows, indent=1) + "\n")
    (HELDOUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_exam"}, indent=2))


if __name__ == "__main__":
    main()
