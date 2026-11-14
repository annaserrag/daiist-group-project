"""Exercise the production drift detector on recorded MCQs with a controlled missing-page shift."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import monitor
from .catalog import OPTIONS
from .grading import grade_exam
from .reference_marks import SHA256, teacher_data


def features(cells, *, missing_page=False):
    answers = [{"question_id": q, "text": cells[q], "state": "read", "confidence": 1.0, "id_confidence": 1.0}
               for q in range(1, 21) if cells[q] in OPTIONS[q] and (not missing_page or q <= 15)]
    payload = {"schema_version": "vision.exam.v1", "exam_status": "in_scope", "answers": answers}
    signals = grade_exam(payload)["record"]["features"]
    return {"reading_count": signals["reading_count"]}


def replay(exams):
    if len(exams) != 50:
        raise ValueError("The published replay requires all 50 exams.")
    def order(prefix):
        return sorted(exams, key=lambda name: hashlib.sha256((prefix + name).encode()).digest())
    observed = {name:features(cells) for name,cells in exams.items()}
    degraded = {name:features(cells, missing_page=True) for name,cells in exams.items()}
    ordered = order("drift-v1/")
    reference, heldout = ordered[:30], ordered[30:]
    ref = pd.DataFrame(observed[name] for name in reference)
    def assess(shifted):
        batch = pd.DataFrame((degraded if shifted else observed)[name] for name in heldout)
        # Reproduce the archived unsmoothed arm even after the live default changes.
        checks = monitor.drift(ref, batch, min_current=monitor.MIN_WINDOW, pseudocount=0)
        candidate = monitor.drift(ref, batch, min_current=monitor.MIN_WINDOW, pseudocount=0.5)
        return {"exam_count":len(batch), "mean_observed_mcqs":float(batch.reading_count.mean()),
                "signals":json.loads(checks.to_json(orient="records")),
                "half_count_candidate":json.loads(candidate.to_json(orient="records"))}
    repeats = {"current":{"normal_alerts":0,"shifted_alerts":0},
               "half_count_candidate":{"normal_alerts":0,"shifted_alerts":0}}
    for i in range(100):
        ids = order(f"drift-repeat/{i}/")
        source = pd.DataFrame(observed[name] for name in ids[:30])
        for label, data in (("normal_alerts", observed), ("shifted_alerts", degraded)):
            batch = pd.DataFrame(data[name] for name in ids[30:])
            for method, smoothing in (("current", 0), ("half_count_candidate", 0.5)):
                signal = monitor.drift(source, batch, min_current=monitor.MIN_WINDOW, pseudocount=smoothing)
                repeats[method][label] += int(signal.alert.iloc[0])
    return {
        "dataset":"10.17632/sf3kvjwknt.1","teacher_csv_sha256":SHA256,
        "scope":"Engineering replay of the published MCQ selections. Each valid letter becomes one observed "
                "slot. Confidence is fixed. Only reading count is evaluated. No live OCR or team course cases.",
        "split":"Deterministic SHA-256 order of drift-v1/ followed by the published exam ID. "
                "First 30 provide the reference. The same remaining 20 form both comparison windows.",
        "shift":"Withhold detections from Q16 through Q20.",
        "reference_count":len(ref), "reference_mean_observed_mcqs":float(ref.reading_count.mean()),
        "normal_window":assess(False), "shifted_window":assess(True),
        "sensitivity":"100 overlapping deterministic 30/20 repartitions of the same 50 exams. Both methods "
                      "use the same alert threshold. The candidate adds half an observation per possible bin.",
        "repartition_alert_counts_out_of_100":repeats}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--teacher-csv", type=Path, help="Optional local copy of the checksum-pinned source.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    data = json.dumps(replay(teacher_data(args.teacher_csv)), indent=2) + "\n"
    if args.output:
        args.output.write_text(data, encoding="utf-8")
    print(data)
