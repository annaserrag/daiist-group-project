"""Evaluation-only adapter for the public teacher CSV. No student-level data are saved."""
import argparse
import csv
import hashlib
import io
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from urllib.request import Request, urlopen

from .catalog import KEY, OPTIONS

SOURCE = "https://data.mendeley.com/public-files/datasets/sf3kvjwknt/files/31c31cf0-01de-42f6-a72b-73a013ffacd1/file_downloaded"
SHA256 = "51e194a98970dddf738dd81bdbc51e2d7dea5637666ee1ff7402e7165154b017"

def reference_tables(local_path=None):
    if local_path:
        raw = Path(local_path).read_bytes()
    else:
        with urlopen(Request(SOURCE, headers={"User-Agent":"Mozilla/5.0"}), timeout=25) as response:
            raw = response.read()
    if hashlib.sha256(raw).hexdigest() != SHA256:
        raise ValueError("Expected the published version 1 anonymized teacher CSV.")
    exams, mcq_totals = {}, {}
    for record in csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))):
        exam_id = record["Student_ID"].strip()
        if not re.fullmatch(r"Student_[0-9]+", exam_id) or exam_id in exams:
            raise ValueError("Unexpected anonymous identifier in the reference file.")
        exams[exam_id] = {q: record[str(q)].strip() for q in KEY}
        total = float(record["MCQ mark"])
        if not total.is_integer() or not 0 <= total <= 20:
            raise ValueError("Invalid published teacher MCQ total.")
        mcq_totals[exam_id] = int(total)
    return exams, mcq_totals

def teacher_data(local_path=None):
    return reference_tables(local_path)[0]

def audit_recorded_choices(exams, section_totals):
    """Replay recorded choices through the production grader with the reading and ID assumed correct."""
    from .grading import grade_exam
    counts = Counter()
    complete_counts = Counter()
    partial_counts = Counter()
    for exam, cells in exams.items():
        observations = [{"question_id":q, "text":cells[q], "state":"read",
                         "confidence":1.0, "id_confidence":1.0} for q in range(1,21)]
        payload = {"schema_version":"vision.exam.v1","exam_status":"in_scope","answers":observations}
        rows = grade_exam(payload)["teacher"]["answers"][:20]
        marks = [r["mark"] for r in rows]
        for r in rows:
            q = r["question_id"]
            expected = reference_mark(q,cells[q])
            if expected is None:
                counts["unscorable_recorded_choices"] += 1
                counts["unscorable_routed_to_review"] += r["mark"] is None
            else:
                counts["scorable_recorded_choices"] += 1
                counts["match_published_key_scoring"] += r["mark"] == expected
        unresolved = marks.count(None)
        subtotal = sum(m for m in marks if m is not None)
        teacher = section_totals[exam]
        group = partial_counts if unresolved else complete_counts
        group["exams"] += 1
        group["teacher_total_in_published_key_range"] += subtotal <= teacher <= subtotal+unresolved
        # Sensitivity analysis only; all Q1 recorded choices are present in source v1.
        if marks[0] is not None:
            hypothetical = subtotal - marks[0] + int(cells[1].strip().upper()=="D")
            group["q1_sensitivity_eligible"] += 1
            group["teacher_total_in_q1_D_range"] += hypothetical <= teacher <= hypothetical+unresolved
    return {"dataset":"10.17632/sf3kvjwknt.1","file_sha256":SHA256,
            "input":"published recorded choices with assumed correct transcription and question assignment",
            "exams":len(exams),"recorded_slots":len(exams)*20,**counts,
            "complete_mcq_sheets":dict(complete_counts),"partial_mcq_sheets":dict(partial_counts),
            "range_definition":"subtotal of proposed MCQ marks through subtotal plus count of unresolved choices",
            "sensitivity":"Q1=D is a hypothetical comparison only. The runtime key uses B."}

def reference_mark(q, value):
    if q <= 20:
        letter = value.strip().upper()
        if letter in OPTIONS[q]:
            return int(letter == KEY[q])
        return None
    if value in ("", "NC", "NM"):
        return None
    try:
        mark = float(value)
    except ValueError:
        return None
    return mark if mark in {0, .5, 1, 1.5, 2} else None

def fit_written_baseline(exams, training_ids, evaluation_ids):
    ids = list(training_ids)
    if not ids or len(set(ids)) != len(ids) or set(ids)-exams.keys():
        raise ValueError("Training source IDs must be nonempty, unique and present in the reference data.")
    if set(ids).intersection(evaluation_ids):
        raise ValueError("A source exam appears in both baseline training and evaluation.")
    baseline = {}
    for q in range(21,36):
        valid = [m for exam in ids if (m := reference_mark(q,exams[exam][q])) is not None]
        baseline[q] = int(mean(valid)*2+.5)/2 if valid else None
    return baseline

def mark_matrix(rows):
    """Rows are teacher marks. Columns are proposals or review. Unlabelled slots remain visible."""
    matrix = {}
    for row in rows:
        teacher = "no_reference" if row["teacher_mark"] == "" else f"{float(row['teacher_mark']):g}"
        proposed = "review" if row["proposed_mark"] == "" else f"{float(row['proposed_mark']):g}"
        cells = matrix.setdefault(teacher, {})
        cells[proposed] = cells.get(proposed,0) + 1
    return matrix


def compare_written(pairs):
    written = [row for row in pairs if row["question_id"] > 20]
    labelled = [row for row in written if row["teacher_mark"] != ""]
    baseline_rows = [row for row in labelled if row["baseline_mark"] is not None]
    matched = [row for row in baseline_rows if row["proposed_mark"] != ""]
    def scores(group, column):
        errors = [abs(float(r[column])-float(r["teacher_mark"])) for r in group]
        return {"n":len(errors),"exact":mean(e==0 for e in errors) if errors else None,
                "within_half":mean(e<=.5 for e in errors) if errors else None,
                "mean_absolute_error":mean(errors) if errors else None}
    return {"written_slots":len(written),"valid_teacher_marks":len(labelled),"teacher_by_proposal":mark_matrix(written),
            "labelled_questions_without_training_marks":len(labelled)-len(baseline_rows),
            "sent_to_review_with_valid_reference":sum(r["proposed_mark"]=="" for r in labelled),
            "baseline_on_all_scorable":scores(baseline_rows,"baseline_mark"),
            "baseline_on_matched_answers":scores(matched,"baseline_mark"),
            "grader_on_matched_answers":scores(matched,"proposed_mark")}

def audit(exams):
    """Leave one entire exam out. Train each question mean on the other exams' valid marks."""
    counts, sums = Counter(), Counter()
    skipped = Counter()
    for answers in exams.values():
        for q, value in answers.items():
            m = reference_mark(q, value)
            if m is None:
                if q <= 20:
                    skipped["unusable_mcq_choice"] += 1
                elif value in ("","NC","NM"):
                    skipped["unmarked_written_slot"] += 1
                else:
                    skipped["invalid_written_mark"] += 1
            else:
                counts[q] += 1
                sums[q] += m
    errors, rounded_errors = [], []
    for answers in exams.values():
        for q in range(21,36):
            m = reference_mark(q, answers[q])
            if m is None or counts[q] < 2:
                continue
            prediction = (sums[q] - m) / (counts[q] - 1)
            # The legal grade scale uses half points; midpoint ties round upward.
            rounded = int(prediction * 2 + .5) / 2
            errors.append(abs(m - prediction))
            rounded_errors.append(abs(m - rounded))
    return {"dataset":"10.17632/sf3kvjwknt.1","exams":len(exams),
            "published_key_scorable_mcq_choices":sum(counts[q] for q in range(1,21)),
            "valid_teacher_written_marks":sum(counts[q] for q in range(21,36)),
            "exclusions":dict(skipped),
            "written_baseline":{"method":"leave-one-exam-out question mean, valid teacher marks only",
              "n":len(errors),"mae_unrounded":mean(errors) if errors else None,
              "exact_after_rounding":mean(e==0 for e in rounded_errors) if errors else None,
              "within_half_after_rounding":mean(e<=.5 for e in rounded_errors) if errors else None}}

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", help="Optional local copy of the unchanged published CSV.")
    parser.add_argument("--mcq-replay",action="store_true",help="Audit Component 2 using the recorded MCQ choices.")
    args = parser.parse_args()
    exams, totals = reference_tables(args.csv)
    result = audit_recorded_choices(exams,totals) if args.mcq_replay else audit(exams)
    print(json.dumps(result, indent=2))
