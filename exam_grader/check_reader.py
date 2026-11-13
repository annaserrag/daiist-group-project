"""Compare a configured PDF reader against an independently checked vision.exam.v1 export."""
import argparse
import json
import os
from collections import defaultdict
from importlib import import_module
from pathlib import Path

import env
from .handoff import inspect_export
from .integration import connect, file_sha256
from .retrieval import Index


def assigned(payload):
    by_question = defaultdict(list)
    for answer in payload["answers"]:
        q = answer["question_id"]
        if q is not None:
            text = " ".join(answer["text"].split())
            by_question[q].append((answer["state"], text))
    return {q: sorted(values) for q, values in by_question.items()}


def compare(pdf, reference, reader, *, written_exam=None):
    gold = json.loads(reference.read_bytes())
    if not inspect_export(gold)["valid"]:
        raise ValueError("Correct the reference export before checking the reader.")
    if gold["exam_status"] == "in_scope" and not assigned(gold):
        raise ValueError("An in-scope reference needs an independently assigned answer.")
    def identity(payload):
        return payload.get("exam_id") or "mendeley-sf3kvjwknt-v1"
    expected_id = written_exam.exam_id if written_exam else "mendeley-sf3kvjwknt-v1"
    if gold["exam_status"] == "in_scope" and identity(gold) != expected_id:
        raise ValueError("Select the written exam definition matching the checked export.")
    observed = {}

    def capture(path):
        payload = reader.read_exam(path)
        observed["contract"] = inspect_export(payload)
        if observed["contract"]["valid"]:
            observed["payload"] = payload
        return payload

    result = connect(pdf, read_exam=capture, vision_version=reader.VERSION, index=Index(), agent=None, written_exam=written_exam)
    report = {"pdf_sha256": file_sha256(pdf), "reference_sha256": file_sha256(reference),
              "reader": reader.VERSION, "reader_failed": result["teacher"]["reader_failed"],
              "contract": observed.get("contract", {"valid": False}),
              "grading_review_count_without_external_calls": result["teacher"]["review_count"],
              "grading_status": result["teacher"]["exam_status"], "exact_handoff": False}
    if "payload" not in observed:
        return report
    actual = observed["payload"]
    expected, detected = assigned(gold), assigned(actual)
    shared = expected.keys() & detected.keys()
    mismatches = sorted(q for q in shared if detected[q] != expected[q])
    missing = sorted(expected.keys() - detected.keys())
    extra = sorted(detected.keys() - expected.keys())
    unassigned = lambda payload: sorted((a["state"], " ".join(a["text"].split())) for a in payload["answers"]
                                       if a["question_id"] is None)
    unassigned_match = unassigned(gold) == unassigned(actual)
    status_match = gold["exam_status"] == actual["exam_status"]
    identity_match = identity(actual) == identity(gold)
    report.update(exam_id_match=identity_match, expected_question_ids=len(expected), exact_question_ids=len(shared) - len(mismatches),
                  missing_ids=missing, unexpected_ids=extra, differing_state_or_text=mismatches,
                  unassigned_readings_match=unassigned_match, exam_status_match=status_match,
                  exact_handoff=status_match and identity_match and unassigned_match and not (mismatches or missing or extra))
    return report


if __name__ == "__main__":
    env.load_env()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pdf", type=Path)
    parser.add_argument("reference", type=Path, help="Teacher-checked export for the identical PDF")
    parser.add_argument("--module", default=os.getenv("GRADE_VISION_MODULE", "vision"))
    parser.add_argument("--written-exam", type=Path, help="Definition for a different written exam")
    args = parser.parse_args()
    if args.pdf.suffix.lower() != ".pdf" or not args.pdf.is_file():
        parser.error("Choose an existing PDF.")
    try:
        module = import_module(args.module)
    except ModuleNotFoundError as error:
        parser.exit(2, f"Reader or its dependency is unavailable: {error.name}\n")
    from .written_exam import WrittenExam
    exam = WrittenExam.load(args.written_exam) if args.written_exam else None
    verdict = compare(args.pdf, args.reference, module, written_exam=exam)
    print(json.dumps(verdict, indent=2))
    raise SystemExit(0 if verdict["exact_handoff"] else 1)
