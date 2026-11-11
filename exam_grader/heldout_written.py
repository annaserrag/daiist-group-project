"""Replay two fixed, de-identified sets of written responses through the Component 2 pipeline."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path
from statistics import mean
from time import monotonic, sleep

import pipeline
from .evaluation_cache import EvaluationCache
from .grading import Policy, grade_exam
from .hyperparameters import VERSION as Q32_VERSION
from .reference_marks import compare_written, fit_written_baseline, reference_mark, teacher_data

# Hashes of redacted answer-only JSON, independent of marks, fixed before scoring.
PAPERS = {
    "paper-a": ("Student_17", "5fa4fc3c3aa3073e9eea8fbf1010c24aa9e6b2e9affb22e597cbbf64bc34bdb9"),
    "paper-b": ("Student_42", "14b929bdca199c99d95ee0a9675b9005e6d02e8d22f18ad051cfb1632c0003cd"),
}
FRESH_PAPERS = {
    "paper-c": ("Student_8", "2ab4594302bbba54b3244f90fe0b719695f2caee6ec26fa82193f02ef88f4991"),
    "paper-d": ("Student_36", "05b6e6dcc25a8e3cbb98611dfa2deb0aff11c293ffa1077be495c1212db499a7"),
}

class PacedCache(EvaluationCache):
    """Space uncached grading requests while replaying completed responses immediately."""
    next_request = 0.0

    def get_or_generate(self, signature, request, generate):
        def paced_generate():
            sleep(max(0, self.next_request - monotonic()))
            try:
                return generate()
            except Exception as error:
                # Provider messages can contain submitted text. Only report the class.
                code = getattr(error, "code", None)
                detail = f", HTTP {code}" if isinstance(code, int) else ""
                print(f"Grading call failed ({type(error).__name__}{detail}).", flush=True)
                raise
            finally:
                self.next_request = monotonic() + 12
        return super().get_or_generate(signature, request, paced_generate)

def compare_revisions(original, revised):
    previous = {(r["paper"], r["question_id"]): r for r in original}
    if len(previous) != len(original) or previous.keys() != {(r["paper"], r["question_id"]) for r in revised}:
        raise ValueError("Pilot papers and question IDs must match the original run.")
    errors = []
    for row in revised:
        old = previous[row["paper"], row["question_id"]]
        teacher = row["teacher_mark"]
        if old["teacher_mark"] != teacher:
            raise ValueError("Teacher reference changed between rubric versions.")
        if teacher != "" and old["proposed_mark"] != "" and row["proposed_mark"] != "":
            errors.append((abs(old["proposed_mark"] - teacher), abs(row["proposed_mark"] - teacher)))
    def metrics(values):
        return {"exact": values.count(0), "within_half": sum(e <= .5 for e in values),
                "mae": mean(values) if values else None}
    return {"paired_marks": len(errors), "original": metrics([a for a, _ in errors]),
            "revised": metrics([b for _, b in errors]), "closer": sum(b < a for a, b in errors),
            "further": sum(b > a for a, b in errors), "unchanged_error": sum(b == a for a, b in errors)}

def replay_q32(folder):
    """Compare the count rule with the archived written runs using the original inputs."""
    folder = folder.resolve()
    if folder.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Choose the private transcription directory.")
    exams = teacher_data()
    rows = []
    for cohort, papers in (("pilot", PAPERS), ("fresh", FRESH_PAPERS)):
        archived = json.loads((folder / f"rubric-v2-{cohort}-0.25.json").read_text(encoding="utf-8"))
        for name, (source, digest) in papers.items():
            path = folder / f"{name}.json"
            if pipeline.file_sha256(path) != digest:
                raise ValueError(f"Changed transcription for {name}.")
            answer = grade_exam(pipeline.read_export(path))["teacher"]["answers"][31]
            previous = next(r for r in archived if r["paper"] == name and r["question_id"] == 32)
            rows.append({"paper": source, "credited_examples": answer["credited_examples"],
                         "mark": answer["mark"], "feedback": answer["feedback"], "decision": answer["reason"],
                         "teacher_mark": reference_mark(32, exams[source][32]),
                         "previous_mark_at_0.25": previous["proposed_mark"] if previous["proposed_mark"] != "" else None,
                         "previous_decision_at_0.25": previous["reason"], "previous_feedback": previous["feedback"]})
    (folder / "q32-count-comparison.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    errors = [abs(r["mark"] - r["teacher_mark"]) for r in rows if r["mark"] is not None and r["teacher_mark"] is not None]
    return {"dataset": "10.17632/sf3kvjwknt.1", "question_id": 32, "rule": Q32_VERSION,
            "papers": len(rows), "proposals": sum(r["mark"] is not None for r in rows),
            "teacher_marks": sum(r["teacher_mark"] is not None for r in rows), "paired_marks": len(errors),
            "exact": errors.count(0), "within_half": sum(e <= .5 for e in errors),
            "mean_absolute_error": mean(errors) if errors else None,
            "scope": "Replay of four previously evaluated papers using fixed manual transcriptions."}

def run(folder, *, cohort="pilot", frozen=False, teacher_csv=None):
    papers = FRESH_PAPERS if cohort == "fresh" else PAPERS
    thresholds = (0.40, 0.25)
    folder = folder.resolve()
    if folder.is_relative_to(Path(__file__).resolve().parents[1]):
        raise ValueError("Use the private transcription directory outside the repository.")
    for name, (_, expected) in papers.items():
        if pipeline.file_sha256(folder / f"{name}.json") != expected:
            raise ValueError(f"Changed transcription for {name}.")
    if not frozen and not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY locally before running the live evaluation.")
    os.environ["GRADE_APPROVED_HASHES"] = " ".join(digest for _, digest in papers.values())
    pipeline.resources.cache_clear()
    index, agent, _ = pipeline.resources()
    cache = EvaluationCache(folder / "responses.sqlite", read_only=True) if frozen else PacedCache(folder / "responses.sqlite")
    if agent is None:
        from .gemini import Gemini
        agent = Gemini(frozen_cache=cache)
    exams = teacher_data(teacher_csv)
    holdout = {source for source, _ in papers.values()}
    if cohort == "fresh":
        holdout.update(source for source, _ in PAPERS.values())
    baseline = fit_written_baseline(exams, sorted(exams.keys() - holdout), holdout)
    report = {"dataset": "10.17632/sf3kvjwknt.1", "exams": len(papers), "cohort": cohort,
              "baseline_training_exams": len(exams) - len(holdout),
              "scope": "All 30 written slots on two papers. Visible answers were manually transcribed and assigned confidence 1. Unlocated answers go to review.",
              "model": agent.version, "index": index.version, "q32_rule": Q32_VERSION,
              "response_mode": "frozen-cache-only" if frozen else "live-with-cache", "thresholds": []}
    for threshold in thresholds:
        pairs = []
        workload = []
        for name, (source, _) in papers.items():
            result = pipeline.run_views(folder / f"{name}.json", policy=Policy(min_similarity=threshold),
                                        evaluation_cache=cache)
            record = result["record"]
            meter = record["outputs"]["component_b"]
            workload.append({"paper":name, "elapsed_seconds_including_pacing":record["latency_s"],
                             "tokens_billed_this_run":record["tokens"], **{k:meter[k] for k in (
                                 "calls", "cache_hits", "uncached_tokens", "uncached_cost_eur", "usage_complete")}})
            for row in result["teacher"]["answers"]:
                q = row["question_id"]
                if q <= 20:
                    continue
                teacher = reference_mark(q, exams[source][q])
                pairs.append({"paper": name, "question_id": q,
                              "teacher_mark": teacher if teacher is not None else "",
                              "baseline_mark": baseline[q], "reason": row["reason"],
                              "feedback": row["feedback"], "deductions": row.get("deductions", []),
                              "proposed_mark": row["mark"] if row["mark"] is not None else "",
                              "retrieval_score": row["retrieval_score"]})
        reasons = Counter(row["reason"] for row in pairs)
        review = sum(row["proposed_mark"] == "" for row in pairs)
        summary = {"threshold": threshold, "review": review, "review_share": review / len(pairs),
                   "reasons": dict(reasons), "evaluation_workload":workload, **compare_written(pairs)}
        marked = [row for row in pairs if row["proposed_mark"] != ""]
        partial = [row for row in marked if row["proposed_mark"] < 2]
        summary["feedback_check"] = {"below_full_credit": len(partial), "fully_accounted_for": sum(
            sum(d["points"] for d in row["deductions"]) == 2 - row["proposed_mark"]
            and all(d["reason"].strip() in row["feedback"] for d in row["deductions"]) for row in partial)}
        if cohort == "pilot":
            original = json.loads((folder / f"comparison-{threshold:.2f}.json").read_text(encoding="utf-8"))
            summary["change_on_same_answers"] = compare_revisions(original, pairs)
        report["thresholds"].append(summary)
        prefix = "frozen" if frozen else "current"
        (folder / f"{prefix}-{cohort}-{threshold:.2f}.json").write_text(json.dumps(pairs, indent=2), encoding="utf-8")
        print(f"{threshold:.2f}: {dict(reasons)}", flush=True)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_folder", type=Path)
    parser.add_argument("--cohort", choices=("pilot", "fresh", "q32"), default="pilot")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen", action="store_true", help="Use saved model responses only across both thresholds.")
    parser.add_argument("--teacher-csv",type=Path, help="Local copy of the published, checksum-pinned reference.")
    args = parser.parse_args()
    archive = Path(__file__).resolve().parents[1] / "eval"
    if args.output.resolve() in {archive / name for name in (
            "heldout_written_pilot.json", "written_rubric_v2_pilot.json", "written_rubric_v2_fresh.json")}:
        parser.error("Choose a new output path to preserve the earlier evaluations.")
    result = replay_q32(args.private_folder) if args.cohort == "q32" else run(args.private_folder, cohort=args.cohort,
                                                                             frozen=args.frozen, teacher_csv=args.teacher_csv)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
