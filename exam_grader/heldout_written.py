"""Compare fixed, de-identified written cohorts through the Component 2 pipeline."""
import argparse
from collections import Counter
import json
import os
import re
from pathlib import Path
from statistics import mean
from tempfile import NamedTemporaryFile
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
INDEPENDENT_PAPERS = {
    "paper-e": ("Student_41", "0a56f1f88acf63799a8825a39cfa5bf7068bed7dfa4c24cce344699ec6e8e067"),
    "paper-f": ("Student_7", "0e5a12234c7739a253d65511138efbb36135a9163f7b06c51c5ceb94be57b5c3"),
}

VALIDATION_PAPERS = {
    "paper-g": ("Student_13", "6dbe03d7fd76353225909c623fadde65a48e8772822c20b3fca18f09541039aa"),
    "paper-h": ("Student_18", "f1552526218433976d63ac7b1b445495b686ee7a69b6a4975b62bfc678858b12"),
    "paper-i": ("Student_35", "22d31a4b7eb26093de4e860cab832e0c2f4d1a5bcd37212644fd610bc1833a0a"),
    "paper-j": ("Student_22", "5e1a9c357d4351c2f838e08a10597d21cd30e3b6507f07e97b580ffb61202e0b"),
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


def save_snapshot(path, rows):
    """Publish a complete private file once. A revised run needs its own label."""
    if path.is_file():
        if json.loads(path.read_bytes()) != rows:
            raise ValueError(f"Existing {path.name} differs. Choose a new --run-label to preserve both runs.")
        return
    with NamedTemporaryFile("w", dir=path.parent, encoding="utf-8", delete=False) as pending:
        temporary = Path(pending.name)
        json.dump(rows,pending,indent=2)
    try:
        # Hard-link publication gives simultaneous runs a single winner without overwriting it.
        try:
            os.link(temporary, path)
        except FileExistsError:
            if json.loads(path.read_bytes()) != rows:
                raise ValueError(f"A different run just published {path.name}. Choose a new --run-label.") from None
    finally:
        temporary.unlink(missing_ok=True)

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

def compare_lookup(folder, teacher_csv=None):
    """Paired diagnostic on the four missing responses identified in the frozen development replays."""
    from .gemini import Gemini
    index, configured, _ = pipeline.resources()
    if configured is None:
        raise ValueError("Enable Gemini for the tool-policy diagnostic.")
    cache = PacedCache(folder / "responses.sqlite")
    agents = {"original":Gemini(finalize_lookup=False).for_evaluation(cache), "final_after_lookup":Gemini(finalize_lookup=True).for_evaluation(cache)}
    papers = PAPERS | FRESH_PAPERS
    trials = []
    for i, (name,q) in enumerate((("paper-a",24), ("paper-a",25), ("paper-b",30), ("paper-c",25))):
        path = folder / f"{name}.json"
        if pipeline.file_sha256(path) != papers[name][1]:
            raise ValueError(f"Changed transcription for {name}.")
        sheet = pipeline.read_export(path)
        sheet["answers"] = [a for a in sheet["answers"] if a["question_id"] == q]
        row = {"paper":name, "q":q}
        for method in (tuple(agents) if i % 2 == 0 else tuple(reversed(agents))):
            out = grade_exam(sheet, index=index, agent=agents[method], allow_remote=True)
            item = out["teacher"]["answers"][q - 1]
            meter = out["record"]["outputs"]["component_b"]
            row[method] = {"mark":item["mark"], "route":item["reason"], "calls":meter["calls"], "hits":meter["cache_hits"]}
        trials.append(row)
    exams = teacher_data(teacher_csv)
    for row in trials:
        row["teacher_mark"] = reference_mark(row["q"], exams[papers[row["paper"]][0]][row["q"]])
    (folder/"lookup-comparison.json").write_text(json.dumps(trials,indent=2),encoding="utf-8")
    report = {"scope":"Four selected missing responses on development papers. Two configurations share "
                      "one passage lookup and two model calls per answer. Order alternates by item.",
              "index":index.version, "n":len(trials)}
    for method, agent in agents.items():
        paired = [r for r in trials if r[method]["mark"] is not None and r["teacher_mark"] is not None]
        report[method] = {"version":agent.version, "routes":dict(Counter(r[method]["route"] for r in trials)),
                          "calls":sum(r[method]["calls"] for r in trials), "cache_hits":sum(r[method]["hits"] for r in trials),
                          "paired":len(paired), "exact":sum(r[method]["mark"] == r["teacher_mark"] for r in paired)}
    return report


def compare_q22_context(folder, teacher_csv=None):
    """Evaluate the explicit-numbers variant on four development papers, holding the Q34 gate fixed."""
    papers = PAPERS | FRESH_PAPERS
    exports = {}
    for name, (_, expected) in papers.items():
        path = folder / f"{name}.json"
        if pipeline.file_sha256(path) != expected:
            raise ValueError(f"Changed transcription for {name}.")
        payload = pipeline.read_export(path)
        exports[name] = payload | {"answers":[row for row in payload["answers"] if row["question_id"] == 22]}
    index, provider, _ = pipeline.resources()
    if provider is None:
        raise ValueError("Set GEMINI_API_KEY to evaluate the self-contained Q22 variant.")
    provider = provider.for_evaluation(PacedCache(folder / "responses.sqlite"))
    rows = []
    for name, payload in exports.items():
        default = grade_exam(payload, index=index, agent=provider, allow_remote=True)["teacher"]["answers"][21]
        variant = grade_exam(payload, policy=Policy(figure_questions=(34,)), index=index,
                             agent=provider, allow_remote=True)["teacher"]["answers"][21]
        rows.append({"paper":name,"default_route":default["reason"],"default_mark":default["mark"],
                     "candidate_route":variant["reason"],"candidate_mark":variant["mark"]})
    exams = teacher_data(teacher_csv)
    for row in rows:
        row["teacher_mark"] = reference_mark(22, exams[papers[row["paper"]][0]][22])
    (folder / "q22-context-comparison.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    paired = [r for r in rows if r["candidate_mark"] is not None and r["teacher_mark"] is not None]
    return {"scope":"Q22 only on four previously evaluated transcriptions. Candidate permits grading when the "
                    "fixed question supplies the train/test accuracies. Q34 remains gated in both configurations.",
            "index":index.version, "model":provider.version, "threshold":Policy().min_similarity,
            "papers":len(rows), "default_review":sum(r["default_mark"] is None for r in rows),
            "candidate_review":sum(r["candidate_mark"] is None for r in rows),
            "paired_candidate_marks":len(paired), "exact":sum(r["candidate_mark"] == r["teacher_mark"] for r in paired),
            "candidate_routes":dict(Counter(r["candidate_route"] for r in rows))}


def run(folder, *, cohort="pilot", frozen=False, teacher_csv=None, run_label=None):
    papers = {"pilot":PAPERS, "fresh":FRESH_PAPERS, "independent":INDEPENDENT_PAPERS,
              "validation":VALIDATION_PAPERS}[cohort]
    independent = cohort in {"independent", "validation"}
    thresholds = (0.40,) if independent else (0.40, 0.25)
    if run_label and not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,39}", run_label):
        raise ValueError("Use a short lowercase run label containing letters, numbers or hyphens.")
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
        agent = Gemini(frozen_cache=cache, finalize_lookup=pipeline.final_after_lookup())
    holdout = {source for source, _ in papers.values()}
    if cohort == "fresh" or independent:
        holdout.update(source for source, _ in PAPERS.values())
    if independent:
        holdout.update(source for source, _ in FRESH_PAPERS.values())
    if cohort == "validation":
        holdout.update(source for source, _ in INDEPENDENT_PAPERS.values())
    report = {"dataset": "10.17632/sf3kvjwknt.1", "exams": len(papers), "cohort": cohort,
              "scope": "All 30 written slots on two papers. Visible answers were manually transcribed and assigned confidence 1. Unlocated answers go to review.",
              "model": agent.version, "index": index.version, "q32_rule": Q32_VERSION,
              "response_mode": "frozen-cache-only" if frozen else "live-with-cache", "thresholds": []}
    if run_label:
        report["run_label"] = run_label
    if independent:
        report["scope"] = ("28 written slots on two additional papers. Q32 excluded because all 50 Q32 responses "
                           "were examined in development. Answer-only transcriptions. Uncertain readings withheld.")
        report["selection"] = ("First two remaining paper IDs sorted by SHA-256 of "
                               "component2-independent-written-v1/{id}, excluding the four earlier papers.")
        report["transcription_sha256"] = {name:digest for name, (_,digest) in papers.items()}
    if cohort == "validation":
        frozen_model = "gemini-3.5-flash-lite:prompt-ae86c803e1c8:rubric-683efac0520a:final-after-lookup-v1"
        frozen_index = "section:46c52674828b:gemini-embedding-2:hybrid-v3:multipart-v1"
        if (agent.version, index.version) != (frozen_model, frozen_index):
            raise ValueError("This validation cohort requires the archived model, rubric and retrieval index from c4b76bb.")
        report["scope"] = ("56 written slots across four additional uncorrected papers, manually transcribed. "
                           "Q32 excluded. Unlocated responses count toward review. Grader fixed at c4b76bb.")
        report["selection"] = ("Next four IDs in the original component2-independent-written-v1/{id} "
                               "SHA-256 order after the first two, excluding the original four development papers.")
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
                if q <= 20 or (independent and q == 32):
                    continue
                pairs.append({"paper": name, "question_id": q,
                              "reason": row["reason"],
                              **({"execution_issue":row["execution_issue"]} if cohort == "validation" and "execution_issue" in row else {}),
                              "feedback": row["feedback"], "deductions": row.get("deductions", []),
                              "proposed_mark": row["mark"] if row["mark"] is not None else "",
                              "retrieval_score": row["retrieval_score"]})
        if independent:
            snapshot = folder / f"{run_label or ('frozen' if frozen else 'live')}-{cohort}-blind.json"
            save_snapshot(snapshot, pairs)
            report["blind_proposals_sha256"] = pipeline.file_sha256(snapshot)
        exams = teacher_data(teacher_csv)
        baseline = fit_written_baseline(exams, sorted(exams.keys() - holdout), holdout)
        report["baseline_training_exams"] = len(exams) - len(holdout)
        for pair in pairs:
            q = pair["question_id"]
            teacher = reference_mark(q, exams[papers[pair["paper"]][0]][q])
            pair["teacher_mark"] = teacher if teacher is not None else ""
            pair["baseline_mark"] = baseline[q]
        reasons = Counter(row["reason"] for row in pairs)
        review = sum(row["proposed_mark"] == "" for row in pairs)
        summary = {"threshold": threshold, "review": review, "review_share": review / len(pairs),
                   "reasons": dict(reasons), "evaluation_workload":workload, **compare_written(pairs)}
        marked = [row for row in pairs if row["proposed_mark"] != ""]
        partial = [row for row in marked if row["proposed_mark"] < 2]
        if cohort == "validation":
            summary["execution_issues"] = dict(Counter(r["execution_issue"] for r in pairs if "execution_issue" in r))
        summary["feedback_check"] = {"below_full_credit": len(partial), "fully_accounted_for": sum(
            sum(d["points"] for d in row["deductions"]) == 2 - row["proposed_mark"]
            and all(d["reason"].strip() in row["feedback"] for d in row["deductions"]) for row in partial)}
        if cohort == "pilot":
            original = json.loads((folder / f"comparison-{threshold:.2f}.json").read_text(encoding="utf-8"))
            summary["change_on_same_answers"] = compare_revisions(original, pairs)
        report["thresholds"].append(summary)
        prefix = run_label or ("frozen" if frozen else "current")
        save_snapshot(folder / f"{prefix}-{cohort}-{threshold:.2f}.json", pairs)
        print(f"{threshold:.2f}: {dict(reasons)}", flush=True)
    return report

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_folder", type=Path)
    parser.add_argument("--cohort", choices=("pilot", "fresh", "independent", "validation", "q22", "q32", "lookup"), default="pilot")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--frozen", action="store_true", help="Use saved model responses only across both thresholds.")
    parser.add_argument("--teacher-csv",type=Path, help="Local copy of the published, checksum-pinned reference.")
    parser.add_argument("--run-label",help="Keep private per-answer files from distinct experimental runs separate.")
    args = parser.parse_args()
    archive = Path(__file__).resolve().parents[1] / "eval"
    if args.output.exists() and args.output.resolve().parent == archive.resolve():
        parser.error("Choose a new output path to preserve the published evaluation.")
    if args.cohort == "q32":
        result = replay_q32(args.private_folder)
    elif args.cohort == "q22":
        result = compare_q22_context(args.private_folder, args.teacher_csv)
    elif args.cohort == "lookup":
        result = compare_lookup(args.private_folder, args.teacher_csv)
    else:
        result = run(args.private_folder, cohort=args.cohort, frozen=args.frozen, teacher_csv=args.teacher_csv,
                     run_label=args.run_label)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
