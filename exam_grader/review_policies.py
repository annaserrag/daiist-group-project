"""Compare teacher-review workloads on the same written proposals without changing issued marks."""
import argparse
import json
from pathlib import Path
from statistics import mean

from .integration import file_sha256
from .reference_marks import mark_matrix

ROOT = Path(__file__).resolve().parents[1]


def summarize(rows, maximum_to_refer):
    numeric = [r for r in rows if r["proposed_mark"] != ""]
    kept = [r for r in numeric if maximum_to_refer is None or r["proposed_mark"] > maximum_to_refer]
    comparable = [r for r in kept if r["teacher_mark"] != ""]
    errors = [r["proposed_mark"] - r["teacher_mark"] for r in comparable]
    return {"slots":len(rows), "proposals":len(kept), "reviews":len(rows)-len(kept),
            "additional_reviews":len(numeric)-len(kept), "paired_marks":len(errors),
            "exact":errors.count(0), "mean_absolute_error":mean(map(abs,errors)) if errors else None,
            "under_by_at_least_one":sum(e <= -1 for e in errors),
            "over_by_at_least_one":sum(e >= 1 for e in errors)}


def compare(folder):
    if folder.resolve().is_relative_to(ROOT):
        raise ValueError("Use the private evaluation directory.")
    report = {"scope":"Post-grading routing simulation on the 14 written questions other than Q32. "
                      "Candidates refer a numerical proposal at or below a cutoff. A review has no mark or "
                      "assumed improvement. Error counts cover only proposals with numeric teacher marks."}
    configs = (("development", ("pilot", "fresh")), ("independent", ("independent",)))
    if (ROOT / "eval/written_independent_revised.json").exists():
        configs += (("revised_same_independent_papers", ("independent",)),)
    for label, cohorts in configs:
        rows = []
        for cohort in cohorts:
            snapshot = ROOT / "eval" / {
                "pilot":"written_semantic_pilot_frozen.json", "fresh":"written_semantic_frozen_thresholds.json",
                "independent":"written_independent_semantic.json"}[cohort]
            prefix = "current" if cohort == "independent" else "frozen"
            if label == "revised_same_independent_papers":
                snapshot = ROOT / "eval/written_independent_revised.json"
                prefix = "finalized"
            archive = json.loads(snapshot.read_bytes())
            run = next(s for s in archive["thresholds"] if s["threshold"] == 0.40)
            path = folder / f"{prefix}-{cohort}-0.40.json"
            observations = json.loads(path.read_bytes())
            if len(observations) != run["written_slots"] or sum(r["proposed_mark"] == "" for r in observations) != run["review"]:
                raise ValueError(f"The private {cohort} run does not match its archived totals.")
            if cohort == "independent":
                blind = folder / f"{'live' if prefix == 'current' else prefix}-independent-blind.json"
                if file_sha256(blind) != archive["blind_proposals_sha256"]:
                    raise ValueError("The frozen independent proposals changed.")
            rows.extend(r for r in observations if r["question_id"] != 32)
            report.setdefault("source_sha256", {})[path.name] = file_sha256(path)
        if len({(r["paper"],r["question_id"]) for r in rows}) != len(rows):
            raise ValueError("Duplicate paper/question in the comparison.")
        report[label] = {key:summarize(rows, threshold) for key, threshold in (
            ("existing_rule", None), ("refer_zero", 0), ("refer_at_most_one", 1))}
        report[label]["teacher_by_proposal"] = mark_matrix(rows)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_folder", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = json.dumps(compare(args.private_folder), indent=2) + "\n"
    if args.output:
        args.output.write_text(result, encoding="utf-8")
    print(result)
