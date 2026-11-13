"""Recalculate coverage and agreement from the released, anonymized answer-level decisions."""
import argparse
import json
from pathlib import Path
from statistics import mean

EVAL = Path(__file__).resolve().parents[1] / "eval"


def summarize(name, pairs):
    proposed = [p for p in pairs if p["mark"] is not None]
    labeled = [p for p in proposed if p["teacher"] is not None]
    errors = [abs(p["mark"] - p["teacher"]) for p in labeled]
    return {
        "sample": name,
        "answers": len(pairs),
        "proposed": len(proposed),
        "review": len(pairs) - len(proposed),
        "paired": len(labeled),
        "exact": sum(e < 1e-8 for e in errors),
        "within_half": sum(e <= 0.5 + 1e-8 for e in errors),
        "differ_by_one_or_more": sum(e >= 1 - 1e-8 for e in errors),
        "mean_absolute_error": mean(errors) if errors else None,
        "always_full_mae_on_same_proposals": mean(2 - p["teacher"] for p in labeled) if labeled else None,
    }


def load_rows(folder):
    def report(name):
        return json.loads((folder / name).read_bytes())

    ml = report("generalization_ml_four.json")
    readable = [p for p in ml["pairs"] if p["input_state"] == "read"]
    for label, arm in (("ML previous", "alt-baseline"), ("ML unrestricted", "alt-candidate"),
                       ("ML selected", "conservative")):
        yield summarize(label, [dict(mark=p[arm]["mark"], teacher=p["teacher"]) for p in readable])
    confirm = report("generalization_ml_confirmation.json")
    yield summarize("ML confirmation", [p for p in confirm["pairs"] if p["state"] == "read"])
    unt = report("generalization_unt.json")
    for label, cohort in (("UNT development", "schema_replay"), ("UNT fresh answers", "fresh"),
                          ("UNT new questions", "unseen_questions")):
        yield summarize(label, unt[cohort]["pairs"])
    followup = report("review_and_exact_transfer.json")
    for arm in ("baseline", "exact"):
        yield summarize("Offline custom exams " + arm, [
            dict(mark=r[arm]["mark"], teacher=r["teacher"]) for r in followup["transfer"]["pairs"]])
    yield summarize("Offline C++ new questions", followup["new_questions"]["pairs"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", type=Path, default=EVAL)
    args = parser.parse_args()
    print(json.dumps(list(load_rows(args.folder)), indent=2))
