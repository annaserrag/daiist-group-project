"""Compare measured policy runs on identical cases with the team's own cost assumptions."""
import argparse
import csv
import json
import math
from pathlib import Path

def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))

def projection(results, pairs, *, hourly_eur, signoff_seconds, review_seconds,
               manual_seconds, wrong_eur, accepted_catch_rate, reviewed_error_rate,
               manual_error_rate, exams_per_day):
    ids = {row["id"] for row in results}
    if not ids or len(ids) != len(results):
        raise ValueError("Expected unique exam results.")
    result_runs = {r.get("run_id") for r in results}
    if len(result_runs) != 1 or result_runs != {r.get("run_id") for r in pairs}:
        raise ValueError("The result and paired-mark files come from different evaluation runs.")
    references = {(r["exam_id"], int(r["question_id"])):r["teacher_mark"] for r in pairs}
    if len(references) != len(pairs) or set(references) != {(i,q) for i in ids for q in range(1,36)}:
        raise ValueError("Cost comparison needs the same complete 35-question sheet for each linked exam.")
    for value in (hourly_eur,signoff_seconds,review_seconds,manual_seconds,wrong_eur,exams_per_day):
        if not math.isfinite(value) or value < 0:
            raise ValueError("Use finite, non-negative unit costs, seconds and volumes.")
    for value in (accepted_catch_rate,reviewed_error_rate,manual_error_rate):
        if not 0 <= value <= 1:
            raise ValueError("Error/catch rates are fractions in [0,1].")
    accepted = [r for r in pairs if r["proposed_mark"]!=""]
    comparable = [r for r in accepted if r["teacher_mark"]!=""]
    wrong = sum(float(r["teacher_mark"]) != float(r["proposed_mark"]) for r in comparable)
    errors = wrong / len(comparable) if comparable else None
    review_share = (len(pairs)-len(accepted))/len(pairs)
    # Old evaluation files always made fresh calls. New files also track billed replay spend.
    prices = [r.get("uncached_cost_eur",r["cost_eur"]) for r in results]
    api_known = all(price not in (None,"") for price in prices)
    api = sum(map(float, prices))/len(pairs)*1000 if api_known else None
    minutes = 1000*(signoff_seconds + review_share*review_seconds)/60
    risk = None
    if errors is not None or not accepted:
        risk = 1000*((1-review_share)*(errors or 0)*(1-accepted_catch_rate) + review_share*reviewed_error_rate)
    return references, {
        "policy_versions":sorted({r["policy"] for r in results}),
        "exams":len(results),"answers":len(pairs),"assessable_proposals":len(comparable),
        "proposals_without_reference":len(accepted)-len(comparable),"unflagged_disagreements":wrong,
        "disagreement_share_among_assessable_proposals":errors,"review_share":review_share,
        "extra_reviews_per_exam":35*review_share,"extra_reviews_per_day":35*review_share*exams_per_day,
        "teacher_minutes_per_1000":minutes,"api_eur_per_1000":api,"estimated_residual_errors_per_1000":risk,
        "estimated_total_eur_per_1000":None if risk is None or api is None else minutes/60*hourly_eur+api+risk*wrong_eur,
        "manual_baseline_eur_per_1000":1000*(manual_seconds/3600*hourly_eur+manual_error_rate*wrong_eur),
    }

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dirs",type=Path,nargs="+",help="Evaluation result folders for the same validation exams")
    p.add_argument("--assumptions",required=True,type=Path,help="Team's measured/assumed input values; see README.")
    args=p.parse_args()
    values=json.loads(args.assumptions.read_text(encoding="utf-8-sig"))
    previous=None
    report={}
    for folder in args.run_dirs:
        refs,row=projection(read_csv(folder/"results.csv"),read_csv(folder/"paired_marks.csv"),**values)
        if previous is not None and refs!=previous:
            raise ValueError("Policies must be compared on identical exam/question references.")
        previous=refs
        report[str(folder)]=row
    print(json.dumps({"assumptions":values,"runs":report},indent=2))
