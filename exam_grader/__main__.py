import argparse
import json
from .catalog import DATA
from .grading import grade_exam
from .retrieval import Index

def main():
    p = argparse.ArgumentParser(description="Exercise the contract on a partial synthetic exam. Never sends an API request.")
    p.add_argument("payload",nargs="?",default=DATA/"vision-example.json")
    p.add_argument("--log",action="store_true",help="Print the privacy-safe template record.")
    args=p.parse_args()
    with open(args.payload,encoding="utf-8-sig") as f:
        payload=json.load(f)
    result=grade_exam(payload,index=Index())
    if args.log:
        print(json.dumps(result["record"],indent=2))
        return
    for r in result["teacher"]["answers"]:
        if r["reason"] != "missing_detection":
            print(f"Q{r['question_id']:02} | {r['mark'] if r['mark'] is not None else 'review':>6} | {r['reason']}")
    t=result["teacher"]
    print(f"{t['review_count']} of 35 require review. Proposed subtotal {t['proposed_subtotal']}; total {t['proposed_total']}. Teacher sign-off required.")
if __name__ == "__main__":
    main()
