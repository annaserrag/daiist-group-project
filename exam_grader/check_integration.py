"""Check the two bundled acceptance PDFs through the app's reader, grader and logger."""
import json
from pathlib import Path

import app
import logbook
from evaluate import compare_case, match_expected, rows
from exam_grader.integration import file_sha256
from exam_grader.vision_finetune import VERSION


def main():
    cases = {row["id"]: row for row in rows("eval/cases.csv")}
    references = json.loads(Path("eval/case_references.json").read_text(encoding="utf-8"))
    passed = True
    for case_id in ("c06", "c24"):
        case, reference = cases[case_id], references[case_id]
        pdf = Path(case["input"])
        if file_sha256(pdf) != reference["sha256"]:
            raise SystemExit(f"{case_id}: PDF differs from the published test case.")
        teacher, record = app.process_once(pdf)
        outcome = compare_case(teacher, record, reference)
        answers = {str(a["question_id"]): a for a in teacher["answers"]}
        ok = (record["versions"]["component_a"] == VERSION and outcome["reader_structure_ok"]
              and match_expected(case["expected"], teacher) and not logbook.validate(record)
              and record["id"] in set(logbook.load()["id"]))
        for q in reference["readings"]:
            answer = answers[q]
            print(f"  Q{q}: {answer['mark']} ({answer['reason']})")
        if case_id == "c06":
            citation = answers["21"]["citation"] or {}
            print("  Q21 source:", citation.get("title"), citation.get("url"))
            ok = ok and bool(citation.get("id") and citation.get("quote"))
        print(f"{case_id}: {'PASS' if ok else 'FAIL'} | {record['versions']['component_a']} | "
              f"{record['latency_s']:.1f}s | {record['tokens']} tokens", flush=True)
        passed = passed and ok
    if not passed:
        raise SystemExit("Integration check failed. For written referrals check the reason, Gemini key, quota and approved PDF hash.")


if __name__ == "__main__":
    main()
