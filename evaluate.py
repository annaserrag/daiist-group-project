"""Run team-authored course cases through the same pipeline as the app, without writing app logs."""
import argparse
import csv
import json
import re
from uuid import uuid4
from pathlib import Path
from statistics import mean

import pipeline
from exam_grader.metrics import grading
from exam_grader.metrics import retrieval
from exam_grader.catalog import DATA
from exam_grader.integration import file_sha256
from exam_grader.grading import Policy
from exam_grader.reference_marks import teacher_data, reference_mark, fit_written_baseline, compare_written
from exam_grader.evaluation_cache import EvaluationCache

def rows(path):
    with Path(path).open(encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def write_csv(path, data, columns):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp = path.with_name(path.name+".tmp")
    with tmp.open("w",encoding="utf-8",newline="") as f:
        writer = csv.DictWriter(f,fieldnames=columns)
        writer.writeheader()
        writer.writerows(data)
    tmp.replace(path)

UNAVAILABLE = {"no_index","provider_disabled","provider_failure","missing_frozen_response",
               "invalid_model_output","invalid_contract"}

def match_expected(text, teacher):
    """A team may use structured case expectations or keep narrative expectations for human assessment."""
    try:
        expected = json.loads(text)
    except (TypeError,json.JSONDecodeError):
        return None
    if not isinstance(expected,dict):
        return None
    if not expected or set(expected)-{"exam_status","marks","review_questions","review_reasons","total"}:
        raise ValueError("Unrecognized structured expectation.")
    if teacher.get("reader_failed") or teacher.get("input_type") == "pdf_awaiting_reader":
        return False
    target_status = expected.get("exam_status","in_scope")
    if target_status not in ("in_scope","unreadable","out_of_scope"):
        raise ValueError("Expected exam_status must be in_scope, unreadable or out_of_scope.")
    if teacher["exam_status"] != target_status:
        return False
    by_id = {str(r["question_id"]):r for r in teacher["answers"]}
    checks = []
    if target_status != "in_scope":
        checks.append(teacher["proposed_total"] is None and all(r["mark"] is None for r in by_id.values()))
    elif not any(r["mark"] is not None for r in by_id.values()):
        # A blanket abstention cannot satisfy an ordinary case just because Q17 also abstained.
        # An intentional all-review case needs an explicit, substantive reason on a target.
        reasons = expected.get("review_reasons",{})
        if (not reasons or any(r in UNAVAILABLE for r in reasons.values())
                or all(r["reason"] in {"unreadable","missing_detection","empty_transcription"} for r in by_id.values())):
            return False
    if "exam_status" in expected:
        checks.append(True)
    for q,mark in expected.get("marks",{}).items():
        if q not in by_id or type(mark) not in (int,float) or mark not in ({0,1} if int(q)<=20 else {0,.5,1,1.5,2}):
            raise ValueError("Expected mark is outside the exam scale.")
        checks.append(by_id[q]["mark"]==mark)
    for q in expected.get("review_questions",[]):
        item = by_id[str(q)]
        checks.append(item["status"]=="review" and item["reason"] not in UNAVAILABLE)
    for q,reason in expected.get("review_reasons",{}).items():
        checks.append(by_id[q]["status"]=="review" and by_id[q]["reason"]==reason and reason not in UNAVAILABLE)
    if "total" in expected:
        checks.append(teacher["proposed_total"]==expected["total"])
    if not checks:
        raise ValueError("Empty structured expectation.")
    return all(checks)

def evaluate(cases, out_dir, *, links=None, reference=None, teacher_dir=None, policy=None,
             training_ids=None, evaluation_cache=None, case_references=None):
    if not cases:
        raise ValueError("Add team-authored cases before evaluating.")
    ids=[c["id"] for c in cases]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"[a-zA-Z0-9_-]+",i) for i in ids):
        raise ValueError("Supply unique short case IDs made of letters, digits, hyphens or underscores.")
    if case_references is not None:
        if set(ids) != set(case_references):
            raise ValueError("The case references must cover exactly the supplied case IDs.")
        for case in cases:
            if file_sha256(case["input"]) != case_references[case["id"]]["sha256"]:
                raise ValueError(f"PDF does not match the checked reference for {case['id']}.")
    if links is not None:
        if reference is None or set(links)-set(ids) or set(links.values())-set(reference):
            raise ValueError("Check the case IDs and anonymous source IDs in the links CSV.")
    baseline = None
    if training_ids is not None:
        if not links:
            raise ValueError("A held-out comparison needs linked evaluation exams.")
        baseline = fit_written_baseline(reference, training_ids, links.values())
    results, pairs = [],[]
    run_id = uuid4().hex
    for case in cases:
        result = pipeline.run_views(case["input"], policy=policy, evaluation_cache=evaluation_cache)
        teacher,record = result["teacher"],result["record"]
        if teacher_dir is not None:
            teacher_dir.mkdir(parents=True,exist_ok=True)
            path = teacher_dir / (case["id"]+".json")
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(teacher,ensure_ascii=False),encoding="utf-8")
            tmp.replace(path)
        b = record["outputs"]["component_b"]
        total_usage = record["outputs"]["total_usage"]
        reader_usage = record["outputs"]["reader_usage"]
        failures = sorted({r.get("execution_issue",r["reason"]) for r in teacher["answers"] if r["reason"] in UNAVAILABLE})
        if teacher.get("reader_failed") or b["input_type"] == "pdf_awaiting_reader":
            failures.append("reader_failure" if teacher.get("reader_failed") else "reader_unavailable")
        results.append({"run_id":run_id,"id":case["id"],"kind":case["kind"],"correct":match_expected(case["expected"],teacher),
                        "review_questions":teacher["review_count"],"total":teacher["proposed_total"],
                        "exam_status":teacher["exam_status"],"execution_issues":"|".join(failures),
                        "latency_s":record["latency_s"],"component_2_s":b["latency_s"],
                        "component_1_s":record["features"]["vision_latency_s"],
                        "component_1_tokens":reader_usage["tokens"],"component_1_eur":reader_usage["cost_eur"],
                        "component_2_tokens":b["tokens"] if b["usage_complete"] else None,"component_2_eur":b["cost_eur"],
                        "tokens":record["tokens"] if total_usage["usage_complete"] else None,"usage_complete":total_usage["usage_complete"],
                        "cost_eur":record["cost_eur"] if total_usage["cost_known"] else None,
                        "cache_hits":b["cache_hits"],"model_calls":b["calls"],
                        "uncached_tokens":total_usage["uncached_tokens"],"uncached_cost_eur":total_usage["uncached_cost_eur"],
                        "input_type":b["input_type"],"vision_version":record["versions"]["component_a"],
                        "grader_version":record["versions"]["component_b"],"index":record["versions"]["retrieval"],
                        "policy":record["versions"]["policy"]})
        if case_references is not None:
            reference_case = case_references[case["id"]]
            outcome = compare_case(teacher, record, reference_case)
            results[-1].update(outcome)
            # Wrong or invented question assignments cannot pass a routing/mark check by coincidence.
            results[-1]["correct"] = bool(results[-1]["correct"] and outcome["reader_structure_ok"])
            # An ideal reader with a simple, optimistic marking rule is an explicit comparator.
            # It gives every visible response full credit, and recognizes the true document status.
            naive = [{"question_id":q, "mark":(1 if q <= 20 else 2) if str(q) in reference_case["readings"] else None,
                      "reason":"full_credit" if str(q) in reference_case["readings"] else "missing_detection",
                      "status":"proposed" if str(q) in reference_case["readings"] else "review"}
                     for q in range(1,36)]
            results[-1]["baseline_case_correct"] = match_expected(case["expected"], {
                "answers":naive, "exam_status":reference_case["exam_status"],
                "proposed_total":sum(r["mark"] for r in naive) if all(r["mark"] is not None for r in naive) else None})
        if links and case["id"] in links:
            gold = reference[links[case["id"]]]
            for answer in teacher["answers"]:
                q = answer["question_id"]
                label = reference_mark(q,gold[q])
                pairs.append({"run_id":run_id,"exam_id":case["id"],"question_id":q,
                              "teacher_mark":"" if label is None else str(label),
                              "proposed_mark":"" if answer["mark"] is None else str(answer["mark"]),
                              "baseline_mark":baseline.get(q) if baseline is not None else None})
    out_dir=Path(out_dir)
    write_csv(out_dir/"results.csv",results,list(results[0]))
    known = [r for r in results if r["correct"] is not None]
    costs = [r["cost_eur"] for r in results if r["cost_eur"] is not None]
    summary = {"run_id":run_id,"cases":len(cases),"automatically_assessed":len(known),
               "model_calls":sum(r["model_calls"] for r in results),"cache_hits":sum(r["cache_hits"] for r in results),
               "awaiting_human_assessment":len(cases)-len(known),
               "cases_with_execution_issues":sum(bool(r["execution_issues"]) for r in results),
               "agreement_with_structured_expectations":mean(r["correct"] for r in known) if known else None,
               "mean_review_questions":mean(r["review_questions"] for r in results),
               "mean_exam_latency_s":mean(r["latency_s"] for r in results),
               "known_cost_exams":len(costs),"mean_known_cost_eur":mean(costs) if costs else None}
    if case_references is not None:
        summary["pdf_packet"] = {field:sum(r[field] for r in results) for field in (
            "baseline_case_correct", "reader_structure_ok", "reader_exact_cases", "expected_question_slots", "exact_reading_slots",
            "numeric_references", "comparable_proposals", "exact_marks", "large_mark_errors",
            "full_credit_baseline_exact_on_proposals", "full_credit_baseline_exact_on_all")}
    if pairs:
        write_csv(out_dir/"paired_marks.csv",pairs,list(pairs[0]))
        summary["mcq_vs_published_key_on_recorded_letters"]=grading([r for r in pairs if r["question_id"]<=20])
        summary["written_vs_teacher"]=grading([r for r in pairs if r["question_id"]>20])
        if baseline is not None:
            summary["held_out_written_comparison"] = {"training_exams":len(training_ids),
                "evaluation_source_exams":len(set(links.values())),**compare_written(pairs)}
    temp_summary = out_dir / ("summary-"+run_id+".tmp")
    temp_summary.write_text(json.dumps(summary,indent=2)+"\n",encoding="utf-8")
    temp_summary.replace(out_dir/"summary.json")
    return summary

def compare_case(teacher, record, reference):
    """Score the very same PDF read that produced the marks. Emit counts and question IDs only."""
    rows_by_id = {str(r["question_id"]): r for r in teacher["answers"]}
    expected = reference["readings"]
    seen = {q for q,r in rows_by_id.items() if r["readings"]}
    shared = seen & expected.keys()
    def normalized(text):
        return " ".join(text.casefold().split())
    wrong_states = set()
    wrong_text = set()
    for q in shared:
        row = rows_by_id[q]
        if sorted(row["reading_states"], key=str) != [reference["states"][q]] * len(expected[q]):
            wrong_states.add(q)
        if sorted(map(normalized,row["readings"])) != sorted(map(normalized,expected[q])):
            wrong_text.add(q)
    missing, extra = expected.keys() - seen, seen - expected.keys()
    structure = (not teacher.get("reader_failed") and teacher.get("input_type") == "pdf"
                 and teacher["exam_status"] == reference["exam_status"]
                 and not (missing or extra or wrong_states or record["features"]["unassigned_answers"]))
    marks = {q:m for q,m in reference["reference_marks"].items() if m is not None}
    paired = {q:rows_by_id[q]["mark"] for q in marks if rows_by_id[q]["mark"] is not None}
    def ids(values):
        return "|".join(map(str, sorted(map(int,values))))
    return dict(reader_structure_ok=bool(structure), reader_exact_cases=bool(structure and not wrong_text),
                expected_question_slots=len(expected), exact_reading_slots=len(shared - wrong_states - wrong_text),
                reader_missing_ids=ids(missing), reader_unexpected_ids=ids(extra),
                reader_state_or_count_mismatches=ids(wrong_states), reader_text_mismatches=ids(wrong_text),
                numeric_references=len(marks), comparable_proposals=len(paired),
                exact_marks=sum(m==marks[q] for q,m in paired.items()),
                large_mark_errors=sum(abs(m-marks[q])>=1 for q,m in paired.items()),
                full_credit_baseline_exact_on_proposals=sum(marks[q]==rows_by_id[q]["max_mark"] for q in paired),
                full_credit_baseline_exact_on_all=sum(m==rows_by_id[q]["max_mark"] for q,m in marks.items()))

def component_evidence(base, cases_path):
    """Reproduce the component evidence present in this checkout. Never invent missing labels."""
    source=base/"retrieval_sources.csv"
    files=[cases_path,DATA/"corpus.json",DATA/"Question.txt",DATA/"answerkey.txt",DATA/"instructions.txt"]
    report={"retrieval":{"status":"no worksheet"},"feedback_judge":{"status":"no paired judge results"}}
    if source.is_file():
        files.append(source)
        labels=rows(source)
        report["retrieval"]={"lexical":retrieval(labels,False,partial=True)}
        vectors=[DATA/f"vectors-{design}.json" for design in ("paragraph","section")]
        if all(p.is_file() for p in vectors):
            report["retrieval"]["semantic"]=retrieval(labels,True,partial=True)
            files.extend(vectors)
        else:
            report["retrieval"]["semantic"]={"status":"not run","missing":[p.name for p in vectors if not p.is_file()]}
    human,judged=base/"human_labels.csv",base/"judge_results.csv"
    if judged.is_file() and human.is_file():
        from exam_grader.metrics import frozen_agreement
        report["feedback_judge"]=frozen_agreement(rows(human),rows(judged))
        files.extend([human,judged])
    report["input_sha256"]={p.name:file_sha256(p) for p in files}
    return report

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cases",type=Path,default=Path("eval/cases.csv"))
    p.add_argument("--out",type=Path,default=Path("eval"))
    p.add_argument("--links",type=Path,help="Optional CSV linking id to source_id (e.g. Student_1).")
    p.add_argument("--baseline-train",type=Path,help="CSV with a source_id column for the team's training exams.")
    p.add_argument("--cache",type=Path,help="Private SQLite file outside the repo for repeat Gemini evaluations.")
    p.add_argument("--policy",type=Path,help="JSON object containing Policy fields for this evaluation run.")
    p.add_argument("--teacher-csv",help="Optional local version 1 source for use with --links.")
    p.add_argument("--case-references",type=Path,help="PDF fingerprints, transcriptions and marks for the case packet.")
    p.add_argument("--private-exports",type=Path,metavar="DIR",help="Keep teacher views outside the deployable repo for judge calibration.")
    args=p.parse_args()
    if not args.cases.exists():
        raise SystemExit("Supply your team's eval/cases.csv or use --cases PATH. See README.md.")
    if args.private_exports and args.private_exports.resolve().is_relative_to(Path(__file__).resolve().parent):
        raise SystemExit("Choose a private export folder outside this repo. The course deployer uploads repository files.")
    base=args.cases.parent
    if (args.case_references is None and args.cases.name == "cases.csv"
            and (base/"case_references.json").is_file()):
        args.case_references = base/"case_references.json"
    for attr,filename in (("links","case_links.csv"),("baseline_train","train_sources.csv"),("policy","grade_policy.json")):
        if getattr(args,attr) is None and (base/filename).is_file():
            setattr(args,attr,base/filename)
    components=component_evidence(base,args.cases)
    for optional in (args.links,args.baseline_train,args.policy,args.case_references):
        if optional:
            components["input_sha256"][optional.name]=file_sha256(optional)
    links = None
    if args.links:
        link_rows = rows(args.links)
        links = {r["id"]:r["source_id"] for r in link_rows}
        if len(links)!=len(link_rows):
            raise ValueError("Duplicate case ID in links.")
    result = evaluate(rows(args.cases),args.out,links=links,reference=teacher_data(args.teacher_csv) if links else None,
                      case_references=json.loads(args.case_references.read_bytes()) if args.case_references else None,
                      teacher_dir=args.private_exports,
                      policy=Policy(**json.loads(args.policy.read_text(encoding="utf-8-sig"))) if args.policy else None,
                      training_ids=[r["source_id"].strip() for r in rows(args.baseline_train)] if args.baseline_train else None,
                      evaluation_cache=EvaluationCache(args.cache) if args.cache else None)
    components["run_id"]=result["run_id"]
    tmp=args.out/"component_results.json.tmp"
    tmp.write_text(json.dumps(components,indent=2)+"\n",encoding="utf-8")
    tmp.replace(args.out/"component_results.json")
    print(json.dumps(result,indent=2))
