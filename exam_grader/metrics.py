"""Calculate metrics from team-owned annotations. Never generate a gold set."""
import argparse
import csv
import json
from collections import defaultdict
from .catalog import DATA, QUESTIONS
from .retrieval import Index
from .annotations import validate
from .integration import file_sha256

def load(path):
    with open(path,encoding="utf-8-sig",newline="") as f:
        return list(csv.DictReader(f))

def retrieval(rows, semantic, partial=False, *, written_only=False, multipart=True, teaching_notes=False):
    labels = validate(rows)
    if written_only:
        labels = {q: value for q, value in labels.items() if q > 20}
    pending = [q for q,(state,_,_) in labels.items() if state=="open"]
    gaps = [q for q,(state,_,_) in labels.items() if state=="gap"]
    missing_context = [q for q,(state,_,_) in labels.items() if state=="missing_context"]
    if pending and not partial:
        raise ValueError(f"{len(pending)} questions still open. Finish the worksheet or use --partial for a diagnostic.")
    mapped = {q:set(ids) for q,(state,ids,_) in labels.items() if state=="supported"}
    out = {"status":"partial" if pending else "complete","search":"semantic" if semantic else "lexical",
           "scope":"written" if written_only else "all", "questions_in_scope":len(labels),
           "reviewed":len(labels)-len(pending),"supported":len(mapped),"open_questions":pending,"corpus_gaps":gaps,
           "missing_exam_context_excluded":missing_context}
    for design in ("paragraph","section"):
        idx = Index(design,semantic,multipart=multipart,teaching_notes=teaching_notes)
        ranking = {}
        for q,good in mapped.items():
            rank = [h["source"] for h in idx.search(q,k=5)]
            ranking[q] = next((i for i,page in enumerate(rank,1) if page in good),None)
        out[design] = {"index":idx.version,"first_supporting_chunk_rank_in_top5":ranking}
        if semantic:
            out[design]["vectors_sha256"] = file_sha256(DATA / f"vectors-{design}.json")
        for k in (1,3,5):
            hits = sum(rank is not None and rank<=k for rank in ranking.values())
            out[design][f"hit@{k}"] = hits/len(mapped) if mapped else None
            out[design][f"hit@{k}_including_reviewed_gaps"] = hits/(len(mapped)+len(gaps)) if mapped or gaps else None
    return out

def grading(rows):
    seen=set(); acc=[]; reviewed=0; excluded=0
    for r in rows:
        q=int(r["question_id"]); ident=(r["exam_id"],q)
        if not r["exam_id"] or ident in seen or q not in QUESTIONS:
            raise ValueError("Invalid or duplicate exam/question.")
        seen.add(ident)
        allowed={0,1} if q<=20 else {0,.5,1,1.5,2}
        def mark(s):
            try:
                x=float(s)
                return x if x in allowed else None
            except ValueError:
                return None
        pred=mark(r["proposed_mark"]); gold=mark(r["teacher_mark"])
        if r["proposed_mark"].strip() and pred is None:
            raise ValueError("Invalid proposed mark.")
        reviewed += pred is None
        if gold is None:
            excluded+=1
        elif pred is not None:
            acc.append(abs(gold-pred))
    return {"n":len(rows),"reviewed":reviewed,"review_share":reviewed/len(rows) if rows else None,
            "missing_or_invalid_teacher_marks":excluded,"scored_proposals":len(acc),
            "exact_agreement_on_scored":sum(d==0 for d in acc)/len(acc) if acc else None,
            "within_half_on_scored":sum(d<=.5 for d in acc)/len(acc) if acc else None,
            "unflagged_disagreements":sum(d>0 for d in acc)}

def agreement(rows):
    if len({r["id"] for r in rows}) != len(rows) or len(rows)<20:
        raise ValueError("Need at least 20 unique paired team/judge labels.")
    if any(not r["id"] or r["human_label"] not in ("pass","fail") or r["judge_label"] not in ("pass","fail") for r in rows):
        raise ValueError("Use pass or fail for both labels.")
    matrix=defaultdict(int)
    for r in rows:
        matrix[r["human_label"]+"/"+r["judge_label"]]+=1
    return {"n":len(rows),"exact_agreement":sum(r["human_label"]==r["judge_label"] for r in rows)/len(rows),
            "human/judge":dict(matrix)}

def compare_labels(judged, human):
    ids = [r["id"].strip() for r in human]
    if len(ids) != len(set(ids)) or set(ids) != set(judged):
        raise ValueError("Judge items and human labels must contain the same unique IDs.")
    paired = [{"id":ident,"human_label":r["label"].strip().lower(),"judge_label":judged[ident]}
              for ident,r in zip(ids,human)]
    return agreement(paired)

def frozen_agreement(human, predictions):
    by_id = {r["id"]:r["judge_label"] for r in predictions}
    if len(predictions) != len(by_id):
        raise ValueError("Duplicate judge result ID.")
    return compare_labels(by_id,human)

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=["retrieval","grading","agreement"])
    p.add_argument("csv")
    p.add_argument("--judge-results",help="Join frozen judge predictions to a completed human-label CSV by ID, offline.")
    p.add_argument("--semantic",action="store_true")
    p.add_argument("--teaching-notes", action="store_true", help="Evaluate the production built-in study notes with the original labels.")
    p.add_argument("--legacy-ranking",action="store_true",help="Reproduce rankings before multipart evidence selection.")
    p.add_argument("--partial",action="store_true",help="Diagnostic for reviewed questions only; no full-set conclusion.")
    p.add_argument("--written-only",action="store_true",help="Score retrieval on Q21 through Q35.")
    a=p.parse_args()
    rows=load(a.csv)
    if a.written_only and a.mode != "retrieval":
        p.error("--written-only applies to retrieval")
    if a.judge_results and a.mode != "agreement":
        p.error("--judge-results applies to agreement")
    if a.mode == "retrieval":
        result = retrieval(rows,a.semantic,a.partial,written_only=a.written_only,multipart=not a.legacy_ranking, teaching_notes=a.teaching_notes)
    elif a.mode == "grading":
        result = grading(rows)
    else:
        result = frozen_agreement(rows,load(a.judge_results)) if a.judge_results else agreement(rows)
    print(json.dumps(result,indent=2))
