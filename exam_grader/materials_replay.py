"""Replay the 15 public written answers against the three-page Materials fixture."""
import json
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import pipeline
from .catalog import KEY
from .integration import connect, file_sha256
from .materials import add_pdf


def comparison(rows):
    final = [r["attempts"][-1] for r in rows]
    return {
        "questions": len(rows),
        "previous_reviews": sum(r["before_status"] == "review" for r in rows),
        "current_reviews": sum(a["status"] == "review" for a in final),
        "new_reviews": [r["question"] for r,a in zip(rows, final, strict=True)
                        if r["before_status"] == "proposed" and a["status"] == "review"],
        "final_execution_issues": [r["question"] for r,a in zip(rows, final, strict=True)
                                   if a.get("execution_issue") or a["reason"] == "provider_failure"],
    }


def main():
    folder = Path(__file__).resolve().parents[1] / "eval"
    baseline = json.loads((folder / "materials_pdf_coverage.json").read_bytes())
    pdf = folder / baseline["pdf"]
    if file_sha256(pdf) != baseline["sha256"]:
        raise ValueError("The comparison requires the original handout.")
    base, agent, _ = pipeline.resources()
    library, course, _ = add_pdf({}, "ML", pdf, base)
    index = library[course]
    if agent is None or index.version != baseline["index"]:
        raise ValueError("Use live grading and the same semantic index as the baseline.")
    attempts = {q: [] for q in range(21, 36)}
    pending = list(attempts)
    usage = []
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "public-key-answers.json"
        for attempt in range(3):
            if not pending:
                break
            if attempt:
                print(f"Retrying transient service failures on {pending} after 65 seconds.", flush=True)
                time.sleep(65)
            path.write_text(json.dumps(dict(schema_version="vision.exam.v1", exam_status="in_scope",
                answers=[dict(question_id=q, text=KEY[q], confidence=1., id_confidence=1.,
                              state="read") for q in pending])), encoding="utf-8")
            run = connect(path, read_exam=pipeline.read_export, vision_version="public-key-replay",
                          index=index, agent=agent, approved_hashes={file_sha256(path)}, reader_charges_possible=False)
            meter = run["record"]["outputs"]["component_b"]
            usage.append({k:meter[k] for k in ("calls","tokens","cost_eur")})
            for q in pending:
                a = run["teacher"]["answers"][q-1]
                attempts[q].append({k:a[k] for k in ("status","reason","mark","citation","criterion_evidence",
                                      "criteria_needing_evidence","execution_issue","feedback") if k in a})
                print(q, a["status"], a.get("execution_issue", a["reason"]), flush=True)
            pending = [q for q in pending if attempts[q][-1].get("execution_issue") in
                       ("provider_rate_limit", "provider_unavailable")]
    rows = []
    for old in baseline["rows"]:
        q = old["question"]
        before = old.get("retry_after_rate_limit", old)
        rows.append(dict(question=q, baseline_coverage=old["coverage"], before_mark=before["mark"],
                         before_status=before["actual_status"], attempts=attempts[q]))
    output = dict(pdf=pdf.name, sha256=baseline["sha256"], index=index.version, grader=agent.version,
                  evidence_rule=run["record"]["versions"]["evidence_rule"], comparison=comparison(rows),
                  utc=datetime.now(timezone.utc).isoformat(), input=baseline["input"], usage=usage, rows=rows)
    (folder / "materials_criterion_replay.json").write_text(json.dumps(output, indent=2, ensure_ascii=False)
                                                          + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
