"""Exercise the custom exam pathway on six synthetic physics answers with declared expectations."""
from pathlib import Path
import argparse
import json
import os
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pipeline
from .build import vector
from .grading import Policy
from .integration import file_sha256
from .materials import add_pdf
from .written_exam import WrittenExam

DATA = Path(__file__).resolve().parents[1] / "eval"


def run(cache=None):
    exam = WrittenExam.load(DATA / "waves-exam.json")
    examples = json.loads((DATA / "waves-examples.json").read_bytes())
    base, agent, _ = pipeline.resources()
    if agent is None or not base.vectors:
        raise ValueError("The live transfer demonstration requires Gemini and a semantic index.")
    library, name, _ = add_pdf({}, "Waves", DATA / "waves-course.pdf", base)
    queries = {str(q.question_id): vector("task: question answering | query: " + q.question,
                                          os.environ["GEMINI_API_KEY"]) for q in exam.questions}
    results = []
    with TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.json"
        for case in examples:
            answers = [dict(question_id=q.question_id, text=text, state="read", confidence=1, id_confidence=1)
                       for q, text in zip(exam.questions, case["answers"], strict=True)]
            path.write_text(json.dumps(dict(schema_version="vision.exam.v1", exam_id=exam.exam_id,
                                            exam_status="in_scope", answers=answers)), encoding="utf-8")
            # These files are freshly generated from the bundled public synthetic examples.
            with patch.object(pipeline, "resources", return_value=(base, agent, {file_sha256(path)})):
                view = pipeline.run_views(path, written_exam=exam, query_vectors=queries, course_index=library[name],
                                          evaluation_cache=cache, policy=Policy())
            for row, expected in zip(view["teacher"]["answers"], case["expected"], strict=True):
                cite = row.get("citation") or {}
                results.append(dict(case=case["id"], question=row["question_id"], expected=expected,
                                    proposed=row["mark"], reason=row["reason"], page=cite.get("page"),
                                    criteria=row.get("criterion_evidence", {})))
    return {"exam_id":exam.exam_id, "scope":"synthetic engineering check of a different written exam",
            "questions":results}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, help="Private response cache outside the repository.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    from .heldout_written import PacedCache
    report = run(PacedCache(args.cache) if args.cache else None)
    data = json.dumps(report, indent=2, ensure_ascii=False)
    if args.output:
        args.output.write_text(data + "\n", encoding="utf-8")
    else:
        print(data)
