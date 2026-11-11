"""Reconstruct blinded human-review items from the two frozen written grading runs."""
import argparse
import csv
import hashlib
import json
import sqlite3
from contextlib import closing
from html import escape
from pathlib import Path

from .catalog import KEY, QUESTIONS
from .gemini import PREVIEW, Proposal
from .grading import checked
from .heldout_written import FRESH_PAPERS, PAPERS
from .integration import file_sha256
from .judge import prepare_items
from .retrieval import Index
from .rubric import CRITERIA

PROJECT = Path(__file__).resolve().parents[1]


def recover(db, version, index, transcript, archived):
    q = archived["question_id"]
    answer, = [a for a in transcript["answers"] if a["question_id"] == q]
    hits = index.search(q, answer["text"])
    searches = 1
    if not hits or hits[0]["score"] < 0.25:
        hits = index.search(q)
        searches += 1
    request = dict(question=QUESTIONS[q], reference_answer=KEY[q], marking_guide=CRITERIA[q],
                   student_answer=answer["text"], evidence=[
                       {key: h[key] for key in ("id", "title", "heading", "text")} for h in hits])
    signature = [version, Proposal.model_json_schema(),
                 {"temperature": 0, "max_tokens": 700, "preview": PREVIEW, "format_version": 2}]
    fingerprint = hashlib.sha256(json.dumps([signature, request], ensure_ascii=False, sort_keys=True,
                                           separators=(",", ":")).encode()).hexdigest()
    record = db.execute("SELECT response FROM grade_responses WHERE fingerprint = ?", (fingerprint,)).fetchone()
    if record is None:
        raise ValueError(f"Q{q} has no exact request match in the frozen cache.")
    bundle = json.loads(record[0])
    visible = [h if h["id"] in bundle["opened"] else h | {"text": h["text"][:PREVIEW]} for h in hits]
    row = checked(q, bundle["proposal"], visible, searches)
    for field, original in (("reason", "reason"), ("mark", "proposed_mark"), ("feedback", "feedback"),
                            ("deductions", "deductions")):
        if row[field] != archived[original]:
            raise ValueError(f"Q{q} differs from the frozen run on {field}.")
    return row | {"readings": [answer["text"]]}


def build(folder):
    folder = folder.resolve()
    if folder.is_relative_to(PROJECT):
        raise ValueError("Use the private archive outside the repository.")
    index = Index("section", semantic=False)
    manifests, views = [], {}
    approved = {digest for _, digest in (PAPERS | FRESH_PAPERS).values()}
    with closing(sqlite3.connect((folder / "responses.sqlite").as_uri() + "?mode=ro", uri=True)) as db:
        for cohort, papers in (("pilot", PAPERS), ("fresh", FRESH_PAPERS)):
            report = json.loads((PROJECT / "eval" / f"written_rubric_v2_{cohort}.json").read_bytes())
            if report["index"] != index.version:
                raise ValueError("The archived index does not match this lexical index.")
            archive = json.loads((folder / f"rubric-v2-{cohort}-0.25.json").read_bytes())
            for name, (_, digest) in papers.items():
                path = folder / f"{name}.json"
                if file_sha256(path) != digest:
                    raise ValueError("A vetted answer export changed.")
                transcript = json.loads(path.read_bytes())
                view = folder / f"feedback-view-{name}.json"
                answers = []
                for entry in archive:
                    q = entry["question_id"]
                    # The historical Q32 prompt was replaced by a deterministic rule.
                    if entry["paper"] != name or entry["reason"] != "grounded_proposal" or q == 32:
                        continue
                    answers.append(recover(db, report["model"], index, transcript, entry))
                    manifests.append(dict(id=f"{name}-q{q}", question_id=q,
                                          ocr_json=str(path), teacher_json=str(view)))
                views[view] = {"answers": answers}
    if len(manifests) < 20:
        raise ValueError(f"Only {len(manifests)} eligible items. The course requires at least 20.")
    for path, data in views.items():
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    items = prepare_items(manifests, approved, index)
    with (folder / "feedback-items.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(manifests[0]))
        writer.writeheader()
        writer.writerows(manifests)
    # Plain HTML opens locally and keeps the team's saved labels in a separate file.
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Written feedback review</title>',
             '<style>body{font:18px/1.5 system-ui;max-width:82ch;margin:3em auto;padding:0 1em}'
             'article{border-top:1px solid #999;padding:1em 0}dt{font-weight:bold}dd{margin:0 0 .7em}</style>',
             '<h1>Written feedback review</h1><p>For each item, enter pass or fail and an optional reason '
             'in the accompanying feedback-labels.csv. Pass when the feedback describes the answer accurately, '
             'explains the proposed credit, and is supported by the quoted passage. Use fail for incorrect, '
             'vague, unsupported or misleading feedback. Keep your labels independent of the automated judge.</p>']
    for ident, item in items:
        parts.append(f"<article><h2>{escape(ident)}</h2><dl>")
        for label, field in (("Question", "question"), ("Student answer", "answer"), ("Official reference", "reference"),
                             ("Proposed mark out of two", "mark"), ("Feedback", "feedback"),
                             ("Quoted evidence", "quote"), ("Source passage", "source")):
            parts.append(f"<dt>{label}</dt><dd>{escape(str(item[field]))}</dd>")
        parts.append("<dt>Marking guide</dt><dd><ul>")
        for points, description in item["marking_guide"].values():
            parts.append(f"<li>{points:g} points. {escape(description)}</li>")
        parts.append("</ul></dd></dl></article>")
    (folder / "feedback-review.html").write_text("".join(parts) + "</html>", encoding="utf-8")
    labels = folder / "feedback-labels.csv"
    if not labels.exists():
        with labels.open("x", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["id", "label", "comment"])
            writer.writerows((ident, "", "") for ident, _ in items)
    return len(items)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("private_folder", type=Path)
    print(f"Prepared {build(parser.parse_args().private_folder)} blinded items.")
