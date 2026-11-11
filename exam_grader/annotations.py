"""Human page-level retrieval judgments. Never infer a label from a retrieval rank."""
import csv
import json
from pathlib import Path
from tempfile import NamedTemporaryFile
from threading import Lock
from .catalog import DATA, QUESTIONS

SHEET = DATA.parents[1] / "eval" / "retrieval_sources.csv"
FIELDS = ("question_id", "question", "acceptable_sources", "review_status", "note")
PAGES = {p["source"]: p["title"] for p in json.loads((DATA/"corpus.json").read_bytes())}
LOCK = Lock()

def decision(row):
    ids = row["acceptable_sources"].strip().split("|") if row["acceptable_sources"].strip() else []
    status = (row.get("review_status") or ("supported" if ids else "open")).strip()
    note = (row.get("note") or "").strip()
    if (status not in {"supported","gap","missing_context","open"} or bool(ids) != (status == "supported")
            or set(ids)-PAGES.keys() or len(set(ids)) != len(ids) or (status in {"gap","missing_context"} and not note)):
        raise ValueError(f"Q{row['question_id']}: use registered pages for supported, or a note and no pages for missing evidence/context.")
    return status, tuple(sorted(ids)), note

def validate(rows):
    if len(rows)!=len(QUESTIONS) or {int(r["question_id"]) for r in rows} != set(QUESTIONS):
        raise ValueError("Keep one worksheet row per published question (1–35).")
    return {int(r["question_id"]): decision(r) for r in rows}

def read(path=SHEET):
    with path.open(encoding="utf-8-sig",newline="") as f:
        rows=list(csv.DictReader(f))
    validate(rows)
    return rows

def save(q, status, ids, note, previous, path=SHEET):
    with LOCK:
        rows=read(path)
        row=next(r for r in rows if int(r["question_id"])==q)
        if decision(row) != previous:
            raise ValueError("This row changed since you opened it. Reload the row before saving.")
        candidate=row | {"review_status":status,"acceptable_sources":"|".join(sorted(ids or [])),"note":(note or "").strip()}
        decision(candidate)
        row.update(candidate)
        tmp=None
        try:
            with NamedTemporaryFile("w",dir=path.parent,encoding="utf-8",newline="",delete=False) as f:
                tmp=Path(f.name)
                writer=csv.DictWriter(f,fieldnames=FIELDS)
                writer.writeheader()
                writer.writerows(rows)
            tmp.replace(path)
        finally:
            if tmp:
                tmp.unlink(missing_ok=True)
