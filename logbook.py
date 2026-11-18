"""
Request and feedback logs. Don't change the fields in REQUIRED_FIELDS: check.py
and the grading rely on them. You may add your own.

Where the logs go:

- Online, on Render, DATABASE_URL points to your free Postgres database (render.yaml
  sets it). Each request is a row of the table `requests`, each 👍/👎 a row of
  `feedback` and each teacher sign-off a row of `signoffs`, with the whole record
  as JSON, so fields you add are kept too. The database survives restarts, sleeps and redeploys; the app's disk does not.
- Everywhere else (your laptop, check.py, grading), or whenever LOG_DIR is set, logs
  are JSON-lines files in logs/, one per process.

`uv run python main.py export-logs` copies the database's logs into logs/, so the
app on your laptop shows them too. See SETUP.md, "Logs".
"""

import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LOG_DIR = Path(os.environ.get("LOG_DIR", ROOT / "logs"))

REQUIRED_FIELDS = {
    "id": str,
    "timestamp": str,
    "input_summary": str,
    "features": dict,
    "outputs": dict,
    "versions": dict,
    "review": bool,
    "latency_s": (int, float),
    "tokens": int,
    "cost_eur": (int, float),
}

TABLES = ("requests", "feedback", "signoffs")
_run_id = uuid.uuid4().hex[:8]
_lock = threading.Lock()


# --- Where the logs go ---------------------------------------------------------

def database_url() -> str:
    """The database the logs go to, or "" for files. LOG_DIR set means files."""
    if os.environ.get("LOG_DIR"):
        return ""
    return os.environ.get("DATABASE_URL", "").strip()


def _connect(url: str | None = None):
    import psycopg  # only needed when there is a database

    return psycopg.connect(url or database_url(), autocommit=True, connect_timeout=10)


def _create_tables(url: str | None = None) -> None:
    with _connect(url) as conn:
        for table in TABLES:
            conn.execute(
                f"CREATE TABLE IF NOT EXISTS {table} ("
                "  row_id bigserial PRIMARY KEY,"
                "  id text NOT NULL,"
                "  ts timestamptz NOT NULL,"
                "  record jsonb NOT NULL)"
            )


def start() -> str:
    """Get the logs ready. Returns a sentence saying where they go."""
    if url := database_url():
        try:
            _create_tables(url)
        except Exception as error:  # noqa: BLE001 - say what is wrong instead of a long traceback
            raise SystemExit(f"Cannot reach the logs database in DATABASE_URL: {type(error).__name__}: {error}") from error
        host = url.split("@")[-1].split("/")[0].split(":")[0]  # never show the password
        return f"logs go to the Postgres database at {host}"
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    return f"logs go to {LOG_DIR}"


# --- Writing -------------------------------------------------------------------

def _write(table: str, record: dict) -> None:
    if database_url():
        from psycopg.types.json import Jsonb

        with _connect() as conn:
            conn.execute(
                f"INSERT INTO {table} (id, ts, record) VALUES (%s, %s, %s)",
                (record["id"], record["timestamp"], Jsonb(record)),
            )
        return
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _lock, (LOG_DIR / f"{table}-{_run_id}.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def validate(record: dict) -> list[str]:
    """What is wrong with a request record, as a list of problems (empty if fine)."""
    problems = []
    for field, kind in REQUIRED_FIELDS.items():
        if field not in record:
            problems.append(f"missing field `{field}`")
        elif not isinstance(record[field], kind) or (kind is int and isinstance(record[field], bool)):
            problems.append(f"`{field}` should be {kind}, not {type(record[field]).__name__}")
    for name, value in record.get("features", {}).items():
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            problems.append(f"feature `{name}` should be a number")
    return problems


def log_request(record: dict) -> dict:
    """Add id and timestamp, check the fields, save. Returns the record as logged."""
    record = {"id": uuid.uuid4().hex, "timestamp": datetime.now(timezone.utc).isoformat()} | record
    problems = validate(record)
    if problems:
        raise ValueError("Log record does not match logbook.REQUIRED_FIELDS: " + "; ".join(problems))
    _write("requests", record)
    return record


def log_feedback(request_id: str, value: int) -> None:
    """value: +1 for 👍, -1 for 👎. The latest feedback for a request wins."""
    _write("feedback", {
        "id": request_id, "timestamp": datetime.now(timezone.utc).isoformat(), "feedback": int(value),
    })


def log_signoff(request_id: str, review_arm: str | None, answers: int, referred: int, changed: int,
                reward: float | None) -> None:
    """The teacher signed off a paper: how many answers the review rule sent to review, how many proposals the
    teacher changed, and the bandit's reward (exam_grader/bandit.py). Counts only, no marks or text."""
    _write("signoffs", {
        "id": request_id, "timestamp": datetime.now(timezone.utc).isoformat(), "review_arm": review_arm,
        "answers": int(answers), "referred": int(referred), "changed": int(changed), "reward": reward,
    })


# --- Reading -------------------------------------------------------------------

def _read(table: str) -> list[dict]:
    if database_url():
        with _connect() as conn:
            return [row[0] for row in conn.execute(f"SELECT record FROM {table} ORDER BY ts, row_id")]
    rows = []
    for path in sorted(LOG_DIR.rglob(f"{table}-*.jsonl")) if LOG_DIR.is_dir() else []:
        for line in path.read_text(encoding="utf-8").splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load() -> pd.DataFrame:
    """All requests, one row each, with their features as `f_<name>` columns and their feedback."""
    requests = pd.DataFrame(_read("requests"))
    if requests.empty:
        return requests
    requests = requests.drop_duplicates("id")
    requests["timestamp"] = pd.to_datetime(requests["timestamp"], utc=True, format="ISO8601")
    features = pd.json_normalize(requests["features"].tolist()).add_prefix("f_")
    requests = pd.concat([requests.reset_index(drop=True), features], axis=1)
    feedback = pd.DataFrame(_read("feedback"))
    if feedback.empty:
        requests["feedback"] = pd.NA
    else:
        latest = feedback.sort_values("timestamp").groupby("id")["feedback"].last()
        requests["feedback"] = requests["id"].map(latest)
    return requests.sort_values("timestamp").reset_index(drop=True)


def load_signoffs() -> pd.DataFrame:
    """Every signed-off paper once (a later sign-off of the same paper replaces the earlier one)."""
    rows = pd.DataFrame(_read("signoffs"))
    if rows.empty:
        return rows
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, format="ISO8601")
    return rows.sort_values("timestamp").drop_duplicates("id", keep="last").reset_index(drop=True)


def export(url: str) -> dict[str, int]:
    """Copy every row of the database at `url` into logs/<table>-export.jsonl. Returns the counts."""
    counts = {}
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with _connect(url) as conn:
        for table in TABLES:
            exists = conn.execute("SELECT to_regclass(%s)", (table,)).fetchone()[0]
            rows = [r[0] for r in conn.execute(f"SELECT record FROM {table} ORDER BY ts, row_id")] if exists else []
            with (LOG_DIR / f"{table}-export.jsonl").open("w", encoding="utf-8") as f:
                f.writelines(json.dumps(row, ensure_ascii=False) + "\n" for row in rows)
            counts[table] = len(rows)
    return counts
