"""
Request and feedback logs. Don't change the fields in REQUIRED_FIELDS: check.py
and the grading rely on them. You may add your own.

Each process writes its own JSON-lines files in logs/ (one line per request,
one per 👍/👎). If HF_LOG_REPO is set, `start()` first downloads the logs
already in that private Hugging Face dataset into logs_archive/, then pushes
logs/ to it every few minutes. That matters on a Space, whose disk is wiped
every time it restarts. Without HF_LOG_REPO, logs only stay in logs/.
"""

import contextlib
import json
import os
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
LOG_DIR = Path(os.environ.get("LOG_DIR", ROOT / "logs"))
ARCHIVE_DIR = ROOT / "logs_archive"
PUSH_EVERY_MINUTES = 5

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

_run_id = uuid.uuid4().hex[:8]
_scheduler = None
_lock = threading.Lock()


def _requests_file() -> Path:
    return LOG_DIR / f"requests-{_run_id}.jsonl"


def _feedback_file() -> Path:
    return LOG_DIR / f"feedback-{_run_id}.jsonl"


def _signoff_file() -> Path:
    return LOG_DIR / f"signoff-{_run_id}.jsonl"


def start() -> str:
    """Start pushing logs to HF_LOG_REPO, if it is set. Returns where logs go."""
    global _scheduler
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    repo = os.environ.get("HF_LOG_REPO", "").strip()
    if not repo or _scheduler is not None:
        return f"logs go to {LOG_DIR}" + (f" and {repo}" if _scheduler else "")

    from huggingface_hub import CommitScheduler, snapshot_download

    token = os.environ.get("HF_TOKEN") or None
    with contextlib.suppress(Exception):  # an empty or new dataset has nothing to download
        snapshot_download(repo, repo_type="dataset", local_dir=ARCHIVE_DIR, token=token, allow_patterns="logs/*")
    _scheduler = CommitScheduler(
        repo_id=repo, repo_type="dataset", folder_path=LOG_DIR, path_in_repo="logs",
        every=PUSH_EVERY_MINUTES, private=True, token=token,
    )
    return f"logs go to {LOG_DIR} and every {PUSH_EVERY_MINUTES} min to the dataset {repo}"


def _append(path: Path, record: dict) -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    with (_scheduler.lock if _scheduler else _lock), path.open("a", encoding="utf-8") as f:
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
    """Add id and timestamp, check the fields, append. Returns the record as logged."""
    record = {"id": uuid.uuid4().hex, "timestamp": datetime.now(timezone.utc).isoformat()} | record
    problems = validate(record)
    if problems:
        raise ValueError("Log record does not match logbook.REQUIRED_FIELDS: " + "; ".join(problems))
    _append(_requests_file(), record)
    return record


def log_feedback(request_id: str, value: int) -> None:
    """value: +1 for 👍, -1 for 👎. The latest feedback for a request wins."""
    _append(_feedback_file(), {
        "id": request_id, "timestamp": datetime.now(timezone.utc).isoformat(), "feedback": int(value),
    })


def log_signoff(request_id: str, review_arm: str | None, answers: int, referred: int, changed: int,
                reward: float | None) -> None:
    """The teacher signed off a paper: how many answers the review rule sent to review, how many proposals the
    teacher changed, and the bandit's reward (exam_grader/bandit.py). Counts only, no marks or text."""
    _append(_signoff_file(), {
        "id": request_id, "timestamp": datetime.now(timezone.utc).isoformat(), "review_arm": review_arm,
        "answers": int(answers), "referred": int(referred), "changed": int(changed), "reward": reward,
    })


def load_signoffs() -> pd.DataFrame:
    """Every signed-off paper once (a later sign-off of the same paper replaces the earlier one)."""
    rows = pd.DataFrame(_read("signoff-*.jsonl"))
    if rows.empty:
        return rows
    rows["timestamp"] = pd.to_datetime(rows["timestamp"], utc=True, format="ISO8601")
    return rows.sort_values("timestamp").drop_duplicates("id", keep="last").reset_index(drop=True)


def _read(pattern: str) -> list[dict]:
    rows = []
    for folder in (LOG_DIR, ARCHIVE_DIR):
        for path in sorted(folder.rglob(pattern)) if folder.is_dir() else []:
            for line in path.read_text(encoding="utf-8").splitlines():
                with contextlib.suppress(json.JSONDecodeError):
                    rows.append(json.loads(line))
    return rows


def load() -> pd.DataFrame:
    """All requests, one row each, with their features as `f_<name>` columns and their feedback."""
    requests = pd.DataFrame(_read("requests-*.jsonl"))
    if requests.empty:
        return requests
    requests = requests.drop_duplicates("id")
    requests["timestamp"] = pd.to_datetime(requests["timestamp"], utc=True, format="ISO8601")
    features = pd.json_normalize(requests["features"].tolist()).add_prefix("f_")
    requests = pd.concat([requests.reset_index(drop=True), features], axis=1)
    feedback = pd.DataFrame(_read("feedback-*.jsonl"))
    if feedback.empty:
        requests["feedback"] = pd.NA
    else:
        latest = feedback.sort_values("timestamp").groupby("id")["feedback"].last()
        requests["feedback"] = requests["id"].map(latest)
    return requests.sort_values("timestamp").reset_index(drop=True)
