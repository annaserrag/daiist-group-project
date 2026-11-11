"""Private, opt-in cache for repeat evaluations. Each successful response is its own transaction."""
import hashlib
import json
import sqlite3
from contextlib import closing
from pathlib import Path


class MissingFrozenResponse(Exception):
    """A replay has no saved response for this exact model request."""


class EvaluationCache:
    def __init__(self, path, *, read_only=False):
        self.path = Path(path).resolve()
        self.read_only = read_only
        project = Path(__file__).resolve().parents[1]
        if self.path.is_relative_to(project):
            raise ValueError("Put the evaluation cache outside the deployable repository.")
        if read_only:
            if not self.path.is_file():
                raise FileNotFoundError("The frozen response database is missing.")
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with closing(sqlite3.connect(self.path, timeout=120)) as db, db:
            db.execute("CREATE TABLE IF NOT EXISTS grade_responses "
                       "(fingerprint TEXT PRIMARY KEY, response TEXT NOT NULL, "
                       "input_tokens INTEGER NOT NULL, output_tokens INTEGER NOT NULL)")

    def get_or_generate(self, signature, request, generate):
        content = json.dumps([signature, request], ensure_ascii=False, sort_keys=True,
                             separators=(",", ":")).encode()
        fingerprint = hashlib.sha256(content).hexdigest()
        if self.read_only:
            with closing(sqlite3.connect(self.path.as_uri() + "?mode=ro", uri=True)) as db:
                saved = db.execute("SELECT response, input_tokens, output_tokens FROM grade_responses "
                                   "WHERE fingerprint = ?", (fingerprint,)).fetchone()
            if saved is None:
                raise MissingFrozenResponse
            return json.loads(saved[0]), (saved[1], saved[2]), True
        with closing(sqlite3.connect(self.path, timeout=120)) as db, db:
            # Serial evaluation jobs sharing this file cannot both pay for the same miss.
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT response, input_tokens, output_tokens FROM grade_responses "
                             "WHERE fingerprint = ?", (fingerprint,)).fetchone()
            if row:
                return json.loads(row[0]), (row[1], row[2]), True
            answer, counts = generate()
            if counts is not None:
                db.execute("INSERT INTO grade_responses VALUES (?, ?, ?, ?)",
                           (fingerprint, json.dumps(answer), *counts))
            return answer, counts, False
