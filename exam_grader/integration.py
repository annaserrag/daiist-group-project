"""Join callables, not file formats. The template's logger receives a safe projection only."""
from pathlib import Path
import hashlib
from time import perf_counter
from pydantic import ValidationError
from .grading import grade_exam
from .handoff import VisionUsage

def file_sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()

def connect(raw_input, *, read_exam, vision_version, index, agent=None,
            approved_hashes=(), policy=None, reader_charges_possible=True):
    """A server-side hash allowlist permits API use only for vetted synthetic/redacted PDFs.
    read_exam(path) must read the same server-owned, immutable upload during this request.
    Never use a UI checkbox or a client-provided hash to authorize remote processing.
    """
    if not isinstance(vision_version,str) or not vision_version.strip():
        raise ValueError("Set the vision version on the server.")
    approved = False
    if agent is not None and approved_hashes:
        approved = file_sha256(raw_input) in approved_hashes
    started = perf_counter()
    reader_failed = False
    try:
        payload = read_exam(raw_input)
    except Exception:
        reader_failed = True
        payload = {"schema_version":"vision.exam.v1","exam_status":"unreadable","answers":[]}
    vision_seconds = perf_counter() - started
    result = grade_exam(payload,index=index,agent=agent,policy=policy,allow_remote=approved)
    record = result["record"]
    grader = record["outputs"]["component_b"]
    vision_usage = VisionUsage(tokens=0, cost_eur=0.0)
    if reader_charges_possible:
        reported = payload.get("usage") if isinstance(payload, dict) else None
        try:
            vision_usage = VisionUsage.model_validate(reported or {})
        except ValidationError:
            vision_usage = VisionUsage()
    record["outputs"]["reader_usage"] = vision_usage.model_dump()
    grader["tokens"] = record["tokens"]
    grader["cost_eur"] = record["cost_eur"] if grader["cost_known"] else None
    record["outputs"]["total_usage"] = {
        "usage_complete": vision_usage.tokens is not None and grader["usage_complete"],
        "cost_known": vision_usage.cost_eur is not None and grader["cost_known"],
        "uncached_tokens": grader["uncached_tokens"] + vision_usage.tokens if vision_usage.tokens is not None and grader["usage_complete"] else None,
        "uncached_cost_eur": grader["uncached_cost_eur"] + vision_usage.cost_eur
            if grader["uncached_cost_eur"] is not None and vision_usage.cost_eur is not None else None,
    }
    # The course logger requires numeric fields. Incomplete totals carry only observed usage.
    record["tokens"] += vision_usage.tokens or 0
    record["cost_eur"] += vision_usage.cost_eur or 0.0
    result["teacher"]["reader_failed"] = reader_failed
    result["record"]["outputs"]["component_b"]["reader_failed"] = reader_failed
    result["record"]["features"]["vision_latency_s"] = round(vision_seconds, 4)
    result["record"]["versions"]["component_a"] = vision_version
    if isinstance(payload,dict):
        features = payload.get("features")
        if isinstance(features,dict):
            for k in ("ink_density","line_count"):
                v = features.get(k)
                if type(v) in (float,int) and (0 <= v <= 1 if k == "ink_density" else type(v) is int and 0 <= v <= 100000):
                    result["record"]["features"][k] = v
    return result
