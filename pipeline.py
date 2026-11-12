"""Course pipeline entry points for JSON handoffs and the team's PDF reader."""
import json
import os
from functools import lru_cache
from importlib import import_module
from importlib.util import find_spec
from pathlib import Path
from time import perf_counter

import env
from exam_grader.catalog import DATA
from exam_grader.grading import Policy
from exam_grader.integration import connect, file_sha256
from exam_grader.retrieval import Index

env.load_env()
EXAMPLE = DATA / "vision-example.json"

def final_after_lookup():
    strategy = os.getenv("GRADE_PASSAGE_STRATEGY", "finalize")
    if strategy not in {"finalize", "original"}:
        raise ValueError("GRADE_PASSAGE_STRATEGY must be finalize or original.")
    return strategy == "finalize"

@lru_cache
def resources():
    design = os.getenv("GRADE_INDEX_DESIGN", "section")
    has_vectors = (DATA / f"vectors-{design}.json").is_file()
    mode = os.getenv("GRADE_RETRIEVAL_MODE", "auto")
    if mode not in {"auto", "lexical", "semantic"}:
        raise ValueError("GRADE_RETRIEVAL_MODE must be auto, lexical or semantic.")
    ranking = os.getenv("GRADE_EVIDENCE_RANKING", "multipart")
    if ranking not in {"multipart", "original"}:
        raise ValueError("GRADE_EVIDENCE_RANKING must be multipart or original.")
    index = Index(design=design, semantic=mode == "semantic" or (mode == "auto" and has_vectors), multipart=ranking == "multipart")
    agent = None
    if os.getenv("GEMINI_API_KEY"):
        from exam_grader.gemini import Gemini
        rates = [os.getenv(name) for name in ("GRADE_INPUT_EUR_PER_M", "GRADE_OUTPUT_EUR_PER_M")]
        agent = Gemini(input_eur_per_million=rates[0], output_eur_per_million=rates[1], finalize_lookup=final_after_lookup())
    approved = {file_sha256(EXAMPLE)}
    approved.update(os.getenv("GRADE_APPROVED_HASHES", "").lower().split())
    return index, agent, frozenset(approved)

def read_export(path):
    if path.stat().st_size > 512_000:
        raise ValueError("OCR export is too large.")
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)

def run_views(raw_input, *, policy=None, evaluation_cache=None, course_index=None):
    path = Path(raw_input)
    index, agent, approved = resources()
    if course_index is not None:
        index = course_index
    if agent is not None and evaluation_cache is not None:
        agent = agent.for_evaluation(evaluation_cache)
    elif evaluation_cache is not None and evaluation_cache.read_only:
        from exam_grader.gemini import Gemini
        agent = Gemini(frozen_cache=evaluation_cache, finalize_lookup=final_after_lookup())
    if path.suffix.lower() == ".json":
        reader, version, input_type = read_export, "ocr-json-replay-v1", "ocr_export"
    elif path.suffix.lower() == ".pdf":
        module_name = os.getenv("GRADE_VISION_MODULE", "vision")
        if find_spec(module_name) is None:
            reader = lambda _: {"schema_version":"vision.exam.v1", "exam_status":"unreadable", "answers":[]}
            version, input_type = "vision-unavailable", "pdf_awaiting_reader"
        else:
            module = import_module(module_name)
            reader, version, input_type = module.read_exam, module.VERSION, "pdf"
    else:
        raise ValueError("Choose an OCR JSON export or a PDF exam.")
    started = perf_counter()
    if policy is None and os.getenv("GRADE_POLICY_FILE"):
        policy = Policy(**json.loads(Path(os.environ["GRADE_POLICY_FILE"]).read_text(encoding="utf-8-sig")))
    result = connect(path, read_exam=reader, vision_version=version, index=index,
                     agent=agent, approved_hashes=approved, policy=policy, reader_charges_possible=input_type == "pdf")
    duration = perf_counter() - started
    result["record"]["outputs"]["component_b"]["input_type"] = input_type
    result["record"]["versions"]["rubric"] = "mendeley-sf3kvjwknt-v1"
    result["teacher"]["input_type"] = input_type
    result["record"]["latency_s"] = round(duration, 4)
    return result

def run(raw_input):
    return run_views(raw_input)["record"]
