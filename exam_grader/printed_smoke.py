"""Optional integration check on a generated, raster-only printed sheet with three fixed rows."""
import argparse
import json
import os
import re
from importlib import metadata
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from .catalog import DATA
from .handoff import inspect_export
from .integration import file_sha256

VERSION = "printed-fixture-rapidocr-1.4.4"
WIDTH, HEIGHT = 1400, 300
ROW_TOPS = (65, 135, 205)
FIXTURE_HASH = None


def make_fixture(path):
    from PIL import Image, ImageDraw, ImageFont
    handoff = json.loads((DATA / "vision-example.json").read_bytes())
    image = Image.new("RGB", (WIDTH, HEIGHT), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=30)
    draw.text((20, 8), "Printed integration sample", font=font, fill="black")
    rows = sorted((row for row in handoff["answers"] if row["question_id"] in (1, 17, 21)),
                  key=lambda row: row["question_id"])
    for top, row in zip(ROW_TOPS, rows, strict=True):
        draw.text((20, top), str(row["question_id"]), font=font, fill="black")
        draw.text((150, top), row["text"], font=font, fill="black")
    image.save(path, "PDF", resolution=72.0)
    return rows


def read_exam(path):
    import numpy as np
    import pypdfium2 as pdfium
    from PIL import ImageOps
    from rapidocr_onnxruntime import RapidOCR
    if FIXTURE_HASH is None or file_sha256(path) != FIXTURE_HASH:
        raise ValueError("This engineering reader accepts only its generated printed fixture.")
    with pdfium.PdfDocument(path) as pdf:
        page = pdf[0]
        raster = page.render(scale=1).to_pil().convert("RGB")
        page.close()
    engine = RapidOCR(intra_op_num_threads=1, inter_op_num_threads=1)
    def recognize(box):
        field = raster.crop(box)
        bounds = ImageOps.invert(field.convert("L")).point(lambda p: 255 if p > 100 else 0).getbbox()
        if not bounds:
            return "", 0.0
        cropped = ImageOps.expand(field.crop(bounds), border=4, fill="white")
        results, _ = engine(np.asarray(cropped), use_det=False, use_cls=False)
        return (str(results[0][0]), float(results[0][1])) if results else ("", 0.0)
    answers = []
    for y in ROW_TOPS:
        label, id_score = recognize((0, y, 120, y + 55))
        text, reading_score = recognize((120, y, WIDTH, y + 55))
        q = int(label) if re.fullmatch(r"[0-9]{1,2}", label) else None
        answers.append({"question_id": q if q in range(1, 36) else None, "text":text,
                        "confidence":reading_score, "id_confidence":id_score, "state":"read" if text else "unreadable"})
    handoff = {"schema_version":"vision.exam.v1", "exam_status":"in_scope", "answers":answers,
               "usage":{"tokens":0,"cost_eur":0.0}}
    if not inspect_export(handoff)["valid"]:
        raise ValueError("The printed reader produced an invalid handoff.")
    return handoff


def run(live):
    import pipeline
    from . import printed_smoke
    with TemporaryDirectory(prefix="printed-handoff-") as tmp:
        pdf = Path(tmp) / "fixture.pdf"
        expected = make_fixture(pdf)
        printed_smoke.FIXTURE_HASH = file_sha256(pdf)
        try:
            config = {"GRADE_VISION_MODULE": printed_smoke.__name__,
                      "GRADE_APPROVED_HASHES": printed_smoke.FIXTURE_HASH if live else ""}
            with patch.dict(os.environ, config):
                pipeline.resources.cache_clear()
                if live and not os.getenv("GEMINI_API_KEY"):
                    raise ValueError("Set GEMINI_API_KEY to request live written grading.")
                result = pipeline.run_views(pdf)
        finally:
            pipeline.resources.cache_clear()
            printed_smoke.FIXTURE_HASH = None
    out = result["teacher"]
    record = result["record"]
    answers = {r["question_id"]: r for r in out["answers"]}
    matched = [r["question_id"] for r in expected if answers[r["question_id"]]["readings"] == [r["text"]]]
    routes = {str(q): {"mark":answers[q]["mark"], "reason":answers[q]["reason"]} for q in (1,17,21)}
    valid = not out["reader_failed"] and out["exam_status"] == "in_scope"
    passed = (valid and len(matched) == len(expected) and routes["1"]["mark"] == 1
              and routes["17"]["reason"] == "invalid_choice" and (routes["21"]["mark"] == 2 if live else
              routes["21"]["reason"] == "provider_disabled"))
    return {"scope":"Generated printed, raster-only PDF in a fixed layout. OCR reads each ID and answer "
                    "independently. This checks the PDF connector and grader. No handwritten student exams.",
            "reader":VERSION, "renderer":metadata.version("pypdfium2"), "live_written_grading":live,
            "grading_model":record["versions"]["component_b"], "index":record["versions"]["retrieval"],
            "q21_cited_passage":(answers[21]["citation"] or {}).get("id"),
            "pdf_path_used":out["input_type"], "contract_valid":valid,
            "exact_readings":matched, "expected_readings":len(expected), "routes":routes,
            "seconds":record["latency_s"], "ocr_seconds":record["features"]["vision_latency_s"],
            "grader_seconds":record["outputs"]["component_b"]["latency_s"], "tokens":record["tokens"],
            "cost_eur":record["cost_eur"] if record["outputs"]["total_usage"]["cost_known"] else None,
            "passed":bool(passed)}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live", action="store_true", help="Allow Gemini grading only for the generated fixture.")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run(args.live)
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text, encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["passed"] else 1)
