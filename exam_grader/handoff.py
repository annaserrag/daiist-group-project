"""Check a vision export before connecting it to the grader. No model or network needed."""
import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

Score = Annotated[float, Field(ge=0, le=1, allow_inf_nan=False)]
Question = Annotated[int, Field(ge=1, le=35)]


class VisionAnswer(BaseModel):
    model_config = ConfigDict(strict=True, json_schema_extra={
        "if":{"properties":{"state":{"const":"confirmed_blank"}}},
        "then":{"properties":{"text":{"const":""}}}})
    question_id: Question | None
    text: Annotated[str, Field(max_length=4000)]
    confidence: Score
    id_confidence: Score
    state: Literal["read", "unreadable", "confirmed_blank"]

    @model_validator(mode="after")
    def blank_has_no_text(self):
        if self.state == "confirmed_blank" and self.text != "":
            raise ValueError("A confirmed blank must have empty text.")
        return self


class VisionFeatures(BaseModel):
    model_config = ConfigDict(strict=True)
    ink_density: Score = Field(default=0)
    line_count: Annotated[int, Field(ge=0, le=100000)] = Field(default=0)


class VisionUsage(BaseModel):
    model_config = ConfigDict(strict=True)
    tokens: Annotated[int, Field(ge=0)] | None = None
    cost_eur: Annotated[float, Field(ge=0, allow_inf_nan=False)] | None = None


class VisionExam(BaseModel):
    model_config = ConfigDict(strict=True)
    schema_version: Literal["vision.exam.v1"]
    exam_id: Annotated[str, Field(min_length=1, max_length=100)] | None = None
    exam_status: Literal["in_scope", "unreadable", "out_of_scope"]
    answers: Annotated[list[VisionAnswer], Field(max_length=100)]
    features: VisionFeatures = Field(default_factory=VisionFeatures)
    usage: VisionUsage | None = None


def inspect_export(payload, *, written_exam=None):
    try:
        exam = VisionExam.model_validate(payload)
    except ValidationError as exc:
        return {"valid": False, "errors": [
            {"field":".".join(map(str,error["loc"])) or "exam","problem":error["msg"]}
            for error in exc.errors(include_url=False,include_input=False,include_context=False)]}
    known = Counter(r.question_id for r in exam.answers if r.question_id is not None)
    expected_id = written_exam.exam_id if written_exam else "mendeley-sf3kvjwknt-v1"
    expected_questions = {q.question_id for q in written_exam.questions} if written_exam else set(range(1, 36))
    if exam.exam_status == "in_scope" and (written_exam or exam.exam_id) and exam.exam_id != expected_id:
        return {"valid": False, "errors": [{"field": "exam_id", "problem": "Select the matching exam definition."}]}
    unexpected = sorted(known.keys() - expected_questions)
    if exam.exam_status == "in_scope" and unexpected:
        return {"valid": False, "unexpected_question_ids": unexpected,
                "errors": [{"field": "answers.question_id", "problem": "An answer is outside the selected exam."}]}
    return {"valid":True,"exam_status":exam.exam_status,"observations":len(exam.answers),
            "states":dict(Counter(r.state for r in exam.answers)),
            "unassigned":sum(r.question_id is None for r in exam.answers),
            "duplicate_question_ids":sorted(q for q,n in known.items() if n>1),
            "undetected_question_ids":sorted(expected_questions-known.keys())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("export",type=Path)
    parser.add_argument("--written-exam", type=Path, help="The selected custom exam definition")
    args = parser.parse_args()
    try:
        from .written_exam import WrittenExam
        definition = WrittenExam.load(args.written_exam) if args.written_exam else None
        with args.export.open(encoding="utf-8-sig") as stream:
            report = inspect_export(json.load(stream), written_exam=definition)
    except (OSError,UnicodeError,json.JSONDecodeError) as exc:
        parser.exit(2,f"Could not read a JSON export ({type(exc).__name__}).\n")
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report["valid"] else 1)
