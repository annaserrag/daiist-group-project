"""Teacher-defined, two-point written exams using the vision.exam.v1 question ID range."""
import hashlib
import json
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=2000)]
Identifier = Annotated[str, StringConstraints(pattern=r"^[a-z][a-z0-9_]{0,39}$")]


class Criterion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    id: Identifier
    points: float
    description: Text


class WrittenQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    question_id: Annotated[int, Field(ge=1, le=35)]
    question: Text
    reference_answer: Text
    criteria: Annotated[list[Criterion], Field(min_length=1, max_length=4)]
    requires_figure: bool = False

    @model_validator(mode="after")
    def validate_criteria(self):
        if len({c.id for c in self.criteria}) != len(self.criteria):
            raise ValueError("Use a unique ID for each criterion within a question.")
        if any(c.points not in (.5, 1, 1.5, 2) for c in self.criteria) or sum(c.points for c in self.criteria) != 2:
            raise ValueError("Criterion weights must total two points in half-point increments.")
        return self

    def guide(self):
        return {c.id: (c.points, c.description) for c in self.criteria}


class WrittenExam(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    title: Text
    questions: Annotated[list[WrittenQuestion], Field(min_length=1, max_length=15)]

    @model_validator(mode="after")
    def unique_questions(self):
        if len({q.question_id for q in self.questions}) != len(self.questions):
            raise ValueError("Question numbers must be unique within this exam.")
        return self

    @property
    def exam_id(self):
        content = json.dumps(self.model_dump(), sort_keys=True, separators=(",", ":")).encode()
        return "written-" + hashlib.sha256(content).hexdigest()

    @classmethod
    def load(cls, path):
        path = Path(path)
        if path.stat().st_size > 150_000:
            raise ValueError("Choose a written exam definition of up to 150 KB.")
        return cls.model_validate_json(path.read_text(encoding="utf-8-sig"))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("definition")
    print(WrittenExam.load(parser.parse_args().definition).exam_id)
