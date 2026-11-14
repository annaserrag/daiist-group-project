"""Exercise the production PDF path on new, typeset two-column layouts with predeclared answers."""
import argparse
import json
import tempfile
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from exam_grader.catalog import KEY
from exam_grader.grading import grade_exam
from exam_grader.vision_finetune import VERSION, read_exam


CASES = {
    "nonconsecutive": [[(3, "C"), (8, "A"), (5, "D")], [(12, "C"), (16, "D"), (19, "B")]],
    "duplicate": [[(4, "C"), (7, "A"), (7, "B")], [(10, "B"), (15, "C"), (13, "B")]],
    "edited_choice": [[(6, "A"), (3, "B D"), (2, "C")], [(13, "D"), (18, "A"), (16, "C")]],
}


def render(columns, path):
    image = Image.new("L", (2000, 2600), 255)
    pen = ImageDraw.Draw(image)
    pen.text((105, 115), "Data Science Examination | Answer sheet", font=ImageFont.load_default(size=48), fill=0)
    font = ImageFont.load_default(size=58)
    for x, rows in zip([240, 1060], columns, strict=True):
        for y, (number, answer) in enumerate(rows):
            top = 290 + 130*y
            pen.text((x, top), str(number), font=font, fill=0)
            pen.line((x+94, top+1, x+77, top+68), fill=0, width=4)
            pen.text((x+145, top), answer, font=font, fill=0)
            if " " in answer:
                pen.line((x+144, top+54, x+178, top+14), fill=0, width=4)
    image.convert("RGB").save(path, "PDF", resolution=160)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    evidence = {"reader": VERSION, "source": "New typeset layout challenges. No student handwriting.", "cases": {}}
    with tempfile.TemporaryDirectory() as temp:
        for name, columns in CASES.items():
            pdf = Path(temp)/f"{name}.pdf"
            render(columns, pdf)
            exported = read_exam(pdf)
            answers = grade_exam(exported, index=None)["teacher"]["answers"]
            expected = {}
            for column in columns:
                for q, choice in column:
                    expected.setdefault(q, []).append(choice)
            outcomes = []
            for item in answers:
                q, mark = item["question_id"], item["mark"]
                if q not in expected and mark is None:
                    continue
                letters = expected.get(q, [])
                ambiguous = len(letters) != 1 or len(letters[0]) != 1
                safe = mark is None or (not ambiguous and item["readings"] == letters and mark == int(letters[0] == KEY[q]))
                outcomes.append({"id": q, "readings": item["readings"], "mark": mark, "reason": item["reason"], "safe": safe})
            evidence["cases"][name] = outcomes
    args.output.write_text(json.dumps(evidence, indent=2)+"\n", encoding="utf-8")
    if not all(item["safe"] for case in evidence["cases"].values() for item in case):
        raise SystemExit("An incorrect or ambiguous choice received an automatic mark.")
    for name, min_marks in {"nonconsecutive": 6, "duplicate": 4, "edited_choice": 5}.items():
        if sum(item["mark"] is not None for item in evidence["cases"][name]) < min_marks:
            raise SystemExit(f"Clean choices stayed in review on the {name} layout.")


if __name__ == "__main__":
    main()
