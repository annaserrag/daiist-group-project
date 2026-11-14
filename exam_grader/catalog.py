"""Read the two public instructor files only. Never load teacher marks or scans."""
import csv
import re
from pathlib import Path

DATA = Path(__file__).with_name("data")
KEY_URL = "https://doi.org/10.17632/sf3kvjwknt.1"
text = (DATA / "Question.txt").read_text(encoding="utf-8-sig")
QUESTIONS = {int(i): q.strip() for i, q in re.findall(r"(?m)^(\d+)\. (.+)$", text)}
with (DATA / "answerkey.txt").open(encoding="utf-8-sig", newline="") as stream:
    KEY = {int(row["Question_Number"]): row["Correct_Answer"].strip() for row in csv.DictReader(stream)}
if set(KEY) != set(QUESTIONS) or set(KEY) != set(range(1, 36)):
    raise RuntimeError("Incomplete question catalogue or published key.")
OPTION_TEXTS = {int(i): dict(re.findall(r"(?m)^([A-D])\. (.+)$", block))
                for i, block in re.findall(r"(?ms)^(\d+)\. (.*?)(?=^\d+\. |\Z)", text) if int(i) <= 20}
OPTIONS = {q: set(choices) for q, choices in OPTION_TEXTS.items()}
if any(KEY[i] not in letters for i, letters in OPTIONS.items()):
    raise RuntimeError("Invalid published MCQ key.")
