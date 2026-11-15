"""
Warm start for the review-threshold bandit (exam_grader/bandit.py): the simulated part of its learning.

Each in-scope PDF of the team's development packet (eval/cases.csv, with the checked teacher marks in
eval/case_references.json) is graded once with the reading gate switched off (min_read = 0), so every answer the
grader can mark gets a proposal. Because the bandit's thresholds are nested, the cost of every threshold can then be
counted offline from the same run: an answer below the threshold would have gone to review, an answer above it is
compared with the teacher's mark. Only the questions with a teacher reference count.

This packet is a development set (it reuses papers the team has looked at), which is fine for a prior: the bandit
then learns from real sign-offs, and the warm start counts as at most bandit.PRIOR_PAPERS papers per arm. The clean
held-out exams (eval/heldout) are never used here.

    uv run python eval/bandit_warmup.py      (needs GEMINI_API_KEY)

Privacy: as for the held-out exams, every faint pixel of each PDF is whitened first (the student's name shows through
faintly from the back of the sheet, and the reader once read a page header into an answer), and only those cleaned
copies, kept outside the repo, are approved for Gemini, for this process only.

Writes monitoring/bandit_prior.json (what the app loads) and eval/bandit_warmup.json (every paper, every arm).
"""

import csv
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exam_grader import bandit  # noqa: E402
import env  # noqa: E402

env.load_env()  # GEMINI_API_KEY from .env

CLEAN = Path.home() / ".cache" / "daiist-vision" / "bandit_warmup"


def cleaned(pdf: Path) -> Path:
    """A copy of the PDF with every faint pixel (grey level above 165) whitened, outside the repo."""
    import numpy as np
    from PIL import Image
    from vision import reader
    CLEAN.mkdir(parents=True, exist_ok=True)
    pages = [Image.fromarray(np.where(p > 165, 255, p).astype(np.uint8)) for p in reader.load_pages(pdf)]
    out = CLEAN / pdf.name
    pages[0].save(out, save_all=True, append_images=pages[1:], resolution=200)
    return out


def main() -> None:
    if not os.getenv("GEMINI_API_KEY"):
        raise SystemExit("Set GEMINI_API_KEY: without it every written answer is referred and the prior is skewed.")
    references = json.loads((ROOT / "eval" / "case_references.json").read_text(encoding="utf-8"))
    cases = [c for c in csv.DictReader((ROOT / "eval" / "cases.csv").open(encoding="utf-8"))
             if references.get(c["id"], {}).get("exam_status") == "in_scope"]
    inputs = {case["id"]: cleaned(ROOT / case["input"]) for case in cases}
    from exam_grader.integration import file_sha256
    os.environ["GRADE_APPROVED_HASHES"] = " ".join(file_sha256(p) for p in inputs.values())
    import pipeline
    from exam_grader.grading import Policy
    papers = []
    for case in cases:
        result = pipeline.run_views(inputs[case["id"]], policy=Policy(min_read=0.0, written_review_at_or_below=1))
        marks = references[case["id"]]["reference_marks"]
        answers = []
        for item in result["teacher"]["answers"]:
            q = str(item["question_id"])
            if q not in marks:
                continue  # a slot the packet does not cover
            final = marks[q] if isinstance(marks[q], (int, float)) else None
            answers.append({"q": int(q), "reading_confidence": item.get("reading_confidence"),
                            "proposed": item["mark"], "final": final})
        if not answers:
            continue
        row = {"case": case["id"], "answers": len(answers)}
        for arm in bandit.ARMS:
            referred, changed = bandit.counterfactual(answers, arm)
            row[str(arm)] = {"referred": referred, "changed": changed,
                             "reward": bandit.paper_reward(len(answers), referred, changed)}
        papers.append(row)
        print(case["id"], {arm: round(row[str(arm)]["reward"], 3) for arm in bandit.ARMS}, flush=True)
    arms = {str(arm): {"mean": sum(p[str(arm)]["reward"] for p in papers) / len(papers), "papers": len(papers)}
            for arm in bandit.ARMS}
    bandit.PRIOR_FILE.write_text(json.dumps(
        {"source": "eval/bandit_warmup.py on the development packet (simulated, not real sign-offs)",
         "review_cost": bandit.REVIEW_COST, "wrong_cost": bandit.WRONG_COST, "arms": arms}, indent=2) + "\n")
    (ROOT / "eval" / "bandit_warmup.json").write_text(json.dumps({"papers": papers, "arms": arms}, indent=2) + "\n")
    print(json.dumps(arms, indent=2))


if __name__ == "__main__":
    main()
