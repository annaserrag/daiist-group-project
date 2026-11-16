"""
Warm start for the review-rule bandit (exam_grader/bandit.py): the simulated part of its learning.

Why a simulator. The course brief allows warming the policy up with a simulator as long as the report says which part
is simulated. The team's rule is that every Mendeley exam is for evaluation only, and the development packet is built
from Mendeley papers, so the warm start cannot replay real exams. Instead it simulates papers from the assumptions
below, which are round guesses written down by the team, not fitted to any exam. They only set where each arm starts:
the warm start counts as at most bandit.PRIOR_PAPERS papers per arm, so a handful of real sign-offs overrule it.

The assumptions, per simulated paper of the built-in exam (20 multiple-choice and 15 written questions):
- multiple choice: MCQ_REFERRED of answers go to review for reading reasons, the rest are proposed and correct
  (the same under every arm, so they only set the scale of the cost);
- written: WRITTEN_REFERRED go to review under every arm (the grader is unsure, or found no evidence);
- the teacher's written mark is equally likely to be 0, 0.5, 1, 1.5 or 2;
- Gemini's proposal equals the teacher's mark with probability EXACT, is half a point off with probability HALF_OFF,
  and a whole point off otherwise (kept within 0-2).
With these numbers a let-through written proposal is wrong 40% of the time, which at WRONG_COST = 5 costs 2 reviews
on average, so the prior leans towards the strict arm; real sign-offs decide whether that holds.

    uv run --no-dev python eval/bandit_simulator.py

Writes monitoring/bandit_prior.json (what the app loads) and eval/bandit_simulator.json (the cost of each arm).
"""

import json
import random
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from exam_grader import bandit  # noqa: E402

PAPERS, SEED = 200, 0
MCQ, WRITTEN = 20, 15
MCQ_REFERRED, WRITTEN_REFERRED = 0.05, 0.30
EXACT, HALF_OFF = 0.60, 0.30
MARKS = (0, 0.5, 1, 1.5, 2)


def simulated_paper(rng: random.Random) -> list[dict]:
    """One paper's answers in the shape bandit.counterfactual reads."""
    answers = [{"written": False, "proposed": None if rng.random() < MCQ_REFERRED else 1, "final": 1}
               for _ in range(MCQ)]
    for _ in range(WRITTEN):
        final = rng.choice(MARKS)
        if rng.random() < WRITTEN_REFERRED:
            answers.append({"written": True, "proposed": None, "final": final})
            continue
        draw = rng.random()
        error = 0 if draw < EXACT else 0.5 if draw < EXACT + HALF_OFF else 1
        proposed = min(2, max(0, final + rng.choice((-1, 1)) * error))
        answers.append({"written": True, "proposed": proposed, "final": final})
    return answers


def main() -> None:
    rng = random.Random(SEED)
    papers = [simulated_paper(rng) for _ in range(PAPERS)]
    arms = {}
    for arm in bandit.ARMS:
        counts = [bandit.counterfactual(p, arm) for p in papers]
        rewards = [bandit.paper_reward(len(p), r, c) for p, (r, c) in zip(papers, counts)]
        arms[arm] = {"mean": round(mean(rewards), 4), "papers": PAPERS,
                     "mean_referred": round(mean(r for r, _ in counts), 2),
                     "mean_changed": round(mean(c for _, c in counts), 2)}
    assumptions = {"papers": PAPERS, "seed": SEED, "mcq_referred": MCQ_REFERRED, "written_referred": WRITTEN_REFERRED,
                   "exact": EXACT, "half_off": HALF_OFF, "wrong_cost": bandit.WRONG_COST}
    prior = {"source": "eval/bandit_simulator.py (simulated from stated assumptions; no exams, no sign-offs)",
             "assumptions": assumptions, "arms": {a: {"mean": v["mean"], "papers": v["papers"]} for a, v in arms.items()}}
    (ROOT / "monitoring" / "bandit_prior.json").write_text(json.dumps(prior, indent=2) + "\n")
    (ROOT / "eval" / "bandit_simulator.json").write_text(
        json.dumps({"assumptions": assumptions, "arms": arms}, indent=2) + "\n")
    print(json.dumps(arms, indent=2))


if __name__ == "__main__":
    main()
