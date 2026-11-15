"""
The reinforcement-learning layer (course bonus): a bandit that learns the review threshold from the teacher's
sign-offs in the app.

The decision. Every answer the reader transcribes has a reading confidence. Below a threshold (Policy.min_read)
the answer goes to the teacher for review; above it the grader proposes a mark. A low threshold saves the teacher
reviews but lets more misread answers through; a high one catches them at the price of more reviews. Today the
threshold is fixed at 0.80, an engineering default nobody calibrated (exam_grader/grading.py). The bandit chooses it
per paper among ARMS, and learns which one costs the teacher least.

The reward arrives at sign-off, when the teacher has checked every mark of the paper:

    reward = -(REVIEW_COST x answers sent to review + WRONG_COST x proposals the teacher changed) / answers

An answer sent to review costs one unit of teacher time. A proposal the teacher had to change costs WRONG_COST:
the time to spot and fix it, plus the risk that a wrong mark is missed. The team set WRONG_COST = 5 reviews.

Exploration: Thompson sampling. Each arm keeps a normal posterior over its mean reward; every paper draws one
sample per arm and plays the best draw. An arm that is clearly worse is almost never drawn, so exploring costs
the teacher little once the evidence is in, which matters with a few papers a day.

Warm start. Before any real traffic, each arm's prior comes from a replay of the team's development packet
(eval/bandit_warmup.py): one reading of each paper with every answer graded, so the cost under every threshold can
be counted offline, because the thresholds are nested. That simulated evidence is capped at PRIOR_PAPERS papers per
arm, so a handful of real sign-offs can overrule it. The real part comes from the app's own logs
(logbook.load_signoffs).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

ARMS = (0.70, 0.80, 0.90, 0.95)   # reading-confidence thresholds, chosen by the team around today's 0.80
DEFAULT_ARM = 0.80                 # what the app used before the bandit, and what the API and checks keep
REVIEW_COST, WRONG_COST = 1.0, 5.0
PRIOR_PAPERS = 5                   # the warm start counts as at most this many papers per arm
NOISE_SD = 0.3                     # spread of one paper's reward around its arm's mean (per-answer cost units)
PRIOR_SD = 1.0                     # how unsure we are about an arm with no evidence at all
PRIOR_FILE = Path(__file__).resolve().parents[1] / "monitoring" / "bandit_prior.json"


def paper_reward(answers: int, referred: int, changed: int) -> float:
    """Minus the teacher's cost per answer for one signed-off paper."""
    if answers <= 0:
        raise ValueError("A paper needs at least one answer.")
    return -(REVIEW_COST * referred + WRONG_COST * changed) / answers


def counterfactual(answers: list[dict], threshold: float) -> tuple[int, int]:
    """Referred and changed answers if the paper had been graded at this threshold. Each answer is a dict with
    reading_confidence, proposed (the mark proposed when not held back by the threshold, or None when another rule
    sent it to review) and final (the teacher's mark). Only the reading gate moves with the threshold."""
    referred = changed = 0
    for a in answers:
        conf = a.get("reading_confidence")
        if a["proposed"] is None or conf is None or conf < threshold:
            referred += 1
        elif a["final"] is not None and a["proposed"] != a["final"]:
            changed += 1
    return referred, changed


def load_prior(path: Path = PRIOR_FILE) -> dict:
    """{arm: {"mean": m, "papers": n}} from the warm start, or empty if there is none."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {float(k): v for k, v in data["arms"].items()}


def posterior(prior: dict, rewards: dict) -> dict:
    """Normal posterior of each arm's mean reward: the warm start (as up to PRIOR_PAPERS pseudo-papers) combined
    with the real sign-off rewards, each with noise NOISE_SD."""
    out = {}
    for arm in ARMS:
        mean0, n0 = 0.0, 0
        if arm in prior:
            mean0, n0 = prior[arm]["mean"], min(prior[arm]["papers"], PRIOR_PAPERS)
        precision = 1 / PRIOR_SD**2 + (n0 + len(rewards.get(arm, []))) / NOISE_SD**2
        total = mean0 * n0 + sum(rewards.get(arm, []))
        mean = (total / NOISE_SD**2) / precision
        out[arm] = {"mean": mean, "sd": math.sqrt(1 / precision), "real_papers": len(rewards.get(arm, [])),
                    "warm_start_papers": n0}
    return out


def choose(prior: dict, rewards: dict, rng: random.Random | None = None) -> tuple[float, dict]:
    """Thompson sampling: one draw per arm from its posterior, play the highest. Returns the arm and the state."""
    rng = rng or random.Random()
    state = posterior(prior, rewards)
    for arm, s in state.items():
        s["draw"] = rng.gauss(s["mean"], s["sd"])
    return max(state, key=lambda arm: state[arm]["draw"]), state


def rewards_from_logs(signoffs) -> dict:
    """{arm: [reward, ...]} from logbook.load_signoffs(), counting each paper's latest sign-off once."""
    out: dict = {}
    if signoffs is None or len(signoffs) == 0:
        return out
    for row in signoffs.itertuples():
        arm = getattr(row, "review_threshold", None)
        if arm is not None and not (isinstance(arm, float) and math.isnan(arm)) and float(arm) in ARMS:
            out.setdefault(float(arm), []).append(float(row.reward))
    return out
