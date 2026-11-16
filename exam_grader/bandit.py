"""
The reinforcement-learning layer (course bonus): a bandit that learns the written-answer review rule from the
teacher's corrections at sign-off.

Why this decision. The course brief lists "choosing the confidence threshold or the review rule and learning from
reviewers' corrections" as an example, and counts the bonus only if the app visibly behaves differently. A first
version chose the reader's confidence threshold (0.70 to 0.95); the reader is almost always above 0.95, so every
choice sent the same answers to review and nothing could be learnt. The written review rule does change what the
teacher sees on every paper: it decides which of Gemini's proposed marks for Q21-Q35 the teacher must mark
themselves, and which arrive pre-filled.

The arms (exam_grader.grading.Policy.written_review_at_or_below; the multiple-choice and reading rules stay fixed):
    usual     only the answers the grader itself is unsure about go to the teacher (no extra cutoff);
    zeros     also every written answer Gemini proposes 0/2 for;
    up_to_1   also every written answer Gemini proposes 1/2 or less for (the default before the bandit).
They are nested: each sends to review everything the one above it does, and more.

The reward arrives at sign-off, when the teacher has checked every mark of the paper:

    reward = -(REVIEW_COST x answers sent to review + WRONG_COST x proposals the teacher changed) / answers

An answer sent to review costs one unit of teacher time. A proposal the teacher had to change costs WRONG_COST:
the time to spot and fix it, plus the risk that a wrong mark is missed. The team set WRONG_COST = 5 reviews.

What it could reward by accident. A teacher who signs off without reading the proposals changes none of them, so a
lenient arm looks free: the bandit would learn to send less to review because nobody checks. The reward is only as
good as the review behind it. The app shows each proposal with its evidence and the time spent, but cannot prove
it was read; the report discusses this.

Exploration: Thompson sampling. Each arm keeps a normal posterior over its mean reward; every paper draws one
sample per arm and plays the best draw. An arm that is clearly worse is almost never drawn, so exploring costs the
teacher little once the evidence is in, which matters with a few papers a day. While it explores, the cost falls on
the paper graded with a more lenient arm: more pre-filled written marks for the teacher to check, never a mark that
skips sign-off.

Warm start. Before any real traffic each arm's prior comes from a simulator with stated assumptions
(eval/bandit_simulator.py), not from student exams. That simulated evidence is capped at PRIOR_PAPERS papers per arm,
so a handful of real sign-offs overrule it. The real part comes from the app's own logs (logbook.load_signoffs).
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path

CUTOFFS = {"usual": None, "zeros": 0, "up_to_1": 1}   # arm -> Policy.written_review_at_or_below
ARMS = tuple(CUTOFFS)
DEFAULT_ARM = "up_to_1"            # what the app used before the bandit, and what the API and checks keep
LABELS = {"usual": "only answers the grader is unsure about",
          "zeros": "also every proposed 0/2",
          "up_to_1": "also every proposed mark of 1/2 or less"}
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


def counterfactual(answers: list[dict], arm: str) -> tuple[int, int]:
    """Referred and changed answers if the paper had been graded under this arm. Each answer is a dict with written
    (bool), proposed (the mark proposed under the most lenient arm, or None when another rule sent it to review) and
    final (the teacher's mark). Only written proposals at or below the arm's cutoff move."""
    cutoff = CUTOFFS[arm]
    referred = changed = 0
    for a in answers:
        held = a["written"] and cutoff is not None and a["proposed"] is not None and a["proposed"] <= cutoff
        if a["proposed"] is None or held:
            referred += 1
        elif a["final"] is not None and a["proposed"] != a["final"]:
            changed += 1
    return referred, changed


def load_prior(path: Path = PRIOR_FILE) -> dict:
    """{arm: {"mean": m, "papers": n}} from the warm start, or empty if there is none."""
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))
    return {k: v for k, v in data["arms"].items() if k in CUTOFFS}


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


def choose(prior: dict, rewards: dict, rng: random.Random | None = None) -> tuple[str, dict]:
    """Thompson sampling: one draw per arm from its posterior, play the highest. Returns the arm and the state."""
    rng = rng or random.Random()
    state = posterior(prior, rewards)
    for arm, s in state.items():
        s["draw"] = rng.gauss(s["mean"], s["sd"])
    return max(state, key=lambda arm: state[arm]["draw"]), state


def rewards_from_logs(signoffs) -> dict:
    """{arm: [reward, ...]} from logbook.load_signoffs(), counting each paper's latest sign-off once. Sign-offs from
    before this version (logged with a reading threshold, a different decision) are ignored."""
    out: dict = {}
    if signoffs is None or len(signoffs) == 0 or "review_arm" not in signoffs:
        return out
    for row in signoffs.itertuples():
        arm, reward = row.review_arm, row.reward
        if arm in CUTOFFS and reward is not None and not (isinstance(reward, float) and math.isnan(reward)):
            out.setdefault(arm, []).append(float(reward))
    return out
