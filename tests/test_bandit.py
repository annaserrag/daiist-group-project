"""The review-threshold bandit (exam_grader/bandit.py): reward, offline counterfactual, learning and logging."""
import os
import random
import tempfile
import unittest
from pathlib import Path

from exam_grader import bandit


class Reward(unittest.TestCase):
    def test_reviews_and_changed_marks_cost_the_teacher(self):
        self.assertEqual(bandit.paper_reward(10, 0, 0), 0)
        self.assertEqual(bandit.paper_reward(10, 2, 0), -0.2)
        self.assertEqual(bandit.paper_reward(10, 0, 1), -bandit.WRONG_COST / 10)

    def test_higher_thresholds_refer_more_and_let_fewer_errors_through(self):
        answers = [{"reading_confidence": 0.75, "proposed": 1, "final": 0},   # misread, low confidence
                   {"reading_confidence": 0.99, "proposed": 1, "final": 1},
                   {"reading_confidence": 0.85, "proposed": 2, "final": 2},
                   {"reading_confidence": 0.99, "proposed": None, "final": 1}]  # sent to review by another rule
        self.assertEqual(bandit.counterfactual(answers, 0.70), (1, 1))
        self.assertEqual(bandit.counterfactual(answers, 0.80), (2, 0))
        self.assertEqual(bandit.counterfactual(answers, 0.90), (3, 0))


class Learning(unittest.TestCase):
    def test_real_sign_offs_overrule_the_warm_start(self):
        prior = {arm: {"mean": -0.2, "papers": 100} for arm in bandit.ARMS}
        state = bandit.posterior(prior, {0.70: [-1.0] * 20})
        self.assertEqual(state[0.70]["warm_start_papers"], bandit.PRIOR_PAPERS)
        self.assertLess(state[0.70]["mean"], state[0.80]["mean"])

    def test_thompson_sampling_plays_the_cheapest_arm_most(self):
        rewards = {0.70: [-0.5] * 10, 0.80: [-0.1] * 10, 0.90: [-0.3] * 10, 0.95: [-0.4] * 10}
        rng = random.Random(0)
        picks = [bandit.choose({}, rewards, rng)[0] for _ in range(300)]
        self.assertGreater(picks.count(0.80), 250)

    def test_with_no_evidence_every_arm_gets_played(self):
        rng = random.Random(1)
        self.assertEqual({bandit.choose({}, {}, rng)[0] for _ in range(200)}, set(bandit.ARMS))


class Logging(unittest.TestCase):
    def test_a_sign_off_reaches_the_bandit(self):
        with tempfile.TemporaryDirectory() as folder:
            import importlib
            import logbook
            os.environ["LOG_DIR"] = folder
            try:
                logbook = importlib.reload(logbook)
                logbook.log_signoff("paper-1", 0.9, 35, 7, 1, bandit.paper_reward(35, 7, 1))
                logbook.log_signoff("paper-1", 0.9, 35, 6, 1, bandit.paper_reward(35, 6, 1))  # signed again
                rewards = bandit.rewards_from_logs(logbook.load_signoffs())
                self.assertEqual(rewards, {0.9: [bandit.paper_reward(35, 6, 1)]})
            finally:
                del os.environ["LOG_DIR"]
                importlib.reload(logbook)


if __name__ == "__main__":
    unittest.main()
