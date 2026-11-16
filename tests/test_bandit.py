"""The review-rule bandit (exam_grader/bandit.py): reward, offline counterfactual, learning and logging."""
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

    def test_stricter_rules_refer_more_written_answers_and_let_fewer_errors_through(self):
        answers = [{"written": True, "proposed": 0, "final": 0.5},     # wrong zero
                   {"written": True, "proposed": 1, "final": 1},
                   {"written": True, "proposed": 2, "final": 1.5},     # wrong, but above every cutoff
                   {"written": True, "proposed": None, "final": 1},    # sent to review by another rule
                   {"written": False, "proposed": 0, "final": 0}]      # multiple choice: no arm moves it
        self.assertEqual(bandit.counterfactual(answers, "usual"), (1, 2))
        self.assertEqual(bandit.counterfactual(answers, "zeros"), (2, 1))
        self.assertEqual(bandit.counterfactual(answers, "up_to_1"), (3, 1))

    def test_every_arm_is_a_valid_grading_policy(self):
        from exam_grader.grading import Policy
        for cutoff in bandit.CUTOFFS.values():
            Policy(written_review_at_or_below=cutoff)


class Learning(unittest.TestCase):
    def test_real_sign_offs_overrule_the_warm_start(self):
        prior = {arm: {"mean": -0.2, "papers": 100} for arm in bandit.ARMS}
        state = bandit.posterior(prior, {"usual": [-1.0] * 20})
        self.assertEqual(state["usual"]["warm_start_papers"], bandit.PRIOR_PAPERS)
        self.assertLess(state["usual"]["mean"], state["zeros"]["mean"])

    def test_thompson_sampling_plays_the_cheapest_arm_most(self):
        rewards = {"usual": [-0.5] * 10, "zeros": [-0.1] * 10, "up_to_1": [-0.3] * 10}
        rng = random.Random(0)
        picks = [bandit.choose({}, rewards, rng)[0] for _ in range(300)]
        self.assertGreater(picks.count("zeros"), 250)

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
                logbook.log_signoff("paper-1", "zeros", 35, 7, 1, bandit.paper_reward(35, 7, 1))
                logbook.log_signoff("paper-1", "zeros", 35, 6, 1, bandit.paper_reward(35, 6, 1))  # signed again
                rewards = bandit.rewards_from_logs(logbook.load_signoffs())
                self.assertEqual(rewards, {"zeros": [bandit.paper_reward(35, 6, 1)]})
            finally:
                del os.environ["LOG_DIR"]
                importlib.reload(logbook)


if __name__ == "__main__":
    unittest.main()
