"""On-demand suggestions: a held-back written answer gets one when the teacher opens it, never a mark."""
import unittest
from unittest.mock import patch

import app


def context(remote_ok=True, suggestable=True):
    item = dict(question="Define supervised learning.", reading="trained on labelled data", key="A type of ML",
                rubric=[], sources=[], passages=[], action="Check the question number.",
                reason="uncertain question id", suggestion=None, suggestable=suggestable)
    return {"id": "req-1", "remote_ok": remote_ok, "limits": [(21, 2)], "referrals": {21: item}}


class OnDemandSuggestions(unittest.TestCase):
    def setUp(self):
        app._SUGGESTIONS.clear()

    def test_opening_a_held_back_answer_shows_a_suggestion_once(self):
        found = {"mark": 1.5, "feedback": "Identifies labels.", "citation": {"title": "Supervised learning",
                                                                            "quote": "Uses labelled examples."}}
        with patch.object(app.pipeline, "suggest_written", return_value=found) as ask:
            card = app.question_view(context(), 21)[0]
            app.question_view(context(), 21)
        self.assertIn("We suggest 1.5/2", card)
        self.assertIn("Uses labelled examples.", card)
        self.assertEqual(ask.call_count, 1)
        # the radio buttons start empty: the teacher still chooses the mark
        self.assertIsNone(app.question_view(context(), 21)[3]["value"])

    def test_no_suggestion_when_the_file_may_not_go_to_gemini(self):
        with patch.object(app.pipeline, "suggest_written") as ask:
            card = app.question_view(context(remote_ok=False), 21)[0]
        self.assertNotIn("We suggest", card)
        ask.assert_not_called()

    def test_a_failed_suggestion_leaves_the_card_as_it_was(self):
        with patch.object(app.pipeline, "suggest_written", side_effect=RuntimeError("quota")):
            card = app.question_view(context(), 21)[0]
        self.assertNotIn("We suggest", card)
        self.assertIn("Define supervised learning.", card)


if __name__ == "__main__":
    unittest.main()
