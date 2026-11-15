"""The small rules the vision chapters rely on, checked without any data or model (vision/README.md says where each
rule was chosen). Run with `uv run python -m unittest tests.test_vision_rules`."""
import unittest
from types import SimpleNamespace

import numpy as np
import pandas as pd

from vision import datasets, devset, metrics, reader


class QuestionNumbers(unittest.TestCase):
    """reader.LEADING_NUMBER, chosen in step 4 (Part 4)."""

    def test_a_number_and_separator_open_a_question(self):
        self.assertEqual(reader.leading_number("26. Distinguish supervised"), ("26", None, "Distinguish supervised"))
        self.assertEqual(reader.leading_number("Q4) yes"), ("4", None, "yes"))

    def test_sub_parts_are_kept(self):
        self.assertEqual(reader.leading_number("3b) the mean")[:2], ("3", "b"))

    def test_up_to_three_stray_marks_before_the_number(self):
        self.assertEqual(reader.leading_number('"12. answer')[0], "12")
        self.assertEqual(reader.leading_number("p16. answer")[0], "16")

    def test_a_decimal_is_not_a_question(self):
        self.assertEqual(reader.leading_number("1.5 is the ratio")[0], None)

    def test_a_line_without_a_number_stays_text(self):
        self.assertEqual(reader.leading_number("the answer is B"), (None, None, "the answer is B"))


class MatchingTheKey(unittest.TestCase):
    """reader.choice_letter and reader.answers_by_number, used by the grader and step 7."""

    def test_one_letter_alone_is_the_choice(self):
        self.assertEqual(reader.choice_letter("B"), "B")
        self.assertEqual(reader.choice_letter("(c)"), "C")

    def test_no_letter_or_two_letters_go_to_review(self):
        self.assertIsNone(reader.choice_letter("A and C"))
        self.assertIsNone(reader.choice_letter("it is a model that learns from labelled data"))

    def test_a_sub_part_the_key_lacks_falls_back_to_the_number(self):
        result = {"questions": [{"number": "21", "part": "a", "text": "x", "ocr_conf": 0.9}]}
        self.assertIn("21", reader.answers_by_number(result, expected={"21", "22"}))

    def test_a_number_found_twice_is_joined_and_marked(self):
        result = {"questions": [{"number": "5", "part": None, "text": "first", "ocr_conf": 0.9},
                                {"number": "5", "part": None, "text": "second", "ocr_conf": 0.4}]}
        answer = reader.answers_by_number(result)["5"]
        self.assertTrue(answer["duplicate"])
        self.assertEqual((answer["text"], answer["ocr_conf"]), ("first second", 0.4))


class BlankPages(unittest.TestCase):
    """reader.ink and BLANK_INK, chosen in step 4 (Part 5) from step 3's ink table."""

    def test_a_noisy_ruled_page_has_no_ink(self):
        page = np.full((400, 300), 250, np.uint8)
        page[::40] = 200  # ruled lines, lighter than 60 grey levels below the background
        self.assertLess(reader.ink(page), reader.BLANK_INK)

    def test_a_light_pen_still_counts(self):
        page = np.full((400, 300), 250, np.uint8)
        page[100:110, 50:250] = 150  # a stroke 100 levels darker than the paper
        self.assertGreater(reader.ink(page), reader.BLANK_INK)


class DevelopmentSet(unittest.TestCase):
    """The truth the readers are measured against (devset.py) and the scoring (metrics.py)."""

    def test_crossed_out_marks_and_extra_spaces_are_removed(self):
        self.assertEqual(devset.clean_truth("the #wrong#  answer \n\n 2) next"), "the wrong answer\n\n2) next")

    def test_numbers_that_open_a_line(self):
        self.assertEqual(devset.numbers_in("1) one\nnot 2) here\nQ3. three\n4 no separator"), ["1", "3"])

    def test_character_error_rate(self):
        self.assertEqual(metrics.cer("the cat", "the cat"), 0)
        self.assertAlmostEqual(metrics.cer("the bat", "the cat"), 1 / 7)
        self.assertEqual(metrics.cer("a , b", "a, b"), 0)  # layout, not text

    def test_answer_sheet_numbers_are_capped_at_the_boxes(self):
        row = SimpleNamespace(kind="answersheet", n_boxes=2, numbers=[])
        self.assertEqual(metrics.number_scores(["1", "2", "3"], row), (1.0, 1))


class TheSplit(unittest.TestCase):
    """datasets.make_split, step 1 (Part 3): no photo and no group in two splits."""

    def test_groups_never_cross_splits(self):
        rows = []
        for booklet in ("31-3-3-Science-Topper-Solution", "Social_Science_087", "32-5-1-Social-Science-Topper-Solution",
                        "Maths_001"):
            for page in range(2):
                for copy in range(3):
                    source = f"{booklet}_page_{page}_png"
                    rows.append(dict(dataset="answersheet", roboflow_split="train", file=f"{source}.rf.{copy}.jpg",
                                     source=source, group=datasets.group_of("answersheet", source), n_boxes=1))
        split = datasets.make_split(pd.DataFrame(rows))
        self.assertEqual(datasets.leakage_after_split(split)["in_more_than_one_split"].sum(), 0)
        self.assertEqual(set(split["our_split"]), {"train", "val", "test"})
        # in validation and test only one copy of each page is used
        held_out = split[split["our_split"] != "train"]
        self.assertEqual(held_out["used"].sum(), held_out["source"].nunique())


if __name__ == "__main__":
    unittest.main()
