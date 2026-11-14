import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

import pipeline
from exam_grader.retrieval import Index
from exam_grader.selected_reader import compare_reads
from exam_grader.written_exam import WrittenExam


def lines(*texts):
    return [(s, [.98] * len(s)) for s in texts]


class SelectedPDF(unittest.TestCase):
    def test_missing_extra_or_changed_parts_are_not_graded(self):
        ids = {"a": 3, "c": 9}
        original = lines("A) B", "C) Carbon dioxide.")
        self.assertEqual(compare_reads(original, original, ids)[0]["text"], "B")
        self.assertEqual(compare_reads(original, lines("C) Carbon dioxide."), ids)[0]["id_confidence"], 0)
        self.assertEqual(compare_reads(original, lines("A) B", "C) Carbon monoxide."), ids)[1]["confidence"], 0)
        self.assertIsNone(compare_reads(lines("D. Other"), lines("D. Other"), ids)[0]["question_id"])
        self.assertEqual(len(compare_reads(lines("A. one", "A. two"), lines("A. one", "A. two"), ids)), 2)

    def test_inline_parts_preserve_words_labels_and_original_confidence(self):
        ids = {"d": 7, "e": 4, "f": 2}
        text = "D. A first sentence. E. Next answer! (F) Last answer."
        scores = [.98] * len(text)
        scores[text.index("E.")] = .62
        rows = compare_reads([(text, scores)], lines("D. A first sentence.", "E. Next answer!", "(F) Last answer."), ids)
        self.assertEqual([(r["question_id"], r["text"]) for r in rows],
                         [(7, "A first sentence."), (4, "Next answer!"), (2, "Last answer.")])
        self.assertEqual(rows[1]["id_confidence"], .62)
        self.assertTrue(all(r["confidence"] == .98 for r in rows))

    def test_inline_initials_references_duplicates_and_unconfirmed_labels(self):
        ids = {"a": 1, "b": 2, "c": 3}
        for text in ("A. Dr. B. Smith mentioned vitamin B. It helps.",
                     "A. J. B. Smith chose option (C) in May.",
                     "A. See Appendix B. It has a table. X. Unknown."):
            rows = compare_reads(lines(text), lines(text), ids)
            self.assertEqual([r["text"] for r in rows], [text[3:]])
        first = lines("A) Starting", "here. B) Next. B) Conflict.")
        self.assertEqual([r["question_id"] for r in compare_reads(first, first, ids)], [1, 2, 2])
        conflict = compare_reads(first, lines("A) Starting here. B) Next."), ids)
        self.assertTrue(all(r["id_confidence"] == 0 for r in conflict))
        loose = lines("Unassigned answer. B. More text.")
        self.assertEqual([r["question_id"] for r in compare_reads(loose, loose, ids)], [None])

    def test_pdf_entry_point_uses_definition_and_fails_closed_on_another_exam(self):
        exam = WrittenExam.model_validate(dict(title="Chemistry", questions=[dict(
            question_id=3, paper_label="A", question="Name the gas.", reference_answer="secret reference",
            criteria=[dict(id="name", points=2, description="secret criterion")])]))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "test.pdf"
            path.write_bytes(b"%PDF")
            for identity, expected in [("SAME", "in_scope"), ("DIFFERENT", "out_of_scope"), ("UNKNOWN", "unreadable")]:
                responses = [lines(identity), lines("A. B"), lines("A. B")]
                with patch.dict("os.environ", {"GRADE_VISION_MODULE": "exam_grader.vision_finetune", "GRADE_VISION_ROOT": ""}), patch.object(
                    pipeline, "resources", return_value=(Index(), None, set())
                ), patch("vision.reader.load_pages", return_value=[np.pad(np.zeros((5, 5), dtype=np.uint8), 10, constant_values=255)]), patch(
                    "exam_grader.selected_reader.page_lines", side_effect=responses
                ) as read:
                    teacher = pipeline.run_views(path, written_exam=exam)["teacher"]
                self.assertFalse(teacher["reader_failed"])
                self.assertEqual(teacher["exam_status"], expected)
                self.assertNotIn("secret", repr([c.kwargs["prompt"] for c in read.call_args_list]))
                if expected == "in_scope":
                    self.assertEqual(teacher["answers"][0]["readings"], ["B"])
                    self.assertIn(teacher["answers"][0]["reason"], ("weak_retrieval", "provider_disabled"))
