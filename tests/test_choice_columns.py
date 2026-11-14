import unittest


from PIL import Image

from exam_grader.choice_columns import verify
from exam_grader.page_confidence import verify_crop_words
from exam_grader.vision_finetune import assign_unresolved
from exam_grader.grading import grade_exam


class ChoiceColumnTests(unittest.TestCase):
    def test_no_sequence_repairs_or_silently_dropped_rows(self):
        def scored(line):
            return line, [.98]*len(line)
        cells = [(Image.new("L", (20, 16)), Image.new("L", (20, 16)), clean) for clean in [True, True, False]]
        outputs = iter([[scored(t) for t in ["6 A", "6 C", "9 B"]],
                        [scored("6")], [scored("A")], [scored("8")], [scored("C")], [scored("9")]])
        result = verify([cells], lambda *_: next(outputs))
        self.assertEqual(result, [("6", .98, "A", .98), (None, 0, "C", .98), ("9", .98, "?", 0)])
        self.assertEqual(verify([cells], lambda *_: [scored("6 A")]), [])

    def test_case_equivalence_keeps_symbols_negation_and_numbers_exact(self):
        text = "K in K-means algorithms"
        words = text.split()
        def check(phrase):
            return verify_crop_words(text, [.55]*4, words, [[(phrase, [.96]*len(phrase))]], course_terms=True)
        self.assertEqual(check("K in k-means algorithms"), .96)
        for phrase in ["k in k-means algorithms", "K in k means algorithms", "K in k-means algorithm"]:
            self.assertEqual(check(phrase), .55)
        original = "not Supervised Learning"
        for phrase in ["Supervised Learning", "Not supervised learning", "not unsupervised learning"]:
            self.assertEqual(verify_crop_words(original, [.55]*3, original.split(), [[(phrase, [.96]*len(phrase))]],
                                              course_terms=True), .55)

    def test_human_assignment_preserves_evidence_and_duplicate_gate(self):
        ans = {"question_id": None, "text": "B", "id_confidence": 0., "confidence": .72, "state": "read"}
        payload = {"schema_version": "vision.exam.v1", "exam_status": "in_scope", "answers": [ans, dict(ans, question_id=1)]}
        for pair in [(0,1), (2,2), (1,36)]:
            with self.assertRaises(ValueError):
                assign_unresolved(payload, [pair])
        assigned = assign_unresolved(payload, [(1,1)])
        self.assertIsNone(payload["answers"][0]["question_id"])
        self.assertEqual(assigned["answers"][0]["confidence"], .72)
        self.assertEqual(grade_exam(assigned, index=None)["teacher"]["answers"][0]["reason"], "duplicate_detection")
