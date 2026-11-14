import csv
import hashlib
import json
import unittest
from collections import Counter
from copy import deepcopy
from pathlib import Path

from pypdf import PdfReader

from evaluate import compare_case, match_expected
from exam_grader.grading import grade_exam


ROOT = Path(__file__).resolve().parents[1]


class PdfCasePacket(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with (ROOT / "eval/cases.csv").open(encoding="utf8", newline="") as stream:
            cls.cases = list(csv.DictReader(stream))
        cls.refs = json.loads((ROOT / "eval/case_references.json").read_bytes())

    def test_packet_and_frozen_expectations(self):
        ids = [c["id"] for c in self.cases]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(set(ids), set(self.refs))
        self.assertGreaterEqual(len(ids), 30)
        kinds = Counter(c["kind"] for c in self.cases)
        for kind, required in dict(normal=1, hard=5, review=5, out_of_scope=3).items():
            self.assertGreaterEqual(kinds[kind], required)
        for case in self.cases:
            with self.subTest(case=case["id"]):
                ref = self.refs[case["id"]]
                path = ROOT / case["input"]
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), ref["sha256"])
                self.assertTrue(all(not p.extract_text().strip() for p in PdfReader(path).pages))
                expected = json.loads(case["expected"])
                self.assertEqual(ref["exam_status"], expected["exam_status"])
                self.assertEqual(set(ref["states"]), set(ref["readings"]))
                for q, mark in expected.get("marks", {}).items():
                    self.assertIn(q, ref["readings"])
                    self.assertEqual(mark, ref["reference_marks"][q])
                for q in expected.get("review_reasons", {}):
                    self.assertIn(q, ref["readings"])
                self.assertTrue(case["baseline"] and case["notes"])

    def test_scoring_reference_replay_and_corrupt_handoffs(self):
        # Offline oracle only. Never registered as a reader or used to measure the real system.
        for case in self.cases:
            with self.subTest(case=case["id"]):
                ref = self.refs[case["id"]]
                observations = [dict(question_id=int(q), text=t, state=ref["states"][q],
                                     confidence=1, id_confidence=1)
                                for q,values in ref["readings"].items() for t in values]
                payload = dict(schema_version="vision.exam.v1", exam_status=ref["exam_status"],
                               answers=observations)
                trial = grade_exam(payload)
                teacher = trial["teacher"] | {"input_type":"pdf", "reader_failed":False}
                record = trial["record"]
                self.assertTrue(compare_case(teacher, record, ref)["reader_exact_cases"])
                missing = dict(teacher, input_type="pdf_awaiting_reader")
                self.assertFalse(compare_case(missing, record, ref)["reader_structure_ok"])
                self.assertFalse(match_expected(case["expected"], missing))
                if not observations:
                    continue
                changed = deepcopy(teacher)
                observed = next(row for row in changed["answers"] if row["readings"])
                observed["readings"][0] = "wrong transcription"
                self.assertFalse(compare_case(changed, record, ref)["reader_exact_cases"])
                observed["reading_states"] = []
                self.assertFalse(compare_case(changed, record, ref)["reader_structure_ok"])
