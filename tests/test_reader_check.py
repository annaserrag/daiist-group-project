import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from exam_grader.check_reader import compare
from exam_grader.integration import connect


class ReaderAcceptance(unittest.TestCase):
    def test_combined_cost_needs_an_explicit_pdf_meter_and_ignores_historical_json_usage(self):
        base = {"schema_version":"vision.exam.v1", "exam_status":"unreadable", "answers":[]}
        def run(usage, possible=True):
            return connect(None, read_exam=lambda _:base | {"usage":usage}, vision_version="example",
                           reader_charges_possible=possible, index=None)["record"]
        unknown = run(None)
        self.assertFalse(unknown["outputs"]["total_usage"]["cost_known"])
        measured = run({"cost_eur":0.001, "tokens":1200})
        self.assertTrue(measured["outputs"]["total_usage"]["cost_known"])
        self.assertEqual((measured["tokens"], measured["cost_eur"]), (1200, 0.001))
        replay = run({"cost_eur":0.001, "tokens":1200}, False)
        self.assertEqual((replay["tokens"], replay["cost_eur"]), (0, 0))
        self.assertIsNone(run({"cost_eur":-1})["outputs"]["reader_usage"]["cost_eur"])

    def test_match_requires_readings_and_status_from_the_pdf_reader(self):
        base = {"schema_version":"vision.exam.v1", "exam_status":"in_scope", "answers":[
            {"question_id":1, "text":"B", "confidence":1.0, "id_confidence":1.0, "state":"read"}]}
        with tempfile.TemporaryDirectory() as tmp:
            pdf, reference = Path(tmp) / "fixture.pdf", Path(tmp) / "gold.json"
            pdf.write_bytes(b"%PDF-1.4\n")
            reference.write_text(json.dumps(base))
            def check(actual):
                reader = SimpleNamespace(VERSION="fixture", read_exam=lambda _: actual)
                return compare(pdf, reference, reader)
            self.assertTrue(check(base)["exact_handoff"])
            for payload in (base | {"exam_id":"different-paper"}, base | {"answers":[]}, base | {"exam_status":"unreadable"},
                            base | {"answers":[base["answers"][0] | {"text":"A"}]}, {}):
                self.assertFalse(check(payload)["exact_handoff"])
            self.assertEqual(check(base | {"answers":[]})["missing_ids"], [1])
