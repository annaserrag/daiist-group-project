"""The online path: without the page reader, a demo PDF uses its stored reading and any other PDF is sent back."""
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pipeline
from exam_grader import precomputed, vision_finetune


class PrecomputedReads(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.patches = [patch.object(precomputed, "DIR", self.root / "reads"),
                        patch.dict(os.environ, {"GRADE_READER": "precomputed"})]
        for p in self.patches:
            p.start()

    def tearDown(self):
        for p in reversed(self.patches):
            p.stop()
        self.tmp.cleanup()

    def pdf(self, name, content):
        path = self.root / name
        path.write_bytes(b"%PDF-1.4\n" + content)
        return path

    def test_stored_reading_is_used_without_the_reader(self):
        payload = json.loads(pipeline.EXAMPLE.read_text(encoding="utf-8"))
        demo = self.pdf("demo.pdf", b"demo")
        precomputed.save(demo, vision_finetune.VERSION, payload, "demo.pdf")
        result = pipeline.run_views(demo)
        self.assertEqual(result["record"]["versions"]["component_a"], vision_finetune.VERSION + "/precomputed")
        self.assertEqual(result["teacher"]["input_type"], "pdf")
        self.assertFalse(result["teacher"]["reader_failed"])

    def test_other_pdf_is_sent_back(self):
        result = pipeline.run_views(self.pdf("new.pdf", b"new"))
        self.assertEqual(result["teacher"]["input_type"], "pdf_awaiting_reader")

    def test_reading_from_an_older_reader_is_ignored(self):
        demo = self.pdf("old.pdf", b"old")
        precomputed.save(demo, "an-older-reader", {"answers": []}, "old.pdf")
        self.assertIsNone(precomputed.lookup(demo, vision_finetune.VERSION))


if __name__ == "__main__":
    unittest.main()
