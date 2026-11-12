import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pypdf import PdfWriter

from exam_grader.grading import checked
from exam_grader.materials import add_pdf
from exam_grader.retrieval import Index


SAMPLE = Path(__file__).resolve().parents[1] / "eval/ml-course-notes.pdf"
QUOTE = "Supervised learning trains a model on examples paired with known target labels."


class CourseMaterials(unittest.TestCase):
    def test_page_provenance_and_duplicate_upload(self):
        base = Index()
        library, name, _ = add_pdf({}, "ML", SAMPLE, base)
        course = library[name]
        hits = course.search(21)
        self.assertEqual(hits[0]["page"], 3)
        self.assertIn(QUOTE, hits[0]["text"])
        response = dict(mark=2, needs_review=False, deductions=[], feedback="Recognizes labeled examples.",
                        passage_id=hits[0]["id"], quote=QUOTE, criterion_evidence=[
                            dict(criterion="definition", coverage="full", passage_id=hits[0]["id"], quote=QUOTE)])
        cite = checked(21, response, hits, 1)["citation"]
        self.assertEqual((cite["title"], cite["page"], cite["quote"]), (SAMPLE.name, 3, QUOTE))
        wrong = dict(response, passage_id=hits[1]["id"])
        self.assertEqual(checked(21, wrong, hits, 1)["reason"], "unverified_citation")
        repeated, name, _ = add_pdf(library, "ml", SAMPLE, base)
        self.assertIs(repeated[name], course)
        self.assertFalse(any("page" in row for row in base.rows))

    def test_title_only_and_partial_support_do_not_award_or_deduct_marks(self):
        library, name, _ = add_pdf({}, "ML", SAMPLE, Index())
        hits = library[name].search(29)
        hit = next(h for h in hits if h["page"] == 2)
        scale = dict(criterion="scale", coverage="full", passage_id=hit["id"],
                     quote="A feature with a much larger range can dominate a distance calculation.")
        title = dict(scale, criterion="distance", quote="Component 2 | Machine learning course notes")
        for needs_review in (False, True):
            proposal = dict(mark=2, needs_review=needs_review, feedback="Correct.", deductions=[],
                            passage_id=hit["id"], quote=scale["quote"], criterion_evidence=[scale,title])
            decision = checked(29, proposal, hits, 1)
            self.assertIsNone(decision["mark"])
            self.assertEqual(decision["criteria_needing_evidence"], ["distance"])
            self.assertEqual(decision["citation"]["quote"], scale["quote"])
            self.assertEqual(decision["citation"]["page"], 2)
        # A quote can support only part of a single criterion.
        partial = dict(scale, criterion="purpose", coverage="partial")
        decision = checked(28, proposal | {"criterion_evidence":[partial]}, hits, 1)
        self.assertEqual((decision["status"], decision["citation"]["coverage"]), ("review","partial"))
        old = checked(29, {k:v for k,v in proposal.items() if k != "criterion_evidence"}, hits, 1)
        self.assertEqual((old["mark"],old["citation"]), (None,None))
        # Even a verifier claiming full coverage cannot use a scaling fact for neighbor selection.
        duplicate = dict(scale, criterion="distance")
        decision = checked(29, proposal | {"needs_review":False, "criterion_evidence":[scale,duplicate]}, hits, 1)
        self.assertEqual((decision["mark"], decision["criteria_needing_evidence"]), (None,["distance"]))
        selection = "K-NN selects the closest neighbors by distance."
        more = dict(hit, id="other", page=4, text=selection)
        evidence = [scale, dict(duplicate, passage_id=more["id"], quote=selection)]
        positive = checked(29, proposal | {"needs_review":False, "criterion_evidence":evidence}, [*hits,more], 1)
        self.assertEqual((positive["mark"],positive["criterion_evidence"]["distance"]["page"]), (2,4))

    def test_failed_upload_preserves_previous_course_and_page_numbers(self):
        base = Index()
        library, _, _ = add_pdf({}, "ML", SAMPLE, base)
        with tempfile.TemporaryDirectory() as folder:
            blank = Path(folder) / "blank.pdf"
            pdf = PdfWriter()
            pdf.add_blank_page(width=200, height=200)
            pdf.write(blank)
            with self.assertRaisesRegex(ValueError, "selectable text"):
                add_pdf(library, "ML", blank, base)
            pdf.append(SAMPLE)
            shifted = Path(folder) / "prefaced.pdf"
            pdf.write(shifted)
            other, _, message = add_pdf(library, "Another course", shifted, base)
            self.assertIn("Skipped 1", message)
            self.assertEqual(other["Another course"].search(21)[0]["page"], 4)
            self.assertEqual(other["ML"].search(21)[0]["page"], 3)
            combined, _, _ = add_pdf(library, "ML", shifted, base)
            self.assertEqual(len(combined["ML"].rows), 6)
            self.assertEqual(len(library["ML"].rows), 3)

    def test_embedding_failure_is_atomic(self):
        base = Index(semantic=True)
        with patch.dict("os.environ", {"GEMINI_API_KEY": "test"}), patch(
            "exam_grader.materials.vector", side_effect=[[0] * 256, RuntimeError()]
        ):
            library = {}
            with self.assertRaisesRegex(ValueError, "could not embed"):
                add_pdf(library, "ML", SAMPLE, base)
            self.assertEqual(library, {})
