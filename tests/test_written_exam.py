import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import logbook
import pipeline
from exam_grader.grading import checked, Policy
from exam_grader.materials import add_pdf
from exam_grader.retrieval import Index
from exam_grader.written_exam import WrittenExam
from exam_grader.written_transfer import DATA


class WrittenTransfer(unittest.TestCase):
    def test_teacher_definition_selects_new_questions_and_marks_through_signoff(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            base = Index()
            with patch.dict(os.environ, {"GEMINI_API_KEY":""}), patch.object(
                    pipeline, "resources", return_value=(base,None,set())), patch.object(
                    logbook, "LOG_DIR", root), patch.object(logbook, "ARCHIVE_DIR", root / "archive"):
                state, info, _ = app.select_exam(DATA / "waves-exam.json")
                exam, _ = state
                self.assertIn(exam.exam_id, info)
                library, course, _ = add_pdf({}, "Waves", DATA / "waves-course.pdf", base)
                self.assertEqual(app.preview_material(32, library, course, state)[0][1], 2)
                payload = dict(schema_version="vision.exam.v1", exam_id=exam.exam_id, exam_status="in_scope",
                               answers=[dict(question_id=1, text="", state="confirmed_blank", confidence=1,
                                             id_confidence=1), dict(question_id=32, text="SVM", state="read",
                                             confidence=1, id_confidence=1)])
                p = root / "reading.json"
                p.write_text(json.dumps(payload))
                status, _, marks, ctx, _ = app.display(p, library, course, state)
                self.assertIn("1 of 2", status)
                self.assertEqual(marks, [[1,0],[32,None]])
                result = app.sign_off(ctx, [[1, .5],[32, 1.5]])
                self.assertEqual((result["maximum"],result["total"]), (4,2))
                for ident in (None, "another-exam"):
                    p.write_text(json.dumps(payload | {"exam_id":ident}))
                    view = pipeline.run_views(p, written_exam=exam, course_index=library[course])
                    self.assertEqual(view["teacher"]["exam_status"], "out_of_scope")
                    self.assertTrue(all(r["mark"] is None for r in view["teacher"]["answers"]))
                p.write_text(json.dumps(payload))
                self.assertEqual(pipeline.run_views(p)["teacher"]["exam_status"], "out_of_scope")

    def test_validation_and_binding_to_rubric_revision(self):
        exam = WrittenExam.load(DATA / "waves-exam.json")
        data = exam.model_dump()
        data["questions"][0]["criteria"][0]["description"] += " Check the dimensions."
        self.assertNotEqual(exam.exam_id, WrittenExam.model_validate(data).exam_id)
        data["questions"][0]["criteria"][0]["points"] = .5
        with self.assertRaises(ValueError):
            WrittenExam.model_validate(data)
        data["questions"][0]["criteria"][0]["points"] = 1.0
        data["questions"][1]["question_id"] = 1
        with self.assertRaises(ValueError):
            WrittenExam.model_validate(data)

    def test_second_assessment_and_unsupported_criteria(self):
        quote = "The time per cycle is the reciprocal of the frequency."
        hit = dict(id="p",text=quote,title="Waves",heading="Period",url="",score=.9)
        entries = [dict(criterion=c,coverage="full",quote=quote,passage_id="p") for c in ("change","reason")]
        response = dict(mark=2,needs_review=False,feedback="Correct explanation.",deductions=[],
                        quote=quote,passage_id="p",criterion_evidence=entries)
        def grade(second, evidence=entries):
            return checked(32, response | {"independent_mark":second, "criterion_evidence":evidence}, [hit], 1,
                           custom_guide={"change":(1,"Direction"),"reason":(1,"Reason")})
        for second in (0, 1, None):
            row = grade(second)
            self.assertEqual((row["reason"],row["mark"]), ("mark_disagreement", None))
            self.assertEqual(set(row["criterion_evidence"]), {"change","reason"})
        self.assertEqual(grade(1.5)["mark"], 2)
        self.assertIsNone(grade(2, entries[:1])["mark"])
        response["quote"] = "Two fragments ... joined by the assessor."
        self.assertEqual(grade(2)["citation"]["quote"], quote)
        self.assertEqual(grade(2)["mark"], 2)
        self.assertIsNone(grade(2, entries[:1])["mark"])
        response["passage_id"] = "invented"
        self.assertIsNone(grade(2)["mark"])


    def test_missing_source_example_cannot_justify_a_deduction(self):
        from exam_grader.grading import checked
        quote = "Sales forecasting predicts future demand from past purchases."
        hit = dict(id="p", text=quote, title="Sales", heading="Demand", url="", score=.9)
        response = dict(mark=1, needs_review=False, feedback="Gives one application.", passage_id="p",
                        quote=quote, independent_mark=1,
                        deductions=[dict(criterion="second", points=1,
                                         reason="This example is not listed in the course materials.")],
                        criterion_evidence=[dict(criterion=c, coverage="full", passage_id="p", quote=quote)
                                            for c in ("first", "second")])
        decision = checked(25, response, [hit], 1)
        self.assertEqual((decision["mark"], decision["reason"]), (None,"unsupported_deduction"))
        response["deductions"][0]["reason"] = "The response names a sector but gives no concrete task."
        self.assertEqual(checked(25, response, [hit], 1)["mark"], 1)

    def test_figure_and_numbers_do_not_inherit_builtin_special_cases(self):
        from exam_grader.grading import grade_exam
        exam = WrittenExam.load(DATA / "waves-exam.json")
        exam.questions[0].requires_figure = True
        payload = dict(schema_version="vision.exam.v1", exam_status="in_scope", exam_id=exam.exam_id,
                       answers=[dict(question_id=1, text="A", state="read",confidence=1,id_confidence=1)])
        row = grade_exam(payload, written_exam=exam, policy=Policy())["teacher"]["answers"][0]
        self.assertEqual((row["max_mark"],row["reason"],row["mark"]), (2,"context_check",None))
