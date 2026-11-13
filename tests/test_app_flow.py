"""Test the teacher workflow and evaluation boundary through the same inputs as the UI."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import app
import gradio as gr
import logbook
import pipeline
from evaluate import evaluate
from evaluate import match_expected
from exam_grader.judge import compare_labels, prepare_items
from exam_grader.reference_marks import audit

class AppFlow(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from exam_grader.retrieval import Index
        offline = patch.object(pipeline,"resources",return_value=(Index(),None,frozenset()))
        offline.start()
        cls.addClassCleanup(offline.stop)

    def test_grade_correct_sign_and_log_feedback(self):
        with tempfile.TemporaryDirectory() as directory, patch.object(logbook,"LOG_DIR",Path(directory)), patch.object(logbook,"ARCHIVE_DIR",Path(directory)/"archive"):
            status,evidence,marks,context,signed = app.display(pipeline.EXAMPLE)
            self.assertIn("33 require review",status)
            self.assertTrue(any("known target labels" in str(row) for row in evidence))
            self.assertIsNone(signed)
            with self.assertRaises(gr.Error):
                app.sign_off(context,marks)
            corrected = [[q, 1 if q==1 else .5 if q==21 else 0] for q,_ in marks]
            signed = app.sign_off(context,corrected)
            self.assertEqual(signed["total"],1.5)
            self.assertEqual(signed["review_session"]["initially_referred"],33)
            self.assertEqual(signed["review_session"]["accepted_proposals"],2)
            self.assertEqual(signed["review_session"]["changed_proposals"],0)
            self.assertGreaterEqual(signed["review_session"]["elapsed_seconds_since_proposals_shown"],0)
            app.rate(context,1)
            self.assertEqual(logbook.load().iloc[0]["feedback"],1)
            self.assertNotIn("known target labels",next(Path(directory).glob("requests-*")).read_text())
            self.assertEqual(app.monitoring()[2].iloc[0]["requests"],1)
    def test_focus_review_preserves_the_sheet_and_uses_the_question_scale(self):
        context = {"limits": [(2, 1), (21, 2), (22, 2)],
                   "referrals": {2: {"reading": ""}, 21: {"reading": "student"},
                                 22: {"reading": "student"}}}
        sheet = [[2, None], [21, None], [22, None]]
        updated, next_question, _, _ = app.apply_review_mark(context, 22, 0, sheet)
        self.assertEqual(updated, [[2, None], [21, None], [22, 0]])
        self.assertEqual(next_question, 21)
        updated, next_question, _, _ = app.apply_review_mark(context, 21, 1.5, updated)
        self.assertEqual(next_question, 2)
        with self.assertRaises(gr.Error):
            app.apply_review_mark(context, 2, .5, updated)
        self.assertIsNone(app.apply_review_mark(context, 2, 1, updated)[1])
        with self.assertRaises(gr.Error):
            app.apply_review_mark(context, 21, 2, updated[::-1])

    def test_wrong_subject_cannot_be_signed(self):
        with self.assertRaises(gr.Error):
            app.sign_off({"id":"test","in_scope":False},[[q,0] for q in range(1,36)])

    def test_unreadable_whole_exam_cannot_pass_question_level_expectations(self):
        from exam_grader.grading import grade_exam
        payload=json.loads(pipeline.EXAMPLE.read_bytes())
        good=grade_exam(payload)["teacher"]
        bad=grade_exam(dict(payload,exam_status="unreadable"))["teacher"]
        review='{"review_questions":[17]}'
        mixed='{"marks":{"1":1},"review_reasons":{"17":"invalid_choice"}}'
        for expected in (review,mixed,'{"total":null}'):
            self.assertFalse(match_expected(expected,bad))
        self.assertTrue(match_expected(mixed,good))
        self.assertTrue(match_expected('{"exam_status":"unreadable"}',bad))
        for field,value in (("reader_failed",True),("input_type","pdf_awaiting_reader")):
            self.assertFalse(match_expected('{"exam_status":"unreadable"}',bad|{field:value}))
        self.assertFalse(match_expected(review,grade_exam(dict(payload,answers=[]))["teacher"]))
        self.assertFalse(match_expected('{"review_questions":[21]}',good))
        self.assertIsNone(match_expected("Assess manually",bad))

    def test_frozen_cache_miss_cannot_satisfy_a_targeted_review(self):
        from evaluate import rows
        from exam_grader.evaluation_cache import EvaluationCache
        from exam_grader.grading import Policy
        from exam_grader.retrieval import Index
        resources = (Index(), None, {pipeline.file_sha256(pipeline.EXAMPLE)})
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline,"resources",return_value=resources):
            root = Path(folder)
            cache_path = root/"empty.sqlite"
            EvaluationCache(cache_path)
            frozen = EvaluationCache(cache_path,read_only=True)
            cases = [dict(id=f"q{q}",kind="review",input=str(pipeline.EXAMPLE),
                          expected=json.dumps({"review_questions":[q]})) for q in (17,21)]
            result = evaluate(cases,root,policy=Policy(min_similarity=0),evaluation_cache=frozen)
            self.assertEqual(result["agreement_with_structured_expectations"],0.5)
            self.assertEqual(result["cases_with_execution_issues"],2)
            checks = rows(root/"results.csv")
            self.assertEqual([r["correct"] for r in checks],["True","False"])
            self.assertEqual([r["execution_issues"] for r in checks],["missing_frozen_response"]*2)

    def test_batch_distinguishes_unreadable_input_from_broken_reader(self):
        from evaluate import rows
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder)
            document=root/"real-status.json"
            document.write_text(json.dumps({"schema_version":"vision.exam.v1","exam_status":"unreadable","answers":[]}))
            broken=root/"broken.json"
            broken.write_text("not a JSON export")
            cases=[dict(id=name,kind="review",input=str(path),expected=expect) for name,path,expect in
                   [("ordinary",document,'{"review_questions":[17]}'),("abstain",document,'{"exam_status":"unreadable"}'),
                    ("reader",broken,'{"exam_status":"unreadable"}')]]
            summary=evaluate(cases,root)
            self.assertAlmostEqual(summary["agreement_with_structured_expectations"],1/3)
            self.assertEqual(summary["cases_with_execution_issues"],1)
            self.assertEqual(rows(root/"results.csv")[-1]["execution_issues"],"reader_failure")

    def test_shared_schema_and_reader_can_be_used_from_pdf_entry(self):
        import sys
        from exam_grader.handoff import inspect_export, VisionExam
        fixture = json.loads(pipeline.EXAMPLE.read_bytes())
        schema = pipeline.EXAMPLE.with_name("vision.exam.v1.schema.json")
        self.assertEqual(json.loads(schema.read_bytes()), VisionExam.model_json_schema())
        self.assertTrue(inspect_export(fixture)["valid"])
        fixture["answers"][0]["id_confidence"] = "private-field-value"
        report = inspect_export(fixture)
        self.assertFalse(report["valid"])
        self.assertIn("answers.0.id_confidence", str(report))
        self.assertNotIn("private-field-value", str(report))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            reader = path/"fixture_team_vision.py"
            reader.write_text("from pathlib import Path\nimport json\nVERSION='fixture-reader-v1'\n"
                              "def read_exam(path):\n    assert isinstance(path,Path) and path.suffix=='.pdf'\n"
                              "    return json.loads(path.with_suffix('.json').read_bytes())\n")
            pdf = path/"sheet.pdf"
            pdf.write_bytes(b"%PDF-1.4\n% routing fixture\n%%EOF")
            pdf.with_suffix(".json").write_bytes(pipeline.EXAMPLE.read_bytes())
            try:
                with patch.object(sys,"path",[folder,*sys.path]), patch.dict("os.environ",{"GRADE_VISION_MODULE":"fixture_team_vision"}):
                    output = pipeline.run_views(pdf)
            finally:
                sys.modules.pop("fixture_team_vision",None)
            self.assertEqual(output["teacher"]["input_type"],"pdf")
            self.assertEqual(output["record"]["versions"]["component_a"],"fixture-reader-v1")
            self.assertEqual(output["teacher"]["answers"][0]["mark"],1)
    def test_batch_results_keep_unassessed_cases_distinct(self):
        with tempfile.TemporaryDirectory() as directory:
            cases=[dict(id="synthetic",kind="hard",input=str(pipeline.EXAMPLE),
                        expected=json.dumps({"marks":{"1":1,"23":0},"review_reasons":{"17":"invalid_choice"}})),
                   dict(id="human",kind="review",input=str(pipeline.EXAMPLE),expected="A teacher assesses this description.")]
            result=evaluate(cases,Path(directory),teacher_dir=Path(directory)/"private")
            self.assertEqual(result["automatically_assessed"],1)
            self.assertEqual(result["agreement_with_structured_expectations"],1)
            self.assertEqual(result["awaiting_human_assessment"],1)
            self.assertNotIn("known target labels",(Path(directory)/"results.csv").read_text())
            self.assertIn("known target labels",(Path(directory)/"private"/"synthetic.json").read_text())
    def test_judge_links_and_blinded_inputs(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict("os.environ",{"GEMINI_API_KEY":""}):
            from exam_grader.retrieval import Index
            from exam_grader.integration import file_sha256
            idx=Index()
            h=idx.search(21,k=1)[0]
            transcript=json.loads(pipeline.EXAMPLE.read_bytes())["answers"][0]["text"]
            teacher={"answers":[{"question_id":21,"mark":1,"reason":"grounded_proposal","readings":[transcript],
                     "feedback":"Recognizes labels.","citation":{"id":h["id"],"quote":h["text"][:30]}}]}
            path=Path(directory)/"view.json"
            path.write_text(json.dumps(teacher))
            row={"id":"f1","question_id":"21","teacher_json":path,"ocr_json":pipeline.EXAMPLE,"label":"fail","comment":"Human opinion"}
            approved={file_sha256(pipeline.EXAMPLE)}
            item=prepare_items([row],approved,idx)[0][1]
            self.assertNotIn("Human opinion",str(item))
            with self.assertRaises(ValueError):
                prepare_items([row],set(),idx)
            self.assertEqual(item["quote"],h["text"][:30])
            teacher["answers"][0]["criterion_evidence"] = {"definition":{"id":h["id"],"quote":h["text"][2:32]}}
            path.write_text(json.dumps(teacher))
            revised = prepare_items([row],approved,idx)[0][1]
            self.assertEqual(revised["criterion_quotes"],{"definition":h["text"][2:32]})
            teacher["answers"][0]["criterion_evidence"]["definition"]["quote"] = "Missing source text."
            path.write_text(json.dumps(teacher))
            with self.assertRaises(ValueError):
                prepare_items([row],approved,idx)
            labels=[{"id":f"f{i}","label":"pass" if i%2 else "fail"} for i in range(20)]
            stats=compare_labels({r["id"]:"pass" for r in labels},labels)
            self.assertEqual(stats["exact_agreement"],.5)
    def test_baseline_removes_held_out_exam_from_its_question_mean(self):
        exams={"a":{q:("A" if q<=20 else "0") for q in range(1,36)},
               "b":{q:("A" if q<=20 else "2") for q in range(1,36)}}
        b=audit(exams)["written_baseline"]
        self.assertEqual(b["mae_unrounded"],2)
        self.assertEqual(b["exact_after_rounding"],0)

    def test_cost_projection_keeps_unknown_costs_and_missing_labels_visible(self):
        from exam_grader.costs import projection
        results=[{"id":"one","policy":"fixture","cost_eur":""}]
        pairs=[{"exam_id":"one","question_id":q,"teacher_mark":"1" if q != 2 else "",
                "proposed_mark":"0" if q == 1 else "1" if q == 2 else ""} for q in range(1,36)]
        _,report=projection(results,pairs,hourly_eur=30,signoff_seconds=5,review_seconds=20,
                            manual_seconds=30,wrong_eur=10,accepted_catch_rate=0,
                            reviewed_error_rate=0,manual_error_rate=0,exams_per_day=10)
        self.assertEqual(report["unflagged_disagreements"],1)
        self.assertEqual(report["proposals_without_reference"],1)
        self.assertAlmostEqual(report["extra_reviews_per_day"],330)
        self.assertIsNone(report["estimated_total_eur_per_1000"])

    def test_drift_rule_on_controlled_engineering_inputs(self):
        import pandas as pd
        import monitor
        base=pd.DataFrame({"mean_read_confidence":[.9,.94,.95,.98]*10})
        self.assertFalse(monitor.drift(base,base)["alert"].any())
        shifted=pd.DataFrame({"mean_read_confidence":[.1,.2,.3,.4]*10})
        self.assertTrue(monitor.drift(base,shifted)["alert"].any())
        only_latency=base.assign(vision_latency_s=0)
        self.assertFalse(monitor.drift(only_latency,only_latency.assign(vision_latency_s=600))["alert"].any())
        self.assertIn("not assessed",monitor.drift_status(pd.DataFrame(),base))
