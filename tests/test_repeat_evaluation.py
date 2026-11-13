"""Regression checks for paired baselines, paid-call reuse and reporting."""
import json
import os
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_google_genai import ChatGoogleGenerativeAI

from exam_grader.costs import projection
from exam_grader.evaluation_cache import EvaluationCache, MissingFrozenResponse
from exam_grader.gemini import Gemini
from exam_grader.reference_marks import fit_written_baseline, compare_written
from exam_grader.retrieval import Index
from evaluate import evaluate, rows
import pipeline


class RepeatEvaluation(unittest.TestCase):
    def test_same_source_cannot_train_and_test_and_missing_marks_stay_missing(self):
        source = {"train":{q:"4" if q==21 else "0.5" for q in range(21,36)}}
        with self.assertRaisesRegex(ValueError,"both baseline training and evaluation"):
            fit_written_baseline(source,["train"],["train"])
        baseline=fit_written_baseline(source,["train"],["heldout"])
        self.assertIsNone(baseline[21])
        self.assertEqual(baseline[22],.5)
        pairs=[{"question_id":q,"baseline_mark":base,"teacher_mark":gold,"proposed_mark":grade}
               for q,base,gold,grade in [(21,None,"1","1"),(22,.5,"1","1"),(23,0,"2",""),(24,2,"","2")]]
        report=compare_written(pairs)
        self.assertEqual(report["baseline_on_all_scorable"]["n"],2)
        self.assertEqual(report["baseline_on_matched_answers"]["mean_absolute_error"],.5)
        self.assertEqual(report["grader_on_matched_answers"]["mean_absolute_error"],0)
        self.assertEqual(report["sent_to_review_with_valid_reference"],1)

    def test_cache_is_transactional_for_concurrent_runs_and_retries(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache=EvaluationCache(Path(tmp)/"private.sqlite")
            calls=[]
            def generate():
                calls.append(1)
                return {"mark":1},(12,3)
            with ThreadPoolExecutor(max_workers=2) as workers:
                results=list(workers.map(lambda _:cache.get_or_generate("v1","same",generate),range(2)))
            self.assertEqual(len(calls),1)
            self.assertEqual(sorted(r[2] for r in results),[False,True])
            with self.assertRaises(RuntimeError):
                cache.get_or_generate("v2","same",lambda:(_ for _ in ()).throw(RuntimeError("transport")))
            self.assertFalse(cache.get_or_generate("v2","same",generate)[2])
            self.assertFalse(cache.get_or_generate("v1","changed",generate)[2])
            self.assertFalse(cache.get_or_generate("v3","same",lambda:({"mark":1},None))[2])
            self.assertFalse(cache.get_or_generate("v3","same",generate)[2])
            frozen = EvaluationCache(cache.path, read_only=True)
            self.assertEqual(frozen.get_or_generate("v1","same",self.fail), ({"mark":1}, (12,3), True))
            with self.assertRaises(MissingFrozenResponse):
                frozen.get_or_generate("absent","same",self.fail)
            with patch.dict(os.environ, {"GEMINI_API_KEY":""}):
                self.assertIsNone(Gemini(frozen_cache=frozen).app)

    def test_two_runs_through_pipeline_bill_once_and_compare_the_same_answers(self):
        idx=Index()
        transcript=json.loads(pipeline.EXAMPLE.read_bytes())["answers"][0]["text"]
        hits=idx.search(21,transcript)
        if hits[0]["score"] < .4:
            hits=idx.search(21)
        best=hits[0]
        data=dict(feedback="Identifies known labels.",passage_id=best["id"],quote=best["text"][:40],
                  needs_review=False,mark=2,deductions=[])
        msg=AIMessage(content="",tool_calls=[{"id":"one","name":"Proposal","args":data,"type":"tool_call"}],
                      usage_metadata={"input_tokens":60,"output_tokens":10,"total_tokens":70})
        cases=[{"id":"demo","kind":"hard","input":str(pipeline.EXAMPLE),"expected":'{"marks":{"21":2}}'}]
        source={name:{q:("A" if q<=20 else value) for q in range(1,36)}
                for name,value in (("training","0"),("heldout","2"))}
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ,{"GEMINI_API_KEY":"offline-fixture"}), patch.object(
                ChatGoogleGenerativeAI,"_generate",side_effect=[
                    ChatResult(generations=[ChatGeneration(message=m)]) for m in
                    (msg, AIMessage(content=json.dumps({"independent_mark":2,"evidence":[{"criterion":"definition","coverage":"full",
                        "passage_id":best["id"],"quote":data["quote"]}]}),usage_metadata=msg.usage_metadata))]) as transport:
            root=Path(tmp)
            agent=Gemini(input_eur_per_million=1,output_eur_per_million=2)
            from exam_grader.integration import file_sha256
            approved={file_sha256(pipeline.EXAMPLE)}
            with patch.object(pipeline,"resources",return_value=(idx,agent,approved)):
                reports=[evaluate(cases,root/run,links={"demo":"heldout"},reference=source,
                                  training_ids=["training"],evaluation_cache=EvaluationCache(root/"cache.sqlite"))
                         for run in ("first","second")]
            self.assertEqual(transport.call_count,2)
            self.assertEqual([r["cache_hits"] for r in reports],[0,1])
            self.assertEqual(reports[0]["held_out_written_comparison"],reports[1]["held_out_written_comparison"])
            first,second=[rows(root/run/"results.csv") for run in ("first","second")]
            self.assertEqual(second[0]["uncached_cost_eur"],first[0]["cost_eur"])
            self.assertEqual(float(second[0]["cost_eur"]),0)
            self.assertEqual((int(second[0]["tokens"]),int(second[0]["uncached_tokens"])),(0,140))
            self.assertNotIn(transcript.encode(),(root/"cache.sqlite").read_bytes())
            costs=dict(hourly_eur=20,signoff_seconds=5,review_seconds=20,manual_seconds=30,wrong_eur=1,
                       accepted_catch_rate=0,reviewed_error_rate=0,manual_error_rate=0,exams_per_day=1)
            _,result=projection(second,rows(root/"second"/"paired_marks.csv"),**costs)
            self.assertGreater(result["api_eur_per_1000"],0)
            with self.assertRaisesRegex(ValueError,"different evaluation runs"):
                projection(first,rows(root/"second"/"paired_marks.csv"),**costs)
            self.assertEqual(json.loads((root/"second"/"summary.json").read_text())["model_calls"],0)
