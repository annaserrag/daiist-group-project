"""Exercise the real LangChain graph with a mocked transport; no network or key required."""
import os
import unittest
from unittest.mock import patch
try:
    from exam_grader.gemini import Gemini, Deduction, PREVIEW
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration,ChatResult
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    Gemini=None

@unittest.skipIf(Gemini is None,"optional live dependencies not installed")
class ProviderContract(unittest.TestCase):
    def test_complete_explanation_survives_the_brief_response_preference(self):
        reason = ("The response identifies a continuous outcome, but leaves out how regression "
                  "estimates that outcome from one or more predictors and how it describes the "
                  "relationship between those variables.")
        parsed = Deduction(criterion="relationship", points=1, reason=reason)
        self.assertGreater(len(reason), 160)
        self.assertEqual(parsed.reason, reason)

    def test_model_can_choose_a_full_passage_and_only_opened_text_becomes_visible(self):
        hit={"id":"p1","title":"Note","heading":"Concept","text":"x"*(PREVIEW+20)+"specific evidence beyond preview"}
        other=hit | {"id":"p2"}
        data=dict(feedback="Identifies the concept.",passage_id="p1",quote="specific evidence beyond preview",
                  needs_review=False,mark=2,deductions=[])
        def message(name,args,ident):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="",tool_calls=[
                {"id":ident,"name":name,"args":args,"type":"tool_call"}],
                usage_metadata={"input_tokens":12,"output_tokens":3,"total_tokens":15}))])
        responses=[message("open_course_passage",{"passage_id":"p1"},"a"),message("Proposal",data,"b")]
        responses.append(ChatResult(generations=[ChatGeneration(message=AIMessage(content='{"evidence":[],"independent_mark":2}',
            usage_metadata={"input_tokens":12,"output_tokens":3,"total_tokens":15}))]))
        responses[0].generations[0].message.tool_calls.append(
            {"id":"extra","name":"open_course_passage","args":{"passage_id":"p2"},"type":"tool_call"})
        with patch.dict(os.environ,{"GEMINI_API_KEY":"offline-test-only"}), patch.object(
                ChatGoogleGenerativeAI,"_generate",side_effect=responses) as transport:
            session=Gemini(finalize_lookup=True).open_session()
            self.assertEqual(session.grade("Q","key","synthetic",[hit,other]),data | {"criterion_evidence":[],"independent_mark":2})
            self.assertEqual(session.last_opened,["p1"])
            self.assertNotIn(data["quote"],str(transport.call_args_list[0].args[0]))
            self.assertIn(data["quote"],str(transport.call_args_list[1].args[0]))
            self.assertEqual([data["quote"] in h["text"] for h in session.visible_evidence([hit,other])],[True,False])
            self.assertEqual(session.usage()["calls"],3)
            self.assertIn("open_course_passage", str(transport.call_args_list[0].kwargs["tools"]))
            self.assertNotIn("open_course_passage", str(transport.call_args_list[1].kwargs["tools"]))
            self.assertIn("Proposal", str(transport.call_args_list[1].kwargs["tools"]))

    def test_custom_evidence_schema_uses_teacher_ids_for_partial_credit(self):
        from exam_grader.gemini import custom_evidence_schema
        guide = {"change":(1,"Identifies the change."), "reason":(1,"Explains its cause.")}
        schema = custom_evidence_schema(guide)
        self.assertEqual(schema["properties"]["standards"]["required"], list(guide))
        quote = "Doubling frequency halves period because period is the reciprocal of frequency."
        hit = dict(id="p", title="Waves", heading="Cycle time", text=quote)
        data = dict(feedback="Identifies the change.", passage_id="p", quote=quote, needs_review=False,
                    mark=1, deductions=[{"criterion":"reason", "points":1, "reason":"Omits the inverse relationship."}])
        entry = {"coverage":"full","passage_id":"p","quote":quote}
        import json
        response = json.dumps({"standards":{"change":entry,"reason":entry},"independent_mark":1})
        usage = {"input_tokens":50,"output_tokens":20,"total_tokens":70}
        msgs = [AIMessage(content="", tool_calls=[{"id":"p","name":"Proposal","args":data}], usage_metadata=usage),
                AIMessage(content=response, usage_metadata=usage)]
        with patch.dict(os.environ, {"GEMINI_API_KEY":"offline-test-only"}), patch.object(
                ChatGoogleGenerativeAI,"_generate",side_effect=[
                    ChatResult(generations=[ChatGeneration(message=m)]) for m in msgs]):
            session = Gemini().open_session()
            result = session.grade("What changes?", "Half", "It halves.", [hit], guide=guide)
            self.assertEqual(result["independent_mark"], 1)
            self.assertEqual([e["criterion"] for e in result["criterion_evidence"]], ["change","reason"])

    def test_zero_loss_annotation_keeps_evidence_and_independent_score_checks(self):
        from exam_grader.gemini import Proposal
        from exam_grader.grading import checked
        quote = "Structured records use a fixed schema. Semi-structured records can use tags."
        hit = dict(id="p", title="Record formats", heading="", text=quote, url="", score=.8)
        deductions = [dict(criterion="structured", points=1, reason="Confuses a table with video."),
                      dict(criterion="semi_structured", points=0, reason="Correctly identifies XML.")]
        msg = Proposal(feedback="Identifies XML.", passage_id="p", quote=quote,
                       needs_review=False, mark=1, deductions=deductions).model_dump()
        entries = [dict(criterion=c, coverage="full", passage_id="p", quote=quote)
                   for c in ("structured", "semi_structured")]
        response = dict(msg, independent_mark=1, criterion_evidence=entries)
        result = checked(26, response, [hit], 1)
        self.assertEqual(result["mark"], 1)
        self.assertEqual(result["deductions"], deductions[:1])
        self.assertNotIn("−0", result["feedback"])
        for changes in ({"independent_mark": 1.5}, {"independent_mark": None}, {"mark": 2},
                        {"criterion_evidence": entries[:1]}, {"deductions": deductions + deductions[1:]},
                        {"deductions": [deductions[0], deductions[1] | {"criterion":"unknown"}]}):
            self.assertIsNone(checked(26, response | changes, [hit], 1)["mark"])

    def test_daily_quota_is_distinct_from_a_short_rate_limit(self):
        from google.genai.errors import ClientError
        from exam_grader.gemini import Session
        for ident, expected in (("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "provider_daily_quota"),
                                ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "provider_rate_limit")):
            error = ClientError(429, {"error":{"message":"private service message",
                                "details":[{"violations":[{"quotaId":ident}]}]}})
            self.assertEqual(Session.failure_reason(error), expected)

    def test_daily_quota_blocks_further_uncached_calls_in_the_same_exam(self):
        from google.genai.errors import ClientError
        from exam_grader.gemini import DailyQuotaReached
        error = ClientError(429, {"error": {"message": "daily limit", "details": [{"violations": [
            {"quotaId": "GenerateRequestsPerDayPerProjectPerModel-FreeTier"}]}]}})
        with patch.dict(os.environ, {"GEMINI_API_KEY": "offline-test-only"}), patch.object(
                ChatGoogleGenerativeAI, "_generate", side_effect=error) as transport:
            agent = Gemini()
            session = agent.open_session()
            with self.assertRaises(Exception) as first:
                session.grade("Q1", "key", "answer", [])
            self.assertEqual(session.failure_reason(first.exception), "provider_daily_quota")
            with self.assertRaises(DailyQuotaReached):
                session.grade("Q2", "key", "answer", [])
            self.assertEqual(transport.call_count, 1)
            self.assertEqual(session.usage()["calls"], 1)
            self.assertFalse(agent.open_session().daily_quota_reached)

    def test_schema_roundtrip_and_meter(self):
        data=dict(feedback="Identifies the target.",passage_id="test",
                  quote="known labels",needs_review=False,mark=1.5,
                  deductions=[{"criterion":"definition","points":.5,"reason":"Leaves the training step unclear."}])
        msg=AIMessage(content="",tool_calls=[{"id":"call_1","name":"Proposal","args":data,"type":"tool_call"}],
                      usage_metadata={"input_tokens":60,"output_tokens":10,"total_tokens":70})
        with patch.dict(os.environ,{"GEMINI_API_KEY":"offline-test-only"}), patch.object(
            ChatGoogleGenerativeAI,"_generate",side_effect=[
                ChatResult(generations=[ChatGeneration(message=m)]) for m in
                (msg,AIMessage(content='{"evidence":[],"independent_mark":2}',usage_metadata=msg.usage_metadata))]) as transport:
            g=Gemini(input_eur_per_million=1,output_eur_per_million=2)
            s=g.open_session()
            r=s.grade("Q","teacher key","synthetic",[])
            self.assertEqual(r,data | {"criterion_evidence":[],"independent_mark":2})
            self.assertEqual(transport.call_count,2)
            self.assertEqual(s.usage()["tokens"],140)
            self.assertAlmostEqual(s.usage()["cost_eur"],.00016)
            self.assertTrue(s.usage()["usage_complete"])
