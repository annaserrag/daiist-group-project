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

    def test_daily_quota_is_distinct_from_a_short_rate_limit(self):
        from google.genai.errors import ClientError
        from exam_grader.gemini import Session
        for ident, expected in (("GenerateRequestsPerDayPerProjectPerModel-FreeTier", "provider_daily_quota"),
                                ("GenerateRequestsPerMinutePerProjectPerModel-FreeTier", "provider_rate_limit")):
            error = ClientError(429, {"error":{"message":"private service message",
                                "details":[{"violations":[{"quotaId":ident}]}]}})
            self.assertEqual(Session.failure_reason(error), expected)

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
