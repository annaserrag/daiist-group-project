"""Exercise the real LangChain graph with a mocked transport; no network or key required."""
import os
import unittest
from unittest.mock import patch
try:
    from exam_grader.gemini import Gemini, Deduction
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
        hit={"id":"p1","title":"Note","heading":"Concept","text":"x"*280+"specific evidence beyond preview"}
        other=hit | {"id":"p2"}
        data=dict(feedback="Identifies the concept.",passage_id="p1",quote="specific evidence beyond preview",
                  needs_review=False,mark=2,deductions=[])
        def message(name,args,ident):
            return ChatResult(generations=[ChatGeneration(message=AIMessage(content="",tool_calls=[
                {"id":ident,"name":name,"args":args,"type":"tool_call"}],
                usage_metadata={"input_tokens":12,"output_tokens":3,"total_tokens":15}))])
        responses=[message("open_course_passage",{"passage_id":"p1"},"a"),message("Proposal",data,"b")]
        with patch.dict(os.environ,{"GEMINI_API_KEY":"offline-test-only"}), patch.object(
                ChatGoogleGenerativeAI,"_generate",side_effect=responses) as transport:
            session=Gemini().open_session()
            self.assertEqual(session.grade("Q","key","synthetic",[hit,other]),data)
            self.assertEqual(session.last_opened,["p1"])
            self.assertNotIn(data["quote"],str(transport.call_args_list[0].args[0]))
            self.assertIn(data["quote"],str(transport.call_args_list[1].args[0]))
            self.assertEqual([data["quote"] in h["text"] for h in session.visible_evidence([hit,other])],[True,False])
            self.assertEqual(session.usage()["calls"],2)

    def test_schema_roundtrip_and_meter(self):
        data=dict(feedback="Identifies the target.",passage_id="test",
                  quote="known labels",needs_review=False,mark=1.5,
                  deductions=[{"criterion":"definition","points":.5,"reason":"Leaves the training step unclear."}])
        msg=AIMessage(content="",tool_calls=[{"id":"call_1","name":"Proposal","args":data,"type":"tool_call"}],
                      usage_metadata={"input_tokens":60,"output_tokens":10,"total_tokens":70})
        with patch.dict(os.environ,{"GEMINI_API_KEY":"offline-test-only"}), patch.object(
            ChatGoogleGenerativeAI,"_generate",return_value=ChatResult(generations=[ChatGeneration(message=msg)])) as transport:
            g=Gemini(input_eur_per_million=1,output_eur_per_million=2)
            s=g.open_session()
            r=s.grade("Q","teacher key","synthetic",[])
            self.assertEqual(r,data)
            self.assertEqual(transport.call_count,1)
            self.assertEqual(s.usage()["tokens"],70)
            self.assertAlmostEqual(s.usage()["cost_eur"],.00008)
            self.assertTrue(s.usage()["usage_complete"])
