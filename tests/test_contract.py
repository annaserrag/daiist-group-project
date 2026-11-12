"""Engineering fixtures only: not case labels or product evaluation."""
import json
import unittest
import tempfile
from pathlib import Path
from exam_grader.catalog import DATA,KEY
from exam_grader.grading import grade_exam,Policy
from exam_grader.integration import connect,file_sha256
from exam_grader.retrieval import Index
from exam_grader.catalog import QUESTIONS
from exam_grader.metrics import grading

def a(q,txt,**changes):
    return dict(question_id=q,text=txt,confidence=.99,id_confidence=.99,state="read") | changes
def envelope(*answers):
    return dict(schema_version="vision.exam.v1",exam_status="in_scope",answers=list(answers))
def result(*answers,**kw):
    return grade_exam(envelope(*answers),**kw)
def get(out,q):
    return out["teacher"]["answers"][q-1]
class Search:
    version="fixture"
    def search(self,q,answer="",k=3):
        return [dict(id="demo",title="Synthetic note",url="https://example.org",heading="Demo",
                     text="Examples carry known target labels. Group values into bins. Ages can be grouped by decade.",score=.2 if answer else .9)]
class Fake:
    version="test-double"
    def __init__(self,**overrides):
        self.overrides=overrides
        self.calls=0
    def open_session(self):
        return self
    def grade(self,question,*args):
        self.calls+=1
        quotes = {"method":"Group values into bins.", "example":"Ages can be grouped by decade."} if question == QUESTIONS[30] else {"definition":"known target labels"}
        return dict(mark=1.5,passage_id="demo",quote="known target labels",feedback="Identifies labels.",needs_review=False,
                    criterion_evidence=[dict(criterion=c, quote=t, coverage="full", passage_id="demo") for c,t in quotes.items()],
                    deductions=[{"criterion":"definition","points":.5,"reason":"Does not say the examples are used to train the model."}]) | self.overrides
    def usage(self):
        return dict(tokens=self.calls*10,cost_eur=self.calls*.001,calls=self.calls,usage_complete=True)

class Contract(unittest.TestCase):
    def test_q32_marks_count_distinct_examples_and_explain_every_deduction(self):
        cases = [
            ("Random Forest", 0, 0),
            ("learning_rate=0.01, batch size", 1.5, 2),
            ("K in K-NN, learning rate, batch size, epochs", 2, 4),
            (KEY[32], 2, 5),
        ]
        for text, mark, count in cases:
            with self.subTest(text=text):
                item = get(result(a(32,text)),32)
                self.assertEqual(item["mark"],mark)
                self.assertEqual(len(item["credited_examples"]),count)
                self.assertEqual(sum(d["points"] for d in item["deductions"]),2-mark)
                for d in item["deductions"]:
                    self.assertIn(d["reason"],item["feedback"])
        self.assertEqual(get(result(a(32,"not learning rate")),32)["reason"],"uncertain_hyperparameter")
        self.assertEqual(get(result(a(32,"batch size",confidence=.2)),32)["reason"],"low_read_confidence")
        self.assertEqual(get(result(a(32,"batch size",id_confidence=.2)),32)["reason"],"uncertain_question_id")

    def test_q32_svm_only_zero_requires_the_whole_answer(self):
        for text in ("SVM", "support-vector machine", "SVM (support vector machine)",
                     "The name few hyperparameters of machine learning algorithm is Suport Vector machine (SVM)."):
            with self.subTest(text=text):
                row = get(result(a(32, text)), 32)
                self.assertEqual((row["mark"], row["credited_examples"]), (0, []))
                self.assertEqual(sum(d["points"] for d in row["deductions"]), 2)
                self.assertIn("tunable settings", row["feedback"])
        mixed = get(result(a(32, "SVM, C")), 32)
        self.assertIsNone(mixed["mark"])
        self.assertEqual(mixed["credited_examples"], ["SVM penalty C"])
        self.assertIsNone(get(result(a(32, "SVM, logistic regression")), 32)["mark"])

    def test_q32_reviews_possible_duplicate_only_when_it_affects_credit(self):
        for text in ("max-leaf, max-node, K", "K; max_nodes; maximum number of leaves"):
            with self.subTest(text=text):
                row = get(result(a(32, text)), 32)
                self.assertEqual((row["status"], row["reason"], row["mark"]),
                                 ("review", "ambiguous_duplicate", None))
                self.assertCountEqual(row["credited_examples"], ["maximum leaf nodes", "neighbor count"])
                self.assertEqual(row["deductions"], [])
                self.assertIn("limit on leaf nodes", row["feedback"])
                self.assertIn("limit on all nodes", row["feedback"])
        clear = get(result(a(32, "max leaf, max terminal nodes, K")), 32)
        self.assertEqual(clear["mark"], 1.5)
        enough = get(result(a(32, "max leaf, max node, K, batch size")), 32)
        self.assertEqual(enough["mark"], 2)

    def test_q32_one_setting_preserves_credit_and_defers_the_mark(self):
        for text, setting, note in [
            ("epoch", "training epochs", ""),
            ("max_depth, max depth tree", "maximum tree depth", "counts once"),
            ("n estimater, rmse, r2 score", "number of estimators", "RMSE measures"),
            ("learning rate, some unfamiliar setting", "learning rate", "remaining wording"),
        ]:
            with self.subTest(text=text):
                row = get(result(a(32, text)), 32)
                self.assertEqual((row["status"], row["reason"], row["mark"]),
                                 ("review", "single_hyperparameter_review", None))
                self.assertEqual(row["credited_examples"], [setting])
                self.assertEqual(row["deductions"], [])
                self.assertIn(setting, row["feedback"])
                self.assertIn("teacher", row["feedback"])
                if note:
                    self.assertIn(note, row["feedback"])

    def test_q32_cluster_explanations_and_epoch_aliases(self):
        for phrase in ("In K-means, K controls the number of clusters",
                       "K determines how many clusters K-means forms",
                       "K value is a hyperparameters of k-mean algorithm that defined the number of cluster",
                       "K - number of clusters"):
            with self.subTest(phrase=phrase):
                row = get(result(a(32, f"epoch; epochs; {phrase}; number of clusters")), 32)
                self.assertEqual(row["credited_examples"], ["training epochs", "number of clusters"])
                self.assertEqual(row["mark"], 1.5)
        for phrase in ("K-means clustering", "K in K-means does not set the number of clusters",
                       "In K-means, K sets the number of neighbors", "not epoch"):
            with self.subTest(phrase=phrase):
                row = get(result(a(32, phrase)), 32)
                self.assertIsNone(row["mark"])
                self.assertEqual(row["credited_examples"], [])
        row = get(result(a(32, "K in K-mean; another unclear setting")), 32)
        self.assertIsNone(row["mark"])
        self.assertEqual(row["credited_examples"], ["number of clusters"])

    def test_real_key_order_and_tf_options(self):
        out=result(a(17,"D"),a(2,"c"),a(1,"D"))
        self.assertEqual([get(out,q)["mark"] for q in [1,2,17]],[0,1,None])
        self.assertEqual(get(out,17)["reason"],"invalid_choice")
    def test_missing_blank_and_read_failure(self):
        out=result(a(1,""),a(2,"",state="confirmed_blank"),a(3,"",state="unreadable"))
        self.assertEqual([get(out,q)["mark"] for q in range(1,5)],[None,0,None,None])
        self.assertIsNone(out["teacher"]["proposed_total"])
    def test_contract_mismatch(self):
        for p in [None,[],{"schema_version":"vision.v0","answers":[]},envelope()|{"answers":None}]:
            self.assertEqual(get(grade_exam(p),1)["reason"],"invalid_contract")
    def test_id_is_separate_and_confidences_are_validated(self):
        out=result(a(1,"B",id_confidence=.2),a(2,"C",confidence=float("nan")),a(3,"A"),a(3,"B"))
        self.assertEqual([get(out,q)["reason"] for q in [1,2,3]],["uncertain_question_id","invalid_answer","duplicate_detection"])
        json.dumps(out,allow_nan=False)
    def test_unassigned_or_wrong_exam_blocks(self):
        agent=Fake()
        for p in [envelope(a(None,"B"),a(21,"labels")),envelope(a(21,"labels"))|{"exam_status":"out_of_scope"}]:
            out=grade_exam(p,index=Search(),agent=agent,allow_remote=True)
            self.assertEqual(out["teacher"]["review_count"],35)
        self.assertEqual(agent.calls,0)
    def test_retry_citation_and_logging(self):
        out=result(a(21,"CANARY_STUDENT_TEXT"),index=Search(),agent=Fake(),allow_remote=True)
        self.assertEqual((get(out,21)["searches"],get(out,21)["mark"]),(2,1.5))
        self.assertEqual(out["record"]["tokens"],10)
        log=json.dumps(out["record"])
        self.assertNotIn("CANARY",log)
        self.assertNotIn("Identifies labels.",log)
        self.assertNotIn("known target labels",log)
    def test_ungrounded_or_invalid_model_output(self):
        for change in [dict(quote="not in source"),dict(passage_id="invented"),dict(mark=.25),dict(mark=True),dict(needs_review=True)]:
            out=result(a(21,"labels"),index=Search(),agent=Fake(**change),allow_remote=True)
            self.assertIsNone(get(out,21)["mark"])
            self.assertEqual(out["record"]["tokens"],10)
    def test_optional_written_cutoff_keeps_reviewed_marks_out_of_totals(self):
        out = result(a(21,"labels"), a(1,KEY[1]), a(23,"",state="confirmed_blank"),
                     index=Search(), agent=Fake(), allow_remote=True, policy=Policy(written_review_at_or_below=1.5))
        proposed = get(out,21)
        self.assertEqual((proposed["reason"],proposed["mark"],proposed["candidate_mark"]),
                         ("written_mark_review",None,1.5))
        self.assertTrue(proposed["citation"] and proposed["deductions"])
        self.assertEqual((get(out,1)["mark"],get(out,23)["mark"],out["teacher"]["proposed_subtotal"]),(1,0,1))
        with self.assertRaises(ValueError):
            Policy(written_review_at_or_below=True)
    def test_low_read_confidence_stays_review_after_strong_retrieval(self):
        agent=Fake()
        out=result(a(21,"labels",confidence=.3),index=Search(),agent=agent,allow_remote=True)
        self.assertEqual((get(out,21)["reason"],get(out,21)["searches"],agent.calls),("low_read_confidence",2,0))

    def test_partial_credit_accounts_for_each_loss_once_in_teacher_feedback(self):
        good = [{"criterion":"method","points":.5,"reason":"Names grouping without explaining the intervals."},
                {"criterion":"example","points":1,"reason":"Omits an example of a numeric variable grouped into bins."}]
        def decision(mark, deductions):
            agent = Fake(mark=mark, deductions=deductions)
            return get(result(a(30,"Group them."),index=Search(),agent=agent,allow_remote=True),30)
        graded = decision(.5, good)
        self.assertEqual(graded["mark"], .5)
        for item in good:
            self.assertIn(item["reason"], graded["feedback"])
        for invalid in ([], good[:1], [good[0],good[0]], [dict(good[0],criterion="other")],
                        [dict(good[0],points=1.5)], [dict(good[0],reason=" ")]):
            self.assertEqual(decision(.5,invalid)["reason"], "unexplained_deduction")
        self.assertEqual(decision(2,[])["mark"],2)
        self.assertEqual(decision(2,good)["reason"],"unexplained_deduction")
    def test_context_gate_and_disabled_provider(self):
        out=result(a(34,"margin"),a(21,"labels"),index=Search(),agent=Fake())
        self.assertEqual([get(out,q)["reason"] for q in (34,21)],["context_check","provider_disabled"])
    def test_all_35_and_pass_band(self):
        aa=[a(q,KEY[q]) if q<=20 else a(q,"",state="confirmed_blank") for q in KEY]
        out=result(*reversed(aa),policy=Policy(pass_mark=20,pass_band=.5))
        self.assertEqual(out["teacher"]["proposed_total"],20)
        self.assertTrue(out["record"]["review"])
        self.assertTrue(out["teacher"]["teacher_signoff_required"])
    def test_integrate_by_file_digest_not_payload_claim(self):
        with tempfile.TemporaryDirectory() as d:
            f=Path(d)/"fixture"; f.write_bytes(b"synthetic")
            allow={file_sha256(f)}
            reader=lambda path:envelope(a(21,"known labels"))|{"approved":True,"features":{"line_count":2,"unexpected":"DO NOT LOG"}}
            agent=Fake()
            good=connect(f,read_exam=reader,vision_version="vision-v1",index=Search(),agent=agent,approved_hashes=allow)
            self.assertEqual(get(good,21)["mark"],1.5)
            f.write_bytes(b"unapproved")
            bad=connect(f,read_exam=reader,vision_version="vision-v1",index=Search(),agent=agent,approved_hashes=allow)
            self.assertEqual((agent.calls,get(bad,21)["reason"]),(1,"provider_disabled"))
            self.assertEqual(bad["record"]["features"]["line_count"],2)
            self.assertNotIn("DO NOT LOG",json.dumps(bad))
    def test_q29_returns_the_distance_explanation_in_the_grader_window(self):
        for design in ("paragraph","section"):
            index = Index(design)
            for answer in ("", "Different feature ranges affect distances between samples."):
                hits = index.search(29,answer,k=3)
                self.assertTrue(any(hit["source"]=="scales" and
                                    all(term in hit["text"] for term in ("proline","hue","distances between samples"))
                                    for hit in hits), (design, answer))

    def test_semantic_outlier_question_covers_detection_and_treatment(self):
        for design in ("section","paragraph"):
            original = Index(design, True, multipart=False)
            revised = Index(design, True)
            for text in ("", KEY[27]):
                before = original.search(27,text)
                after = revised.search(27,text)
                self.assertEqual(before[:2],after[:2])
                self.assertTrue(any("detect" in r["text"].lower() for r in after[:2]))
                self.assertTrue(any("robust to outliers" in r["text"] or "Clipping is a technique" in r["text"] for r in after))
            for q in set(range(21,36)) - {27}:
                self.assertEqual(original.search(q),revised.search(q))

    def test_top_three_explain_both_fitting_failures_and_illustrate_discretization(self):
        for design in ("paragraph","section"):
            index = Index(design)
            for q in (24,30):
                for answer in ("", KEY[q]):
                    with self.subTest(design=design, question=q, with_answer=bool(answer)):
                        passages = [hit["text"].lower() for hit in index.search(q,answer,k=3)]
                        if q == 24:
                            self.assertTrue(any(all(s in text for s in ("overfit model","poor predictions on new data",
                                                                       "underfit model","training data")) for text in passages))
                        else:
                            self.assertTrue(any("group numerical data into categories" in text for text in passages))
                            self.assertTrue(any("10,000 dollars" in text and "bucket" in text for text in passages))

    def test_chunk_designs_return_distinct_citable_passages(self):
        for design in ("section","paragraph"):
            i=Index(design)
            hits=i.search(21,k=5)
            self.assertEqual(len(hits),5)
            self.assertEqual(len(set(r["id"] for r in hits)),5)
            self.assertTrue(all(r["url"].startswith("https://") for r in hits))
    def test_metrics_exclude_missing_and_impossible_labels(self):
        base=dict(exam_id="fixture",proposed_mark="1",teacher_mark="1")
        data=[base|dict(question_id=q,teacher_mark=g,proposed_mark=p) for q,g,p in [(21,"1","1"),(22,"4","1"),(23,"","0"),(24,"1",""),(25,"2","1.5")]]
        m=grading(data)
        self.assertEqual((m["reviewed"],m["scored_proposals"],m["missing_or_invalid_teacher_marks"]),(1,2,2))
        self.assertEqual((m["within_half_on_scored"],m["exact_agreement_on_scored"]),(1,.5))
