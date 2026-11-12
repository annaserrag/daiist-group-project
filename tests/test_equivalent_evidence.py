import unittest

from exam_grader.grading import checked
from exam_grader.rubric import check_interval_example


class EquivalentEvidence(unittest.TestCase):
    def test_numeric_examples_use_the_rule_and_verify_membership(self):
        rule = "Binning groups numerical measurements into distinct intervals."
        hit = dict(id="b", title="Binning", heading="Notes", url="", text=rule, score=.8)
        entry = dict(criterion="method", coverage="full", passage_id="b", quote=rule)
        proposal = dict(mark=2, feedback="Defines binning and illustrates it.", deductions=[],
                        needs_review=False, passage_id="b", quote=rule, criterion_evidence=[entry])
        student = "We group the readings. Example: -2.5, -1, 3, 4 -> -3 to -1, 2 to 5."
        result = checked(30, proposal, [hit], 1, student)
        self.assertEqual(result["mark"], 2)
        self.assertEqual(result["criterion_evidence"]["example"]["quote"], rule)
        # An assessor's positive example verdict cannot override contradictory arithmetic.
        both = proposal | {"criterion_evidence": [entry, entry | {"criterion":"example"}]}
        for invalid in ("2, 4, 7 -> 1-4, 4-9", "2, 20 -> 0-5, 6-10", "2, 9 -> 5-0, 6-10"):
            self.assertIsNone(checked(30, both, [hit], 1, invalid)["mark"])
        self.assertIsNone(checked(30, proposal, [hit], 1, "Divide into bins.")["mark"])
        self.assertIsNone(checked(30, proposal | {"criterion_evidence":[]}, [hit], 1, student)["mark"])
        self.assertIsNone(checked(30, proposal | {"criterion_evidence":[entry | {"coverage":"partial"}]},
                                  [hit], 1, student)["mark"])
        unrelated = "An unrelated course topic has a numerical measurement."
        self.assertIsNone(checked(30, proposal | {"quote":unrelated,"criterion_evidence":[entry | {"quote":unrelated}]},
                                  [hit | {"text":unrelated}], 1, student)["mark"])

    def test_prose_and_half_open_ranges_are_left_to_semantic_assessment(self):
        for example in ("Temperatures below freezing are cold; other readings are warm.",
                        "0, 1, 2 -> [0,1), [1,3)", "The earlier example is wrong: 1, 2 -> 0-1, 2-3"):
            self.assertIsNone(check_interval_example(example))

    def test_prose_application_requires_the_students_own_example_and_course_rule(self):
        text = "Readings below 15 degrees are cool; those at or above 15 are warm."
        rule = "Binning divides continuous measurements into discrete intervals."
        hit = dict(id="p", title="Numerical features", heading="Reading", url="", text=rule, score=.8)
        source = dict(criterion="method", coverage="full", passage_id="p", quote=rule)
        application = dict(verdict="valid", student_quote=text, explanation="The two bins meet at 15 without overlap.")
        response = dict(mark=2, feedback="Defines intervals and illustrates the boundary.", needs_review=False,
                        deductions=[], passage_id="p", quote=rule, criterion_evidence=[source],
                        binning_application=application)
        self.assertEqual(checked(30, response, [hit], 1, text)["mark"], 2)
        for alternate in (dict(verdict="uncertain"), dict(student_quote="A different student's example.")):
            altered = response | {"binning_application":application | alternate}
            self.assertIsNone(checked(30, altered, [hit], 1, text)["mark"])
        self.assertIsNone(checked(30, response | {"criterion_evidence":[]}, [hit], 1, text)["mark"])
