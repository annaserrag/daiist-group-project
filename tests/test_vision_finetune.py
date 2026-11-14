import unittest
from unittest.mock import patch

from exam_grader.grading import grade_exam
from exam_grader.handoff import inspect_export
from exam_grader.page_confidence import confirmed_text_score, confirm_numbers, confirms_other_document, scored_lines
from exam_grader.vision_finetune import adapt


def row(number, text, **overrides):
    return {"number": number, "part": None, "text": text,
            "number_conf": 0.95, "ocr_conf": 0.91, **overrides}


def convert(*rows, loose=None):
    return adapt({"questions": list(rows), "unassigned": loose or [], "stats": {"lines": len(rows)}})


class VisionFinetuneTests(unittest.TestCase):
    def test_recovered_labels_require_agreement_on_ids_and_answer_words(self):
        from vision import reader
        from exam_grader.page_confidence import recover_layout
        def lines(*values):
            return [(text, [.97] * len(text)) for text in values]
        first = lines("C", "A")
        layout = lines("Data Science Examination | Answer sheet", "18 C", "20 A")
        recovered, scores = recover_layout(first, layout, lines("18 C", "20 A"), reader)
        self.assertEqual([s for s, _ in recovered][1:], ["18. C", "20. A"])
        self.assertEqual(scores, [.97, .97])
        for other in [lines("18 C", "19 A"), lines("18 A", "20 C"), lines("18 C")]:
            self.assertEqual(recover_layout(first, layout, other, reader), (first, []))
        self.assertEqual(recover_layout(lines("C", "D"), layout, lines("18 C", "20 A"), reader)[1], [])

    def test_crop_verification_requires_unique_exact_phrase(self):
        from exam_grader.page_confidence import verify_crop_words
        phrase = "learning from examples"
        crops = [[(phrase, [.96] * len(phrase))]]
        words = phrase.split()
        self.assertEqual(verify_crop_words(phrase, [.5]*3, words, crops), .96)
        self.assertEqual(verify_crop_words(phrase, [.5]*3, words*2, crops), .5)
        self.assertEqual(verify_crop_words("learning from mistakes", [.5]*3, words, crops), .5)

    def test_low_confidence_digit_keeps_its_own_review_gate(self):
        from vision import reader
        from exam_grader.page_confidence import recover_layout
        original = [("A", [.98]), ("B", [.96])]
        layout = [("3. A", [.5, .99, .99, .97]), ("7. B", [.99]*4)]
        lines, scores = recover_layout(original, layout, layout, reader)
        self.assertEqual([s for s, _ in lines], ["3. A", "7. B"])
        self.assertEqual(scores, [.5, .99])
    def test_wrong_document_needs_a_title_and_matching_confident_image_read(self):
        def line(text, confidence=1.0):
            return text, [confidence] * len(text)

        with patch("exam_grader.page_confidence.page_lines") as classify:
            classify.return_value = [line("A")]
            self.assertTrue(confirms_other_document(None, None, [line("Supplier invoice"), line("Total due 200")]))
            classify.return_value = [line("B")]
            self.assertTrue(confirms_other_document(None, None, [line("History quiz"), line("1. Rome")]))
            for heading in ("Supplier invoice", "Data Science quiz", "1. Test accuracy", "My answer"):
                self.assertFalse(confirms_other_document(None, None, [line(heading), line("Supporting text")]))
            self.assertFalse(confirms_other_document(None, None, [line("History quiz")]))
            self.assertFalse(confirms_other_document(None, None, [line("History quiz", .5), line("1. Rome")]))
            for reply in ([line("C")], [line("B", .6)], [line("B or C")], []):
                classify.return_value = reply
                self.assertFalse(confirms_other_document(None, None, [line("History quiz"), line("1. Rome")]))
            classify.side_effect = RuntimeError("optional image check unavailable")
            self.assertFalse(confirms_other_document(None, None, [line("History quiz"), line("1. Rome")]))

    def test_rejected_document_has_no_exam_identity_or_gradeable_answers(self):
        raw = {"questions": [row("1", "A")], "unassigned": [], "confirmed_other_document": True,
               "stats": {"lines": 2, "ink_density": .01}}
        rejected = adapt(raw)
        self.assertEqual((rejected["exam_status"], rejected["exam_id"], rejected["answers"]),
                         ("out_of_scope", None, []))
        self.assertEqual(rejected["features"], {"line_count": 2, "ink_density": .01})
        self.assertTrue(inspect_export(rejected)["valid"])

    def test_written_reread_aligns_whole_words_across_line_wraps(self):
        lines = [(" labeled", [.1] + [.96] * 7), ("data.", [.5] * 5)]
        self.assertEqual(confirmed_text_score("labeled data.", [.59, .99], lines), .96)
        self.assertEqual(confirmed_text_score("label", [.59], [("label", [.98, .97, .6, .98, .99])]), .6)

    def test_written_reread_rejects_changed_case_negation_and_duplicate_words(self):
        for first, second in [("k-NN", "K-NN"), ("not labeled", "labeled"), ("number number", "number"),
                              ("0.5", "05"), ("[?]", "[?]")]:
            with self.subTest(first=first, second=second):
                self.assertIsNone(confirmed_text_score(first, [.7] * len(first.split()),
                                                       [(second, [.99] * len(second))]))

    def test_confirmed_text_score_does_not_promote_a_question_number_or_an_mcq(self):
        answers = convert(row("21", "words", number_conf=.5, ocr_conf=.59, confirmed_answer_conf=.96),
                          row("1", "A", ocr_conf=.45, confirmed_answer_conf=.99))["answers"]
        self.assertEqual([(a["id_confidence"], a["confidence"]) for a in answers], [(.5, .96), (.95, .45)])

    def test_second_read_must_match_the_id_at_the_same_position(self):
        rows = [row("21", "written", number_token_conf=.99), row("32", "other", number_token_conf=.94)]
        confirm_numbers(rows, [("21", [.98, .91])])
        self.assertNotIn("confirmed_number_conf", rows[0])
        confirm_numbers(rows, [("21)", [.98, .91, .4]), ("31", [.99, .99])])
        self.assertEqual(rows[0]["confirmed_number_conf"], .91)
        self.assertNotIn("confirmed_number_conf", rows[1])

    def test_page_scores_require_confirmation_and_keep_written_answer_confidence(self):
        q = row("4", "B, regression", number_conf=.588, ocr_conf=.588, number_token_conf=.99,
                choice_token_conf=.96, answer_token_conf=.588)
        self.assertEqual(convert(q)["answers"][0]["id_confidence"], .588)
        answers = convert(dict(q, confirmed_number_conf=.97),
                          row("21", "Answer", confirmed_number_conf=.96, answer_token_conf=.53))["answers"]
        self.assertEqual([(a["id_confidence"], a["confidence"]) for a in answers], [(.97, .96), (.96, .53)])
        self.assertEqual(list(scored_lines([("21", .93), (".\n3", .55), ("4", .89)])),
                         [("21.", [.93, .93, .55]), ("34", [.55, .89])])

    def test_labelled_option_uses_the_current_question_and_keeps_review_gates(self):
        payload = convert(row("1", "D, All of the mentioned"), row("2", "b, MINIMUM."),
                          row("3", "D, All of the mentioned."), row("4", "B, regression", number_conf=.58))
        self.assertEqual([a["text"] for a in payload["answers"]], ["D", "B", "D, All of the mentioned.", "B"])
        decisions = grade_exam(payload, index=None)["teacher"]["answers"][:4]
        self.assertEqual([a["mark"] for a in decisions], [0, 0, None, None])
        self.assertEqual([a["reason"] for a in decisions[2:]], ["invalid_choice", "uncertain_question_id"])

    def test_labelled_choice_requires_the_whole_option_including_math_symbols(self):
        texts = ["C)  Both a  &  B", "C 0.5", "B: TP / (TP + FP)", "B, TP / (TP - FP)",
                 "B, Minimum or C", "A, Minimum", "D, False", "B, Minimum."]
        ids = ["4", "7", "19", "19", "2", "2", "17", "21"]
        payload = convert(*(row(q, t) for q, t in zip(ids, texts, strict=True)))
        self.assertEqual([a["text"] for a in payload["answers"]], ["C", "C", "B", *texts[3:]])

    def test_arrow_requires_one_valid_mcq_option_and_keeps_confidence(self):
        payload = convert(row("1", " → b "), row("2", "→A / C"), row("3", "→ C extra"),
                          row("4", "→ D", ocr_conf=.45), row("17", "→ D"), row("21", "→ A"))
        answers = payload["answers"]
        self.assertEqual([a["text"] for a in answers], ["B", "→A / C", "→ C extra", "D", "→ D", "→ A"])
        self.assertEqual(answers[0]["id_confidence"], .95)
        self.assertEqual(answers[3]["confidence"], .45)
        decisions = grade_exam(payload, index=None)["teacher"]["answers"]
        self.assertEqual([a["reason"] for a in decisions[:4]],
                         ["matches_key", "invalid_choice", "invalid_choice", "low_read_confidence"])
        self.assertEqual(decisions[16]["reason"], "invalid_choice")

    def test_ambiguous_and_written_text_survive_conversion(self):
        payload = convert(row("1", " (a) "), row("2", "B or C"), row("21", "  données\nlabelled  "))
        self.assertTrue(inspect_export(payload)["valid"])
        self.assertEqual([a["text"] for a in payload["answers"]], ["A", "B or C", "  données\nlabelled  "])
        self.assertEqual(payload["answers"][0]["id_confidence"], 0.95)
        self.assertEqual(payload["answers"][0]["confidence"], 0.91)
        self.assertEqual(payload["features"], {"line_count": 3})
        self.assertEqual(grade_exam(payload, index=None)["teacher"]["answers"][1]["reason"], "invalid_choice")

    def test_missing_numbers_subparts_and_loose_lines_block_assignment(self):
        payload = convert(row("1", "A"), row(None, "illegible number"), row("37", "other"),
                          row("21", "second part", part="ii"), loose=[{"text": "loose", "ocr_conf": .99}])
        self.assertEqual(inspect_export(payload)["unassigned"], 4)
        self.assertTrue(all(a["reason"] == "unassigned_answer"
                            for a in grade_exam(payload, index=None)["teacher"]["answers"]))

    def test_duplicate_blank_unknown_confidence_and_missing_do_not_get_marks(self):
        payload = convert(row("1", "B"), row("1", "D"), row("2", "", ocr_conf=None),
                          row("3", "A", number_conf=float("nan")), row("4", "C", ocr_conf=None))
        answers = grade_exam(payload, index=None)["teacher"]["answers"]
        self.assertEqual([a["reason"] for a in answers[:5]],
                         ["duplicate_detection", "unreadable", "uncertain_question_id",
                          "low_read_confidence", "missing_detection"])
        self.assertTrue(all(a["mark"] is None for a in answers))

    def test_empty_and_malformed_responses(self):
        for payload in (convert(), convert(row("4", "")), convert(row(None, "unknown"))):
            self.assertEqual(payload["exam_status"], "unreadable")
        with self.assertRaises(KeyError):
            adapt({"questions": []})
        with self.assertRaises(ValueError):
            convert(row("21", None))

    def test_recover_bare_mcq_only_on_matching_page(self):
        def line(page, text, confidence=.96):
            return {"page": page, "text": text, "ocr_conf": confidence}
        loose = [line(1, "Data Science Examination | Answer sheet"), line(1, "1  d"),
                 line(1, "2 c", .4), line(1, "2 C"), line(1, "3 A or D"),
                 line(1, "21 A"), line(2, "4 C")]
        answers = convert(loose=loose)["answers"]
        self.assertEqual([a["question_id"] for a in answers], [1, 2, 2, None, None, None])
        self.assertEqual(answers[0]["text"], "D")
        self.assertEqual(answers[1]["id_confidence"], .4)
        self.assertEqual(answers[1]["confidence"], .4)
