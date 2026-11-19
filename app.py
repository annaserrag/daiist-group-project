"""Local Component 2 workbench and course-compatible Gradio app."""
from collections import Counter
from html import escape
import math
import os
import tempfile
from pathlib import Path
from subprocess import SubprocessError
from time import perf_counter

import gradio as gr
import pandas as pd

import desk_ui as ui
import logbook
import monitor
import pipeline
from exam_grader import bandit
from exam_grader.catalog import KEY, KEY_URL, OPTION_TEXTS, QUESTIONS
from exam_grader.grading import Policy
from exam_grader.rubric import CRITERIA
from exam_grader.build import vector
from exam_grader.written_exam import WrittenExam
from exam_grader.materials import BUILTIN, add_pdf

REVIEW_ACTIONS = {
    "missing_detection": "Check whether this question has a response on the original page.",
    "duplicate_detection": "Compare the readings with the page and identify the answer for this question.",
    "unassigned_answer": "Assign the unnumbered reading before signing off the exam.",
    "uncertain_question_id": "Check the question number against the original page.",
    "unreadable": "Inspect the original or request a clearer scan.",
    "low_read_confidence": "Verify the transcription against the handwritten answer.",
    "invalid_answer": "Check the reading, confidence and state in the vision export.",
    "empty_transcription": "Check the answer area and confirm whether it was left blank.",
    "unverified_blank": "Inspect the answer area before assigning zero for a blank.",
    "invalid_choice": "Check the marked option against the choices printed on the exam.",
    "context_check": "Open the figure referenced by the question and compare it with the answer before assigning credit.",
    "weak_retrieval": "Use the official key and check a teaching passage that supports this answer.",
    "no_index": "Open the official key or load the course index before grading.",
    "provider_disabled": "Grade from the key, or run the written grader on an approved input.",
    "provider_failure": "The written grader could not finish. Retry or mark from the key.",
    "lookup_limit": "The grader requested extra evidence beyond its lookup limit. Consult the course source.",
    "model_call_limit": "The grader reached its call limit. Inspect the answer against the key.",
    "provider_rate_limit": "The grading service reached its request limit. Retry later or mark manually.",
    "provider_daily_quota": "The configured model reached its daily request limit. Mark manually or retry after the quota resets.",
    "provider_unavailable": "The grading service was unavailable. Retry or use the official key.",
    "missing_frozen_response": "This evaluation has no saved response for the exact request.",
    "invalid_model_output": "Check the rubric and enter a mark on the question's scale.",
    "structured_response_invalid": "The model response failed validation. Check the answer against the rubric.",
    "unexplained_deduction": "Assess each rubric criterion and explain the points deducted.",
    "unverified_citation": "Check the claimed support in the course material before awarding credit.",
    "insufficient_evidence": "Read the student's answer with the key and decide which criteria it meets.",
    "out_of_scope": "Check that the correct exam was uploaded.",
    "invalid_contract": "Validate the vision export before grading this exam.",
}

def process_once(raw_input, course_index=None, exam_state=None, review_arm=None):
    """Grade one paper and record whether the bandit selected its written review rule."""
    started = perf_counter()
    exam, queries = exam_state or (None, None)
    policy = None
    if exam is not None:
        review_arm = None
    if review_arm is not None:
        policy = Policy(written_review_at_or_below=bandit.CUTOFFS[review_arm])
    views = pipeline.run_views(raw_input, policy=policy, course_index=course_index, written_exam=exam, query_vectors=queries)
    record = views["record"]
    record["versions"]["review_arm"] = review_arm
    record["versions"]["review_policy_source"] = "bandit" if review_arm is not None else "fixed"
    record["latency_s"] = round(perf_counter() - started, 4)
    return views["teacher"], logbook.log_request(record)

def serve(raw_input):
    return process_once(raw_input)[1]

def selected_index(library, course):
    if course == BUILTIN:
        return None
    if course not in (library or {}):
        raise gr.Error("Select an indexed course or add a PDF in Materials.")
    return library[course]


def index_material(file, course, library):
    try:
        updated, active, message = add_pdf(library, course, file, pipeline.resources()[0])
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return updated, gr.Dropdown(choices=[BUILTIN, *updated], value=active), message, []


def select_exam(file):
    if not file:
        return None, "Built-in ML exam selected.", gr.Dropdown(choices=[(QUESTIONS[q], q) for q in range(21,36)], value=21)
    try:
        exam = WrittenExam.load(file)
        queries = None
        key = os.getenv("GEMINI_API_KEY")
        if key and pipeline.resources()[0].vectors:
            queries = {str(q.question_id): vector("task: question answering | query: " + q.question, key)
                       for q in exam.questions}
    except ValueError as exc:
        raise gr.Error("Invalid exam definition. Check the question IDs, required fields and rubric weights.") from exc
    except OSError as exc:
        raise gr.Error("Could not read the exam or embed its questions. Try again.") from exc
    choices = [(f"Q{q.question_id}. {q.question}", q.question_id) for q in exam.questions]
    return (exam, queries), f"{exam.title}, reader exam_id: {exam.exam_id}", gr.Dropdown(choices=choices, value=choices[0][1])


def preview_material(question, library, course=BUILTIN, exam_state=None):
    index = selected_index(library, course) or pipeline.resources()[0]
    if exam_state:
        exam, queries = exam_state
        index = index.for_questions({q.question_id:q.question for q in exam.questions}, queries, exam.exam_id)
    return [[hit["title"], hit.get("page"), hit["text"], round(hit["score"], 3)]
            for hit in index.search(int(question))]


def choose_review_arm(exam_state=None):
    """The bandit's written review rule for the next paper (Thompson sampling over the warm start and real
    sign-offs), or None when a policy file fixes the policy for an evaluation, or for a custom exam, whose written
    questions the rule was not built for."""
    if exam_state or os.getenv("GRADE_POLICY_FILE") or os.getenv("GRADE_BANDIT", "on") == "off":
        return None
    arm, _ = bandit.choose(bandit.load_prior(), bandit.rewards_from_logs(logbook.load_signoffs()))
    return arm


# "Try a sample paper": the demo exams, shown as page thumbnails the teacher clicks to grade one (pipeline.SAMPLE_PAPERS).
SAMPLES = [("h46", "Student 46"), ("h50", "Student 50"), ("h43", "Student 43"), ("h47", "Student 47"),
           ("c06", "Printed page c06")]
PAGE_CACHE = Path(tempfile.gettempdir()) / "exam-desk-pages"


def page_images(pdf, width=900):
    """The pages of a PDF as PNG files, so the teacher sees the paper next to its marks. Empty for a JSON export."""
    pdf = Path(pdf) if pdf else None
    if pdf is None or pdf.suffix.lower() != ".pdf" or not pdf.is_file():
        return []
    folder = PAGE_CACHE / f"{pipeline.file_sha256(pdf)[:16]}-{width}"
    if not any(folder.glob("page-*.png")):
        import pypdfium2 as pdfium
        folder.mkdir(parents=True, exist_ok=True)
        document = pdfium.PdfDocument(str(pdf))
        try:
            for number in range(len(document)):
                page = document[number]
                image = page.render(scale=width / page.get_width()).to_pil().convert("RGB")
                image.save(folder / f"page-{number + 1}.png")
        finally:
            document.close()
    return sorted((str(p) for p in folder.glob("page-*.png")), key=lambda p: int(Path(p).stem.split("-")[1]))


def marking_guide() -> str:
    """The built-in exam's questions with the published answer key (Mendeley), as the teacher would read them."""
    lines = ["**Data Science Examination**, 35 questions, 50 points. Answer key published with the dataset "
             f"([Mendeley]({KEY_URL})).", "", "**Part I, multiple choice** (1 point each)", ""]
    for q in range(1, 21):
        options = "; ".join(f"{letter}. {text}" for letter, text in OPTION_TEXTS[q].items())
        lines.append(f"{q}. {QUESTIONS[q]}  \n{options}  \n**Key: {KEY[q]}**")
    lines += ["", "**Part II, short answers** (2 points each)", ""]
    for q in range(21, 36):
        lines.append(f"{q}. {QUESTIONS[q]}  \n**Key:** {KEY[q]}")
    return "\n".join(lines)


def guide_thumbnail():
    """A small picture of the marking guide's first page, so the card looks like the document it opens."""
    out = PAGE_CACHE / "marking-guide-v2.png"
    if not out.is_file():
        from PIL import Image, ImageDraw, ImageFont
        PAGE_CACHE.mkdir(parents=True, exist_ok=True)
        page = Image.new("RGB", (520, 340), "#fffefa")
        draw = ImageDraw.Draw(page)
        title, body = ImageFont.load_default(size=22), ImageFont.load_default(size=14)
        draw.text((28, 22), "Data Science Examination", fill="#1d2b25", font=title)
        draw.text((28, 54), "Marking guide and answer key", fill="#4d5d55", font=body)
        y = 88
        for q in (1, 2, 3, 21, 22):
            text = f"{q}. {QUESTIONS[q]}"
            draw.text((28, y), text[:62] + ("…" if len(text) > 62 else ""), fill="#1d2b25", font=body)
            draw.text((44, y + 19), f"Key: {KEY[q]}"[:60], fill="#22533f", font=body)
            y += 48
        page.save(out)
    return str(out)


def sample_gallery():
    return [(page_images(pipeline.SAMPLE_PAPERS[name], 360)[0], caption) for name, caption in SAMPLES]


def grade_sample(library, course, exam_state, evt: gr.SelectData):
    """Grade the sample paper the teacher clicked, and put it in the upload box so it is clear which one it is."""
    pdf = pipeline.SAMPLE_PAPERS[SAMPLES[evt.index][0]]
    return (*display(pdf, library, course, exam_state), str(pdf))


def display(raw_input, library=None, course=BUILTIN, exam_state=None):
    if not raw_input:
        raise gr.Error("Choose an OCR export or a PDF first.")
    arm = choose_review_arm(exam_state)
    teacher, record = process_once(raw_input, selected_index(library, course), exam_state, arm)
    review_index = None
    evidence, marks = [], []
    referrals = {}
    custom = {q.question_id: q for q in exam_state[0].questions} if exam_state else {}
    for item in teacher["answers"]:
        source = item["citation"] or {}
        feedback = item["feedback"] or REVIEW_ACTIONS.get(item.get("execution_issue",item["reason"]), "")
        evidence.append([item["question_id"], item["question"], " | ".join(item["readings"]), item["max_mark"], item["mark"],
                         item["reason"].replace("_", " "), feedback,
                         source.get("title", ""), source.get("page"), source.get("url", ""), source.get("quote", ""),
                         ", ".join(item.get("opened_passages",[])) or "",
                         "\n".join(f"{criterion} ({c['coverage']}), {c['title']}"
                                   + (f", p. {c['page']}" if "page" in c else "")
                                   + f", {c['quote']}" + (f", Example check: {c['basis']}" if "basis" in c else "")
                                   for criterion,c in item.get("criterion_evidence",{}).items())])
        q = item["question_id"]
        if teacher["exam_status"] == "in_scope" and item["mark"] is None:
            guide = custom[q].guide() if custom else CRITERIA.get(q, {})
            key = custom[q].reference_answer if custom else KEY[q]
            rubric = [f"{name} ({points:g} points), {standard}" for name, (points, standard) in guide.items()]
            proofs = item.get("criterion_evidence", {})
            sources = []
            for name, cite in proofs.items():
                location = cite["title"] + (f", p. {cite['page']}" if "page" in cite else "")
                sources.append(f"{name} ({cite['coverage']}), {location}, {cite['quote']}")
            passages = []
            review_hits = item.get("review_passages", [])
            if (guide and len(item["readings"]) == 1
                    and item["reason"] in {"uncertain_question_id", "low_read_confidence"}):
                # Local search for the reviewer only. The unverified OCR cannot authorize a mark.
                if review_index is None:
                    review_index = selected_index(library, course) or pipeline.resources()[0]
                    if exam_state:
                        exam, queries = exam_state
                        review_index = review_index.for_questions(
                            {entry.question_id: entry.question for entry in exam.questions}, queries, exam.exam_id)
                review_hits = [dict(hit, excerpt=hit["text"][:1200])
                               for hit in review_index.search(q, item["readings"][0])]
            for hit in review_hits:
                location = hit["title"] + (f", p. {hit['page']}" if "page" in hit else "")
                if hit.get("url"):
                    location += f", {hit['url']}"
                passages.append(f"{location}\n{hit['excerpt']}")
            # The teacher can assess a large deduction independently. The original
            # model suggestion stays available in the full evidence table.
            action = ("Assess each rubric point before entering a mark."
                      if item["reason"] == "written_mark_review" else feedback)
            choices = {} if custom else OPTION_TEXTS.get(q, {})
            if key in choices:
                key = f"{key}, {choices[key]}"
            if choices and item["reason"] == "invalid_choice":
                action = "Check the original paper. The reader found a choice outside " + ", ".join(
                    f"{letter} ({value})" for letter, value in choices.items()) + "."
            referrals[q] = dict(question=item["question"], reading=" | ".join(item["readings"]),
                                key=key, rubric=rubric, sources=sources, passages=passages, action=action,
                                reason=item["reason"].replace("_", " "))
        marks.append([q, item["mark"]])
    count = teacher["review_count"]
    if teacher["input_type"] == "pdf_awaiting_reader":
        status = ("PDF received, but this online version can't read new exams: the page reader needs several GB of"
                  " memory and the free server has 512 MB. Try one of the demo exams, or run the app on a laptop"
                  " (SETUP.md) to read any PDF.")
    elif teacher.get("reader_failed"):
        status = "The vision reader failed. No exam total available; retry or inspect the reader."
    elif teacher["exam_status"] != "in_scope":
        status = f"Exam status: {teacher['exam_status'].replace('_',' ')}. No exam total available."
    else:
        status = f"{len(marks)-count} of {len(marks)} questions have a proposed mark. {count} require review."
        if arm is not None:
            status += (f" Written answers sent to you: {bandit.LABELS[arm]}. The review rule chose this from earlier"
                       " sign-offs.")
        if teacher["near_pass_boundary"]:
            status += " Check the total against the configured pass boundary."
    context = {"id": record["id"], "in_scope": teacher["exam_status"] == "in_scope",
               "near_boundary": teacher.get("near_pass_boundary", False),
               "limits":[(r["question_id"],r["max_mark"]) for r in teacher["answers"]],
               "referrals":referrals, "review_started":perf_counter(), "proposals":[point for _,point in marks],
               "review_arm":arm,
               # What the teacher needs to check any proposed mark (no student identity, only the reading).
               "details":{item["question_id"]: {
                   "question": item["question"], "reading": " | ".join(item["readings"]), "max": item["max_mark"],
                   "reason": item["reason"].replace("_", " "), "feedback": item["feedback"] or "",
                   "source": ((item["citation"] or {}).get("title", "")
                              + (f", p. {(item['citation'] or {})['page']}" if (item["citation"] or {}).get("page") else "")),
                   "quote": (item["citation"] or {}).get("quote", "")} for item in teacher["answers"]}}
    eligible = {q for q, referral in referrals.items() if referral["reason"] == "low read confidence"}
    for page in teacher.get("crop_pages", []):
        for answer in page["answers"]:
            q = answer["question_id"]
            if q in eligible and q in custom:
                referrals[q]["crop"] = dict(page=page["page"], parts=page["parts"], words=answer["words"],
                                            targets=answer["targets"], label=custom[q].paper_label or str(q))
    if any("crop" in referral for referral in referrals.values()):
        context["crop_pdf"] = str(Path(raw_input).resolve())
        context["crop_sha256"] = pipeline.file_sha256(raw_input)
    return status, evidence, marks, context, None

def question_view(context, question):
    item = (context or {}).get("referrals", {}).get(question)
    if not item:
        return "", "", "", gr.update(choices=[], value=None), gr.update(interactive=False)
    maximum = dict(context["limits"])[question]
    reading = item["reading"] or "No answer was transcribed. Check the original paper."
    card = (f'<article class="review-question"><h3>Question {question}</h3>'
            f'<p class="question-prompt">{escape(item["question"])}</p>'
            '<div class="answer-pair"><div><h4>Student answer</h4>'
            f'<p>{escape(reading)}</p></div><div><h4>Answer key</h4><p>{escape(item["key"])}</p></div></div>'
            f'<p class="review-reason"><strong>What to check</strong> {escape(item["action"])}</p></article>')
    support = list(item["sources"])
    if item["passages"]:
        support += ["Retrieved passages for teacher inspection (unverified)", *item["passages"]]
    guide = "\n".join(item["rubric"]) or item["key"]
    choices = [(f"{point:g}", point) for point in
               ([0, 1] if maximum == 1 else [n / 2 for n in range(int(2 * maximum) + 1)])]
    return (card, guide, "\n\n".join(support) or "No verified teaching passage available.",
            gr.update(choices=choices, value=None, label=f"Your mark, out of {maximum:g}"),
            gr.update(interactive=False))


def show_review_lines(context, question):
    from exam_grader.vision_finetune import read_review_lines
    item = (context or {}).get("referrals", {}).get(question, {})
    if not item.get("crop") or not context.get("crop_pdf"):
        return [], "No handwriting crop is available for this referral."
    path = Path(context["crop_pdf"])
    try:
        if not path.is_file() or pipeline.file_sha256(path) != context["crop_sha256"]:
            return [], "The graded PDF is no longer available. Grade the original upload again."
        images, text = read_review_lines(path, item["crop"])
    except (ImportError, OSError, RuntimeError, ValueError, IndexError, SubprocessError):
        return [], "The scan lines could not be loaded. Inspect the original PDF."
    return images, text or "No scan line could be matched to the disputed words."


def apply_review_mark(context, question, point, marks):
    referrals = (context or {}).get("referrals", {})
    if question not in referrals:
        raise gr.Error("Choose a referred question.")
    limits = dict(context["limits"])
    if (type(point) not in (int, float) or not math.isfinite(point) or not 0 <= point <= limits[question]
            or (point if limits[question] == 1 else point * 2) % 1):
        raise gr.Error(f"Enter a valid mark between 0 and {limits[question]} for Q{question}.")
    if [row[0] for row in marks] != list(limits):
        raise gr.Error("Restore the question numbers on the teacher sheet before recording this mark.")
    updated = [[q, point if q == question else old] for q, old in marks]
    remaining = [q for q, value in updated if q in referrals and value in (None, "")]
    remaining.sort(key=lambda q: (not referrals[q]["reading"].strip(), q))
    return updated, remaining[0] if remaining else None, None, None


def checked_mark(question, maximum, values):
    try:
        q, point = values
        if type(q) is bool or type(point) is bool or float(q) != question:
            raise ValueError()
        point = float(point)
        if not math.isfinite(point) or point < 0 or point > maximum or (point if maximum == 1 else point * 2) % 1:
            raise ValueError()
    except (TypeError, ValueError, OverflowError) as exc:
        raise gr.Error(f"Check Q{question}. Enter a valid teacher mark from 0 to {maximum}.") from exc
    return point


def sign_off(context, marks):
    if not context or not context["in_scope"]:
        raise gr.Error("Run an in-scope exam before signing off.")
    limits = context["limits"]
    if len(marks) != len(limits):
        raise gr.Error("The final sheet needs every question from the graded exam.")
    final_marks = [{"question_id": q, "mark": checked_mark(q, limit, row)}
                   for (q, limit), row in zip(limits, marks, strict=True)]
    signed = {"request_id": context["id"], "teacher_signed_off": True, "maximum": sum(maximum for _, maximum in limits),
              "total": sum(row["mark"] for row in final_marks), "marks": final_marks}
    if "review_started" in context and "proposals" in context:
        previous = context["proposals"]
        signed["review_session"] = {
            "elapsed_seconds_since_proposals_shown":round(perf_counter() - context["review_started"], 2),
            "initially_referred":sum(p is None for p in previous),
            "accepted_proposals":sum(p == r["mark"] for p,r in zip(previous, final_marks, strict=True)),
            "changed_proposals":sum(p is not None and p != r["mark"] for p,r in zip(previous, final_marks, strict=True))}
        # The bandit's reward: what this paper cost the teacher under the review rule it was graded with.
        session = signed["review_session"]
        arm = context.get("review_arm")
        reward = bandit.paper_reward(len(final_marks), session["initially_referred"], session["changed_proposals"])
        logbook.log_signoff(context["id"], arm, len(final_marks), session["initially_referred"],
                            session["changed_proposals"], round(reward, 4) if arm is not None else None)
    return signed

def rate(ctx, value):
    if not ctx:
        raise gr.Error("Run an exam before giving feedback.")
    logbook.log_feedback(ctx["id"], value)
    return "Feedback saved."

def monitoring():
    logs = logbook.load()
    ref = monitor.load_reference()
    alerts = monitor.alerts(logs, ref) if not logs.empty else []
    message = " ".join([monitor.drift_status(logs,ref),*alerts])
    frame = monitor.summary(logs)
    columns = ["time","requests","answer review share","thumbs down among rated"]
    if logs.empty:
        return message, frame, pd.DataFrame(columns=columns), review_rule()
    data = logs.set_index("timestamp")
    groups = data.resample("h")
    rows = []
    for when, group in groups:
        if group.empty:
            continue
        rated = group.feedback.dropna()
        answer_shares = group.get("f_review_share", pd.Series(dtype=float)).dropna()
        rows.append([str(when), len(group), round(answer_shares.mean(), 3) if len(answer_shares) else None,
                     round((rated < 0).mean(), 3) if len(rated) else None])
    return message, frame, pd.DataFrame(rows, columns=columns), review_rule()


def review_rule():
    """What the review-rule bandit has learnt: each arm's posterior and how often it was played."""
    signoffs = logbook.load_signoffs()
    state = bandit.posterior(bandit.load_prior(), bandit.rewards_from_logs(signoffs))
    logs = logbook.load()
    signed_arms = dict(zip(signoffs["id"], signoffs["review_arm"])) if not signoffs.empty else {}
    played = Counter()
    for request in logs.itertuples():
        arm = request.versions.get("review_arm")
        source = request.versions.get("review_policy_source")
        # Older logs used up_to_1 for both fixed and adaptive runs. Only their sign-off can distinguish them.
        if source is None and arm == bandit.DEFAULT_ARM:
            arm = signed_arms.get(request.id)
        if source in (None, "bandit") and arm in bandit.ARMS:
            played[arm] += 1
    return pd.DataFrame([{"rule": arm, "written answers sent to you": bandit.LABELS[arm],
                          "papers graded": int(played.get(arm, 0)),
                          "papers signed off": s["real_papers"], "warm-start papers": s["warm_start_papers"],
                          "teacher cost per answer (estimate)": round(-s["mean"], 3), "uncertainty": round(s["sd"], 3)}
                         for arm, s in state.items()])

def results_view(context, sheet, selected, signed, status):
    context = context or {}
    limits = context.get("limits", [])
    in_scope = bool(context.get("in_scope"))
    pending = []
    invalid = []
    total = 0
    structure_ok = len(sheet) == len(limits) and all(
        len(row) == 2 and row[0] == q for (q, _), row in zip(limits, sheet))
    if in_scope and structure_ok:
        for (q, maximum), row in zip(limits, sheet):
            if row[1] in ("", None):
                pending.append(q)
            else:
                try:
                    total += checked_mark(q, maximum, row)
                except gr.Error:
                    invalid.append(q)
    referrals = context.get("referrals", {})
    queue = sorted((q for q in pending if q in referrals), key=lambda q: (not referrals[q]["reading"].strip(), q))
    current = selected if selected in queue else next(iter(queue), None)
    completed = len(limits) - len(pending) - len(invalid)
    ready = in_scope and bool(limits) and structure_ok and completed == len(limits)
    signed_here = ready and bool(signed) and signed.get("request_id") == context.get("id")
    if not in_scope:
        title, detail = "Check the uploaded paper", status or ""
    elif not structure_ok or invalid:
        title = "Check the mark sheet"
        detail = (f"Correct the mark for Q{invalid[0]} in the mark sheet." if invalid
                  else "Restore the original question rows in the mark sheet.")
    elif ready:
        title = "Paper signed off" if signed_here else "Ready to sign off"
        detail = f"{total:g} / {sum(limit for _, limit in limits):g} points, {len(limits)} questions marked."
    else:
        title = f"{len(pending)} {'question needs' if len(pending) == 1 else 'questions need'} a mark"
        detail = f"{completed} of {len(limits)} marked."
        if not queue:
            detail += " Open the mark sheet to fill the empty marks."
    if in_scope and context.get("near_boundary"):
        detail += " Check the total against the pass boundary."
    progress = (f'<progress aria-label="Questions with valid marks" value="{completed}" max="{len(limits)}"></progress>'
                if in_scope and structure_ok and limits else "")
    summary = (f'<section class="review-summary" aria-live="polite"><h2>{title}</h2>'
               f'<p>{escape(detail)}</p>{progress}</section>')
    return (gr.update(visible=bool(context)), summary, gr.update(visible=bool(queue)),
            gr.update(choices=[(f"Question {q}", q) for q in queue], value=current),
            gr.update(visible=in_scope), gr.update(visible=ready and not signed_here),
            gr.update(visible=signed_here))


# ---------------------------------------------------------------- the redesigned Grade tab

def paper_view(context, sheet, selected, focus, signed):
    """The stepper, the summary band and the answer map, from the context and the teacher's current sheet."""
    context = context or {}
    limits = context.get("limits", [])
    if not context.get("in_scope"):
        return ui.stepper(0), "", ui.map_value(None, [], None)
    marks = {row[0]: row[1] for row in sheet or [] if len(row) == 2}
    referrals = context.get("referrals", {})
    pending = [q for q, _ in limits if marks.get(q) in (None, "")]
    need_you = [q for q in pending if q in referrals]
    signed_here = bool(signed) and signed.get("request_id") == context.get("id")
    stage = 3 if signed_here else 2 if not pending else 1
    total = sum(v for v in marks.values() if isinstance(v, (int, float)))
    maximum = sum(m for _, m in limits)
    proposals = context.get("proposals", [])
    changed = sum(1 for (q, _), p in zip(limits, proposals) if p is not None and marks.get(q) not in (None, "") and marks[q] != p)
    if signed_here:
        title, detail = "Paper signed off", f"{total:g} of {maximum:g} points. The marks are final."
    elif not pending:
        title, detail = "Ready to sign off", f"{total:g} of {maximum:g} points. Check anything you want, then sign off."
    else:
        title = f"{len(need_you) or len(pending)} answer{'s' if (len(need_you) or len(pending)) != 1 else ''} need you"
        detail = "Click a question marked ? on the map, or work through them in order below."
    chips = [("proposed by the product", str(sum(p is not None for p in proposals))),
             ("sent to you", str(len(referrals))), ("you changed", str(changed)),
             ("points so far", f"{total:g}/{maximum:g}")]
    if context.get("review_arm") is not None:
        chips.append(("written review rule", context["review_arm"].replace("_", " ")))
    band = ui.summary_band(title, detail, len(limits) - len(pending), len(limits), chips)
    shown = focus if focus is not None else selected
    return ui.stepper(stage, len(need_you)), band, ui.map_value(context, sheet, shown)


def open_question(context, picked, sheet):
    """A click on the answer map. A question still waiting for the teacher opens in the review panel; any other
    question opens in the check panel, where its mark can be confirmed or changed."""
    context = context or {}
    q = (picked or {}).get("selected")
    limits = dict(context.get("limits", []))
    if q not in limits:
        return None, gr.update(), gr.update(visible=False), "", gr.update(choices=[], value=None)
    marks = {row[0]: row[1] for row in sheet or [] if len(row) == 2}
    referrals = context.get("referrals", {})
    if q in referrals and marks.get(q) in (None, ""):
        return None, gr.update(value=q), gr.update(visible=False), "", gr.update(choices=[], value=None)
    detail = dict(context.get("details", {}).get(q, {}))
    if q in referrals:  # already marked by the teacher: show the referral's reading and key
        item = referrals[q]
        detail.update(reading=item["reading"], feedback=f"Answer key: {item['key']}", source="", quote="")
    detail.setdefault("question", QUESTIONS.get(q, f"Question {q}"))
    detail.setdefault("reading", ""); detail.setdefault("reason", ""); detail.setdefault("feedback", "")
    detail["max"] = limits[q]
    maximum = limits[q]
    choices = [(f"{point:g}", point) for point in ([0, 1] if maximum == 1 else [n / 2 for n in range(int(2 * maximum) + 1)])]
    current = marks.get(q)
    return (q, gr.update(), gr.update(visible=True), ui.proposal_card(q, detail, current),
            gr.update(choices=choices, value=current if current not in ("", None) else None,
                      label=f"Mark for Q{q}, out of {maximum:g}"))


def apply_check_mark(context, question, point, marks):
    """Confirm or change the mark of a question opened in the check panel."""
    limits = dict((context or {}).get("limits", []))
    if question not in limits:
        raise gr.Error("Choose a question on the answer map first.")
    if (type(point) not in (int, float) or not math.isfinite(point) or not 0 <= point <= limits[question]
            or (point if limits[question] == 1 else point * 2) % 1):
        raise gr.Error(f"Choose a mark between 0 and {limits[question]:g} for Q{question}.")
    if [row[0] for row in marks] != list(limits):
        raise gr.Error("Restore the question numbers on the mark sheet before changing a mark.")
    return [[q, point if q == question else old] for q, old in marks], None


# ---------------------------------------------------------------- the redesigned Monitoring tab

def monitoring_view():
    """Everything the Monitoring tab shows: the alert, the tiles, quality over time, drift per signal, and what the
    review rule has learnt."""
    logs = logbook.load()
    reference = monitor.load_reference()
    message, frame, timeline, rule = monitoring()
    fired = monitor.alerts(logs, reference) if not logs.empty else []
    quality = monitor.quality(logs) if not logs.empty else pd.DataFrame(columns=["time", "signal", "value"])
    if not quality.empty:  # the plot reads naive datetimes; the logs are in UTC
        quality["time"] = pd.to_datetime(quality["time"]).dt.tz_localize(None)
    drift = pd.DataFrame(columns=["feature", "psi", "threshold", "alert"])
    if reference is not None and len(logs) >= monitor.MIN_WINDOW:
        drift = monitor.drift(reference, monitor.live_features(logs), monitor.MIN_WINDOW)
    costs = rule.assign(rule=rule["rule"].str.replace("_", " "), cost=rule["teacher cost per answer (estimate)"])
    return (ui.alert_banner(message, fired), ui.tiles(frame), quality, drift, costs, frame, timeline)


def build_app():
    with gr.Blocks(title="Exam desk, exam grading assistant") as demo:
        masthead = gr.HTML(ui.masthead(bool(os.getenv("GEMINI_API_KEY"))))
        library = gr.State({})
        exam_state = gr.State(None)
        ctx = gr.State(None)
        focus = gr.State(None)  # a question opened from the answer map that is not waiting for the teacher

        # ------------------------------------------------------------ 1. Grade
        with gr.Tab("Grade", id="grade"):
            progress_steps = gr.HTML(ui.stepper(0), visible=False)  # kept for paper_view; the paper bar replaces it
            # ---- start: one drop zone, the marking guide next to it, and the sample papers below
            with gr.Column(elem_id="start-box") as start_box:
                with gr.Row(elem_classes="desk-columns"):
                    with gr.Column(scale=3, min_width=280, elem_classes="desk-card upload-card"):
                        gr.HTML('<h2 class="card-title">Grade a paper</h2><p class="card-note">Drop a scanned exam '
                                'as a PDF, or the reader’s JSON export.</p>')
                        upload = gr.File(label="Student paper (PDF or reader JSON)", file_types=[".pdf", ".json"],
                                         type="filepath", height=150, elem_id="paper-drop", show_label=False)
                        go = gr.Button("Grade paper", variant="primary", elem_id="grade-button")
                    with gr.Column(scale=2, min_width=250, elem_classes=["desk-card", "desk-settings"]):
                        gr.HTML('<h2 class="card-title">Marking guide</h2><p class="card-note">The Data Science '
                                'exam and its published key (Mendeley).</p>')
                        gr.Image(value=guide_thumbnail, show_label=False, interactive=False, height=170,
                                 elem_id="guide-thumb", buttons=[], container=False)
                        with gr.Accordion("Open the marking guide", open=False, elem_classes="guide-accordion"):
                            gr.Markdown(marking_guide(), elem_id="guide-text")
                        exam_info = gr.Textbox(label="Exam", value="Built-in ML exam selected.", interactive=False,
                                               visible=False)
                        course = gr.Dropdown([BUILTIN], value=BUILTIN, label="Course materials")
                with gr.Column(elem_classes="desk-card", elem_id="sample-papers"):
                    gr.HTML('<h2 class="card-title">Or try a sample paper</h2><p class="card-note">Click a paper '
                            'to grade it straight away.</p>')
                    samples = gr.Gallery(value=sample_gallery, columns=5, height=250, allow_preview=False,
                                         show_label=False, object_fit="contain", interactive=False)

            with gr.Column(visible=False, elem_id="grading-results") as results:
                status = gr.Textbox(visible=False)
                summary_card = gr.HTML(visible=False)  # kept for results_view; the paper bar below replaces it
                with gr.Row(elem_classes="paper-bar-row"):
                    band = gr.HTML(elem_id="paper-bar")
                    another = gr.Button("Grade another paper", variant="secondary", size="sm", elem_id="another")
                answer_map = ui.AnswerMap(elem_id="answer-map")

                # ---- the workspace: the paper on the left, one question at a time on the right
                with gr.Row(elem_classes="workspace", equal_height=False):
                    with gr.Column(scale=5, min_width=300, elem_classes="desk-card paper-viewer"):
                        gr.HTML('<p class="panel-kicker quiet">The paper</p>')
                        paper_pages = gr.Gallery(columns=4, height=640, show_label=False, object_fit="contain",
                                                 interactive=False, preview=True, selected_index=0,
                                                 elem_id="paper-pages")
                    with gr.Column(scale=6, min_width=320, elem_classes="workspace-side"):
                        with gr.Column(visible=False, elem_classes="desk-card check-panel", elem_id="check-panel") as check_panel:
                            check_card = gr.HTML()
                            check_point = gr.Radio(label="Mark", choices=[], interactive=True, elem_classes="mark-pills")
                            with gr.Row(elem_classes="mark-actions"):
                                check_close = gr.Button("Close", variant="secondary")
                                check_save = gr.Button("Save mark", variant="primary")

                        with gr.Column(visible=False, elem_classes="desk-card", elem_id="active-review") as review:
                            gr.HTML('<p class="panel-kicker">Needs your mark</p>')
                            reading = gr.HTML()
                            review_point = gr.Radio(label="Your mark", choices=[], interactive=True,
                                                    elem_classes="mark-pills", elem_id="review-point")
                            with gr.Row(elem_classes="mark-actions"):
                                gr.HTML('<p class="key-hint">Number keys pick a mark from the left (0 is the first), Enter saves</p>')
                                record_point = gr.Button("Save and next →", variant="primary", interactive=False,
                                                         elem_id="save-next")
                            with gr.Accordion("Marking guide and sources", open=False):
                                standard = gr.Textbox(label="Marking guide", interactive=False)
                                support = gr.Textbox(label="Supporting passages", interactive=False)
                            with gr.Accordion("Jump to another question that needs you", open=False):
                                review_q = gr.Dropdown(label="Questions that need you", choices=[], type="value")

                            @gr.render(inputs=[ctx, review_q])
                            def handwriting(context, question):
                                item = (context or {}).get("referrals", {}).get(question, {})
                                if "crop" not in item:
                                    return
                                key = f"{context['id']}-{question}"
                                button = gr.Button("Show the handwriting", key=f"crop-button-{key}", size="sm")
                                crops = gr.Gallery(label="Original scan lines", columns=1, height="auto", format="png",
                                                   interactive=False, key=f"crop-gallery-{key}")
                                wording = gr.Textbox(label="Disputed words in context", interactive=False, lines=4,
                                                     key=f"crop-text-{key}")
                                button.click(show_review_lines, [ctx, review_q], [crops, wording], api_name=False)

                        with gr.Column(elem_classes="desk-card done-card", visible=True):
                            approve = gr.Button("Sign off this paper", variant="primary", visible=False, size="lg")
                            with gr.Row(elem_classes="feedback-row"):
                                gr.HTML('<p class="feedback-q">Was this grading useful?</p>')
                                up = gr.Button("👍 Useful", size="sm")
                                down = gr.Button("👎 Not useful", size="sm")
                            feedback_status = gr.Markdown()

                with gr.Accordion("Signed record", open=False, visible=False) as signed_sheet:
                    signed = gr.JSON(label="Teacher-approved marks")
                with gr.Accordion("Mark sheet and grading evidence", open=False):
                    with gr.Tab("Mark sheet", visible=False) as sheet_tab:
                        teacher_marks = gr.Dataframe(headers=["Q", "Teacher mark"], type="array",
                                                     datatype=["number", "number"], interactive=True,
                                                     static_columns=[0], label="Edit any mark")
                    with gr.Tab("Grading evidence"):
                        evidence = gr.Dataframe(headers=["Q", "Question", "OCR reading", "Out of", "Proposal", "Decision",
                                                         "Feedback", "Source", "PDF page", "Source URL", "Supporting quote",
                                                         "Passage opened by model", "Evidence by rubric point"],
                                                interactive=False, label="Proposals and evidence", wrap=True, max_height=560,
                                                column_widths=["52px", "230px", "200px", "64px", "84px", "150px", "260px",
                                                               "170px", "72px", "180px", "260px", "200px", "280px"])

            outputs = [status, evidence, teacher_marks, ctx, signed]
            detail_outputs = [reading, standard, support, review_point, record_point]
            view_outputs = [progress_steps, band, answer_map]

            def refresh_after(event, *, scroll=False):
                event.success(results_view, [ctx, teacher_marks, review_q, signed, status],
                              [results, summary_card, review, review_q, sheet_tab, approve, signed_sheet],
                              scroll_to_output=scroll, api_name=False).success(
                                  question_view, [ctx, review_q], detail_outputs, api_name=False).success(
                                  paper_view, [ctx, teacher_marks, review_q, focus, signed], view_outputs, api_name=False)

            graded = go.click(display, [upload, library, course, exam_state], outputs, api_name="run",
                              show_progress_on=upload)
            refresh_after(graded.success(lambda: None, None, focus, api_name=False), scroll=True)
            graded.success(page_images, upload, paper_pages, api_name=False)
            # A sample paper grades as soon as it is clicked, like the examples of a Gradio demo, and shows its pages.
            picked = samples.select(grade_sample, [library, course, exam_state], outputs + [upload], api_name=False,
                                    show_progress_on=samples)
            refresh_after(picked.success(lambda: None, None, focus, api_name=False), scroll=True)
            picked.success(page_images, upload, paper_pages, api_name=False)
            # Once a paper is graded the start area folds away; "Grade another paper" brings it back.
            for event in (graded, picked):
                event.success(lambda: gr.update(visible=False), None, start_box, api_name=False)
            another.click(lambda: gr.update(visible=True), None, start_box, api_name=False)
            review_q.input(question_view, [ctx, review_q], detail_outputs, api_name=False).success(
                paper_view, [ctx, teacher_marks, review_q, focus, signed], view_outputs, api_name=False)
            review_point.input(lambda point: gr.update(interactive=point is not None),
                               review_point, record_point, api_name=False)
            refresh_after(record_point.click(apply_review_mark, [ctx, review_q, review_point, teacher_marks],
                                             [teacher_marks, review_q, review_point, signed], api_name=False))
            # The answer map: a click opens that question in the review panel or in the check panel.
            answer_map.change(open_question, [ctx, answer_map, teacher_marks],
                              [focus, review_q, check_panel, check_card, check_point], api_name=False).success(
                question_view, [ctx, review_q], detail_outputs, api_name=False).success(
                paper_view, [ctx, teacher_marks, review_q, focus, signed], view_outputs, api_name=False)
            refresh_after(check_save.click(apply_check_mark, [ctx, focus, check_point, teacher_marks],
                                           [teacher_marks, focus], api_name=False).success(
                lambda: (gr.update(visible=False), None), None, [check_panel, signed], api_name=False))
            check_close.click(lambda: (gr.update(visible=False), None), None, [check_panel, focus], api_name=False).success(
                paper_view, [ctx, teacher_marks, review_q, focus, signed], view_outputs, api_name=False)
            refresh_after(approve.click(sign_off, [ctx, teacher_marks], signed, api_name=False), scroll=True)
            ctx.change(lambda: "", None, feedback_status, api_name=False)
            up.click(lambda c: rate(c, 1), ctx, feedback_status, api_name=False)
            down.click(lambda c: rate(c, -1), ctx, feedback_status, api_name=False)
            refresh_after(teacher_marks.input(lambda: None, None, signed, api_name=False))

        # ------------------------------------------------------------ 2. Materials
        with gr.Tab("Materials", id="materials"):
            gr.HTML('<div class="tab-intro"><h1>Course materials</h1><p>Choose the exam and the sources the written '
                    'answers are graded against.</p></div>')
            with gr.Row(elem_classes="desk-columns"):
                with gr.Column(scale=3, min_width=280, elem_classes="desk-card"):
                    gr.HTML('<h2 class="card-title">Add a course PDF</h2>')
                    name = gr.Textbox(label="Course name", value="My ML course")
                    material = gr.File(label="Course PDF", file_types=[".pdf"], type="filepath", height=130)
                    gr.HTML('<p class="card-note">With semantic search on, the passages are sent to Gemini to be indexed.</p>')
                    add = gr.Button("Index PDF and select course", variant="primary")
                    indexed = gr.Textbox(label="Indexing result", interactive=False)
                with gr.Column(scale=2, min_width=250, elem_classes=["desk-card", "desk-settings"]):
                    gr.HTML('<h2 class="card-title">Choose a marking guide</h2>')
                    exam_file = gr.File(label="Exam definition (JSON)", file_types=[".json"], type="filepath", height=130)
                    load_exam = gr.Button("Apply exam definition")
                    material_exam = gr.Textbox(label="Active exam", value="Built-in ML exam selected.", interactive=False)
            with gr.Accordion("Find a supporting passage", open=False):
                material_course = gr.Dropdown([BUILTIN], value=BUILTIN, label="Course materials")
                question = gr.Dropdown([(f"Q{q}. {QUESTIONS[q]}", q) for q in range(21, 36)], value=21, label="Exam question")
                search = gr.Button("Find supporting passages")
                passages = gr.Dataframe(headers=["Source", "PDF page", "Passage", "Score"],
                                        label="Passages for this question", interactive=False, wrap=True)
            add.click(index_material, [material, name, library], [library, course, indexed, passages], api_name="index_material")
            course.change(lambda lib, active: gr.update(choices=[BUILTIN, *lib], value=active),
                          [library, course], material_course, api_name=False)
            material_course.input(lambda value: value, material_course, course, api_name=False)
            material_exam.change(lambda value: value.split(", reader exam_id: ")[0],
                                 material_exam, exam_info, api_name=False)
            refresh_after(load_exam.click(select_exam, [exam_file], [exam_state, material_exam, question]).success(
                lambda: ("", [], [], None, None, []), None, [*outputs, passages], api_name=False))
            search.click(preview_material, [question, library, material_course, exam_state], passages, api_name="find_passages")
            course.change(lambda: [], None, passages, api_name=False)
            question.change(lambda: [], None, passages, api_name=False)

        # ------------------------------------------------------------ 3. Monitoring
        with gr.Tab("Monitoring", id="monitoring"):
            with gr.Row(elem_classes="tab-head"):
                gr.HTML('<div class="tab-intro"><h1>Monitoring</h1><p>Is the product still seeing the papers it was '
                        'tested on, and are teachers finding it useful?</p></div>')
                refresh = gr.Button("Refresh", size="sm", elem_classes="refresh-btn")
            banner = gr.HTML()
            kpis = gr.HTML()
            with gr.Row(elem_classes="desk-columns"):
                with gr.Column(scale=3, elem_classes="desk-card"):
                    gr.HTML('<h2 class="card-title">Quality over time</h2><p class="card-note">Share of answers sent '
                            'to review, and share of 👎 among rated papers, per hour.</p>')
                    quality_plot = gr.LinePlot(x="time", y="value", color="signal", show_label=False, height=260,
                                               y_lim=[0, 1])
                with gr.Column(scale=2, elem_classes="desk-card"):
                    gr.HTML('<h2 class="card-title">Input drift</h2><p class="card-note">Population stability index '
                            'of each scan signal over the last 100 papers against the reference set. An alert fires '
                            'above 0.25 and above twice what chance gives at this sample size.</p>')
                    drift_plot = gr.BarPlot(x="feature", y="psi", show_label=False, height=260, sort="-y")
            with gr.Column(elem_classes="desk-card"):
                gr.HTML('<h2 class="card-title">Review rule (learning from sign-offs)</h2><p class="card-note">For '
                        'each adaptive run the app picks which written marks from Gemini go to the teacher instead of '
                        'arriving pre-filled, by Thompson sampling. The cost per answer counts each review as 1 and each proposal the teacher '
                        'changed as 5: lower is better. Papers graded counts confirmed rule selections. '
                        'Warm-start papers are simulated; signed-off papers are real.</p>')
                rule_plot = gr.BarPlot(x="rule", y="cost", y_title="cost per answer", show_label=False, height=220)
                learning = gr.Dataframe(label="Review rules", interactive=False)
            with gr.Accordion("All numbers", open=False):
                summary = gr.Dataframe(label="Request summary", interactive=False)
                timeline = gr.Dataframe(label="Hourly quality signals", interactive=False)
                drift_table = gr.Dataframe(label="Drift per signal", interactive=False)
            monitor_outputs = [banner, kpis, quality_plot, drift_plot, rule_plot, summary, timeline]

            def refresh_monitoring():
                banner_html, tiles_html, quality, drift, costs, frame, hourly = monitoring_view()
                return banner_html, tiles_html, quality, drift, costs, frame, hourly, drift, costs.drop(columns="cost")

            refresh.click(refresh_monitoring, None, [*monitor_outputs, drift_table, learning], api_name=False)
            demo.load(refresh_monitoring, None, [*monitor_outputs, drift_table, learning], api_name=False)
    return demo

demo = build_app()
def launch_app(**options):
    # Fonts come from app.css: Gradio 6.31 cannot compare a theme with custom font objects to its built-in themes.
    theme = gr.themes.Base(primary_hue="emerald", secondary_hue="teal", neutral_hue="stone")
    return demo.launch(theme=theme, css_paths=Path(__file__).with_name("app.css"), head=ui.HEAD, **options)


if __name__ == "__main__":
    print(logbook.start())
    # Render gives the port in $PORT; locally it is 7860.
    launch_app(server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
