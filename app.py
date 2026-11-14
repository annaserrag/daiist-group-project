"""Local Component 2 workbench and course-compatible Gradio app."""
from html import escape
import math
import os
from pathlib import Path
from subprocess import SubprocessError
from time import perf_counter

import gradio as gr
import pandas as pd

import logbook
import monitor
import pipeline
from exam_grader.catalog import KEY, OPTION_TEXTS, QUESTIONS
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

def process_once(raw_input, course_index=None, exam_state=None):
    started = perf_counter()
    exam, queries = exam_state or (None, None)
    views = pipeline.run_views(raw_input, course_index=course_index, written_exam=exam, query_vectors=queries)
    record = views["record"]
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
    return (exam, queries), f"{exam.title} · Reader exam_id: {exam.exam_id}", gr.Dropdown(choices=choices, value=choices[0][1])


def preview_material(question, library, course=BUILTIN, exam_state=None):
    index = selected_index(library, course) or pipeline.resources()[0]
    if exam_state:
        exam, queries = exam_state
        index = index.for_questions({q.question_id:q.question for q in exam.questions}, queries, exam.exam_id)
    return [[hit["title"], hit.get("page"), hit["text"], round(hit["score"], 3)]
            for hit in index.search(int(question))]


def display(raw_input, library=None, course=BUILTIN, exam_state=None):
    if not raw_input:
        raise gr.Error("Choose an OCR export or a PDF first.")
    teacher, record = process_once(raw_input, selected_index(library, course), exam_state)
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
                         "\n".join(f"{criterion} ({c['coverage']}) · {c['title']}"
                                   + (f", p. {c['page']}" if "page" in c else "")
                                   + f" · {c['quote']}" + (f" · Example check: {c['basis']}" if "basis" in c else "")
                                   for criterion,c in item.get("criterion_evidence",{}).items())])
        q = item["question_id"]
        if teacher["exam_status"] == "in_scope" and item["mark"] is None:
            guide = custom[q].guide() if custom else CRITERIA.get(q, {})
            key = custom[q].reference_answer if custom else KEY[q]
            rubric = [f"{name} ({points:g} points) · {standard}" for name, (points, standard) in guide.items()]
            proofs = item.get("criterion_evidence", {})
            sources = []
            for name, cite in proofs.items():
                location = cite["title"] + (f", p. {cite['page']}" if "page" in cite else "")
                sources.append(f"{name} ({cite['coverage']}) · {location} · {cite['quote']}")
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
                    location += f" · {hit['url']}"
                passages.append(f"{location}\n{hit['excerpt']}")
            # The teacher can assess a large deduction independently. The original
            # model suggestion stays available in the full evidence table.
            action = ("Assess each rubric point before entering a mark."
                      if item["reason"] == "written_mark_review" else feedback)
            choices = {} if custom else OPTION_TEXTS.get(q, {})
            if key in choices:
                key = f"{key} · {choices[key]}"
            if choices and item["reason"] == "invalid_choice":
                action = "Check the original paper. The reader found a choice outside " + ", ".join(
                    f"{letter} ({value})" for letter, value in choices.items()) + "."
            referrals[q] = dict(question=item["question"], reading=" | ".join(item["readings"]),
                                key=key, rubric=rubric, sources=sources, passages=passages, action=action,
                                reason=item["reason"].replace("_", " "))
        marks.append([q, item["mark"]])
    count = teacher["review_count"]
    if teacher["input_type"] == "pdf_awaiting_reader":
        status = "PDF received. Connect the team's vision reader to extract these answers."
    elif teacher.get("reader_failed"):
        status = "The vision reader failed. No exam total available; retry or inspect the reader."
    elif teacher["exam_status"] != "in_scope":
        status = f"Exam status: {teacher['exam_status'].replace('_',' ')}. No exam total available."
    else:
        status = f"{len(marks)-count} of {len(marks)} questions have a proposed mark. {count} require review."
        if teacher["near_pass_boundary"]:
            status += " Check the total against the configured pass boundary."
    context = {"id": record["id"], "in_scope": teacher["exam_status"] == "in_scope",
               "near_boundary": teacher.get("near_pass_boundary", False),
               "limits":[(r["question_id"],r["max_mark"]) for r in teacher["answers"]],
               "referrals":referrals, "review_started":perf_counter(), "proposals":[point for _,point in marks]}
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
            gr.update(choices=choices, value=None, label=f"Your mark · out of {maximum:g}"),
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
        return message, frame, pd.DataFrame(columns=columns)
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
    return message, frame, pd.DataFrame(rows, columns=columns)

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
        detail = f"{total:g} / {sum(limit for _, limit in limits):g} points · {len(limits)} questions marked."
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


def build_app():
    with gr.Blocks(title="Exam desk · Exam grading assistant") as demo:
        gr.HTML('<header class="desk-mast"><span>Exam desk</span></header>')
        library = gr.State({})
        exam_state = gr.State(None)
        ctx = gr.State(None)
        with gr.Tab("Grade exams"):
            gr.Markdown("# Grade a paper\nUpload a student’s work, then review the suggested marks.", elem_classes="desk-intro")
            with gr.Row(elem_classes="desk-columns"):
                with gr.Column(scale=3, min_width=280, elem_classes="desk-card"):
                    upload = gr.File(label="Student paper · PDF or reader JSON", file_types=[".pdf", ".json"],
                                     type="filepath", height=160, elem_id="paper-drop")
                    with gr.Row(elem_classes="desk-actions"):
                        go = gr.Button("Grade paper", variant="primary")
                        example = gr.Button("Try a sample paper")
                with gr.Column(scale=2, min_width=250, elem_classes=["desk-card", "desk-settings"]):
                    gr.Markdown("## Marking setup")
                    exam_info = gr.Textbox(label="Exam", value="Built-in ML exam selected.", interactive=False)
                    course = gr.Dropdown([BUILTIN], value=BUILTIN, label="Course materials")
                    gr.Markdown("Use the Materials tab to add a marking guide or course PDF.")
            with gr.Column(visible=False, elem_id="grading-results") as results:
                status = gr.Textbox(visible=False)
                summary_card = gr.HTML()
                with gr.Column(visible=False, elem_classes="desk-card", elem_id="active-review") as review:
                    with gr.Accordion("Jump to another unmarked question", open=False):
                        review_q = gr.Dropdown(label="Unmarked questions", choices=[], type="value")
                    reading = gr.HTML()
                    with gr.Accordion("See rubric and sources", open=False):
                        standard = gr.Textbox(label="Marking guide", interactive=False)
                        support = gr.Textbox(label="Supporting passages", interactive=False)
                    @gr.render(inputs=[ctx, review_q])
                    def crop_panel(context, question):
                        item = (context or {}).get("referrals", {}).get(question, {})
                        if "crop" not in item:
                            return
                        # A newly selected referral gets its own output components, including during a slow OCR call.
                        key = f"{context['id']}-{question}"
                        button = gr.Button("Show disputed handwriting", key=f"crop-button-{key}")
                        crops = gr.Gallery(label="Original scan lines", columns=1, height="auto", format="png",
                                           interactive=False, key=f"crop-gallery-{key}")
                        wording = gr.Textbox(label="Disputed words in context", interactive=False, lines=4,
                                             key=f"crop-text-{key}")
                        button.click(show_review_lines, [ctx, review_q], [crops, wording], api_name=False)

                    with gr.Row(elem_classes="mark-actions"):
                        review_point = gr.Radio(label="Your mark", choices=[], interactive=True)
                        record_point = gr.Button("Save & continue", variant="primary", interactive=False)
                approve = gr.Button("Sign off this paper", variant="primary", visible=False)
                with gr.Accordion("View signed record", open=False, visible=False) as signed_sheet:
                    signed = gr.JSON(label="Teacher-approved marks")
                with gr.Accordion("Mark sheet and paper details", open=False):
                    with gr.Tab("Mark sheet", visible=False) as sheet_tab:
                        teacher_marks = gr.Dataframe(headers=["Q", "Teacher mark"], type="array",
                                                     datatype=["number", "number"], interactive=True,
                                                     static_columns=[0],
                                                     label="Edit a mark")
                    with gr.Tab("Grading evidence"):
                        evidence = gr.Dataframe(headers=["Q", "Question", "OCR reading", "Out of", "Proposal", "Decision",
                                                         "Feedback", "Source", "PDF page", "Source URL", "Supporting quote",
                                                         "Passage opened by model", "Evidence by rubric point"],
                                                interactive=False, label="Proposals and evidence")
                    with gr.Tab("Give feedback"):
                        with gr.Row():
                            up = gr.Button("Helpful")
                            down = gr.Button("Unhelpful")
                        feedback_status = gr.Markdown()
            outputs = [status,evidence,teacher_marks,ctx,signed]
            detail_outputs = [reading, standard, support, review_point, record_point]

            def refresh_after(event, *, scroll=False):
                event.success(results_view, [ctx, teacher_marks, review_q, signed, status],
                              [results, summary_card, review, review_q, sheet_tab, approve, signed_sheet],
                              scroll_to_output=scroll, api_name=False).success(
                                  question_view, [ctx, review_q], detail_outputs, api_name=False)

            refresh_after(go.click(display, [upload, library, course, exam_state], outputs, api_name="run",
                                   show_progress_on=upload), scroll=True)
            refresh_after(example.click(lambda lib, active, state: display(pipeline.EXAMPLE, lib, active, state),
                                        [library, course, exam_state], outputs, api_name="grade_example",
                                        show_progress_on=upload), scroll=True)
            review_q.input(question_view, [ctx, review_q], detail_outputs, api_name=False)
            review_point.input(lambda point: gr.update(interactive=point is not None),
                               review_point, record_point, api_name=False)
            refresh_after(record_point.click(apply_review_mark, [ctx,review_q,review_point,teacher_marks],
                               [teacher_marks,review_q,review_point,signed], api_name=False))
            refresh_after(approve.click(sign_off,[ctx,teacher_marks],signed,api_name=False), scroll=True)
            ctx.change(lambda: "", None, feedback_status, api_name=False)
            up.click(lambda c: rate(c,1),ctx,feedback_status,api_name=False)
            down.click(lambda c: rate(c,-1),ctx,feedback_status,api_name=False)
            refresh_after(teacher_marks.input(lambda: None, None, signed, api_name=False))
        with gr.Tab("Materials"):
            gr.Markdown("# Course materials\nChoose the sources and marking guide for this session.", elem_classes="desk-intro")
            with gr.Row(elem_classes="desk-columns"):
                with gr.Column(scale=3, min_width=280, elem_classes="desk-card"):
                    gr.Markdown("## Add a course PDF")
                    name = gr.Textbox(label="Course name", value="My ML course")
                    material = gr.File(label="Course PDF", file_types=[".pdf"], type="filepath", height=130)
                    gr.Markdown("When semantic search is enabled, passages are sent to Gemini for indexing.")
                    add = gr.Button("Index PDF and select course", variant="primary")
                    indexed = gr.Textbox(label="Indexing result", interactive=False)
                with gr.Column(scale=2, min_width=250, elem_classes=["desk-card", "desk-settings"]):
                    gr.Markdown("## Choose a marking guide")
                    exam_file = gr.File(label="Exam definition · JSON", file_types=[".json"], type="filepath", height=130)
                    load_exam = gr.Button("Apply exam definition")
                    material_exam = gr.Textbox(label="Active exam", value="Built-in ML exam selected.", interactive=False)
            with gr.Accordion("Find a supporting passage", open=False):
                material_course = gr.Dropdown([BUILTIN], value=BUILTIN, label="Course materials")
                question = gr.Dropdown([(f"Q{q}. {QUESTIONS[q]}", q) for q in range(21,36)], value=21, label="Exam question")
                search = gr.Button("Find supporting passages")
                passages = gr.Dataframe(headers=["Source", "PDF page", "Passage", "Score"],
                                        label="Passages for this question", interactive=False)
            add.click(index_material,[material,name,library],[library,course,indexed,passages],api_name="index_material")
            course.change(lambda lib, active: gr.update(choices=[BUILTIN, *lib], value=active),
                          [library, course], material_course, api_name=False)
            material_course.input(lambda value: value, material_course, course, api_name=False)
            material_exam.change(lambda value: value.split(" · Reader exam_id: ")[0],
                                 material_exam, exam_info, api_name=False)
            refresh_after(load_exam.click(select_exam,[exam_file],[exam_state,material_exam,question]).success(
                lambda: ("", [], [], None, None, []), None, [*outputs, passages], api_name=False))
            search.click(preview_material,[question,library,material_course,exam_state],passages,api_name="find_passages")
            course.change(lambda: [],None,passages,api_name=False)
            question.change(lambda: [],None,passages,api_name=False)
        with gr.Tab("Monitoring"):
            gr.Markdown("# Monitoring\nCheck scan quality and review activity across graded papers.", elem_classes="desk-intro")
            refresh = gr.Button("Refresh monitoring")
            alert = gr.Textbox(label="Drift alert",interactive=False)
            summary = gr.Dataframe(label="Request summary",interactive=False)
            with gr.Accordion("Hourly activity", open=False):
                timeline = gr.Dataframe(label="Hourly quality signals",interactive=False)
            refresh.click(monitoring,None,[alert,summary,timeline],api_name=False)
            demo.load(monitoring,None,[alert,summary,timeline],api_name=False)
    return demo

demo = build_app()
def launch_app(**options):
    theme = gr.themes.Base(primary_hue="emerald", secondary_hue="teal", neutral_hue="stone",
                           font=["Source Sans 3", "Segoe UI", "sans-serif"])
    return demo.launch(theme=theme, css_paths=Path(__file__).with_name("app.css"), **options)


if __name__ == "__main__":
    logbook.start()
    launch_app()
