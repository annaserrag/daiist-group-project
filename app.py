"""Local Component 2 workbench and course-compatible Gradio app."""
import math
from time import perf_counter

import gradio as gr
import pandas as pd

import logbook
import monitor
import pipeline
from exam_grader.catalog import QUESTIONS
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
    "context_check": "Consult the original question and its accompanying context before marking.",
    "weak_retrieval": "Use the official key and check a teaching passage that supports this answer.",
    "no_index": "Open the official key or load the course index before grading.",
    "provider_disabled": "Grade from the key, or run the written grader on an approved input.",
    "provider_failure": "The written grader could not finish. Retry or mark from the key.",
    "lookup_limit": "The grader requested extra evidence beyond its lookup limit. Consult the course source.",
    "model_call_limit": "The grader reached its call limit. Inspect the answer against the key.",
    "provider_rate_limit": "The grading service reached its request limit. Retry later or mark manually.",
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

def process_once(raw_input, course_index=None):
    started = perf_counter()
    views = pipeline.run_views(raw_input, course_index=course_index)
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


def preview_material(question, library, course=BUILTIN):
    index = selected_index(library, course) or pipeline.resources()[0]
    return [[hit["title"], hit.get("page"), hit["text"], round(hit["score"], 3)]
            for hit in index.search(int(question))]


def display(raw_input, library=None, course=BUILTIN):
    if not raw_input:
        raise gr.Error("Choose an OCR export or a PDF first.")
    teacher, record = process_once(raw_input, selected_index(library, course))
    evidence, marks = [], []
    for item in teacher["answers"]:
        source = item["citation"] or {}
        feedback = item["feedback"] or REVIEW_ACTIONS.get(item.get("execution_issue",item["reason"]), "")
        evidence.append([item["question_id"], item["question"], " | ".join(item["readings"]), item["max_mark"], item["mark"],
                         item["reason"].replace("_", " "), feedback,
                         source.get("title", ""), source.get("page"), source.get("url", ""), source.get("quote", ""),
                         ", ".join(item.get("opened_passages",[])) or "",
                         "\n".join(f"{criterion} ({c['coverage']}) · {c['title']}"
                                   + (f", p. {c['page']}" if "page" in c else "")
                                   + f" · {c['quote']}" for criterion,c in item.get("criterion_evidence",{}).items())])
        marks.append([item["question_id"], item["mark"]])
    count = teacher["review_count"]
    if teacher["input_type"] == "pdf_awaiting_reader":
        status = "PDF received. Connect the team's vision reader to extract these answers."
    elif teacher.get("reader_failed"):
        status = "The vision reader failed. No exam total available; retry or inspect the reader."
    elif teacher["exam_status"] != "in_scope":
        status = f"Exam status: {teacher['exam_status'].replace('_',' ')}. No exam total available."
    else:
        status = f"{35-count} of 35 questions have a proposed mark. {count} require review."
        if teacher["near_pass_boundary"]:
            status += " Check the total against the configured pass boundary."
    context = {"id": record["id"], "in_scope": teacher["exam_status"] == "in_scope",
               "review_started":perf_counter(), "proposals":[point for _,point in marks]}
    return status, evidence, marks, context, None

def sign_off(context, marks):
    if not context or not context["in_scope"]:
        raise gr.Error("Run an in-scope exam before signing off.")
    if len(marks) != 35:
        raise gr.Error("The final sheet needs all 35 question rows.")
    final_marks = []
    for question, values in enumerate(marks, start=1):
        maximum = 1 if question <= 20 else 2
        try:
            q, point = values
            if type(q) is bool or type(point) is bool or float(q) != question:
                raise ValueError()
            point = float(point)
            if not math.isfinite(point) or point < 0 or point > maximum or (point if maximum == 1 else point * 2) % 1:
                raise ValueError()
        except (TypeError, ValueError, OverflowError) as exc:
            raise gr.Error(f"Check Q{question}. Enter a valid teacher mark from 0 to {maximum}.") from exc
        final_marks.append({"question_id": question, "mark": point})
    signed = {"request_id": context["id"], "teacher_signed_off": True, "maximum": 50,
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

def build_app():
    with gr.Blocks(title="Exam grading assistant") as demo:
        gr.Markdown("# Exam grading assistant")
        library = gr.State({})
        course = gr.Dropdown([BUILTIN], value=BUILTIN, label="Course materials used for grading")
        with gr.Tab("App"):
            upload = gr.File(label="Exam PDF or vision JSON", file_types=[".pdf",".json"], type="filepath")
            with gr.Row():
                go = gr.Button("Grade uploaded exam", variant="primary")
                example = gr.Button("Try synthetic OCR example")
            status = gr.Textbox(label="Grading status", interactive=False)
            evidence = gr.Dataframe(headers=["Q","Question","OCR reading","Out of","Proposal","Decision","Feedback","Source","PDF page","Source URL","Supporting quote","Passage opened by model","Evidence by rubric point"],
                                    interactive=False, label="Proposals and evidence")
            teacher_marks = gr.Dataframe(headers=["Q","Teacher mark"], type="array", datatype=["number","number"],
                                         interactive=True, label="Teacher marks · complete empty cells before sign-off")
            ctx = gr.State(None)
            approve = gr.Button("Sign off all 35 marks")
            signed = gr.JSON(label="Final teacher-approved sheet")
            outputs = [status,evidence,teacher_marks,ctx,signed]
            go.click(display,[upload,library,course],outputs,api_name="run")
            example.click(lambda lib, active: display(pipeline.EXAMPLE, lib, active),[library,course],outputs,api_name="grade_example")
            approve.click(sign_off,[ctx,teacher_marks],signed,api_name=False)
            with gr.Row():
                up = gr.Button("Helpful")
                down = gr.Button("Unhelpful")
            feedback_status = gr.Textbox(label="Feedback",interactive=False)
            up.click(lambda c: rate(c,1),ctx,feedback_status,api_name=False)
            down.click(lambda c: rate(c,-1),ctx,feedback_status,api_name=False)
            teacher_marks.input(lambda: None,None,signed,api_name=False)
        with gr.Tab("Materials"):
            gr.Markdown("Add searchable course PDFs for this browser session. With semantic retrieval enabled, "
                        "indexing sends course passages to Gemini for embeddings. Page numbers start at the PDF's first page.")
            name = gr.Textbox(label="Course to add to", value="My ML course")
            material = gr.File(label="Course PDF", file_types=[".pdf"], type="filepath")
            add = gr.Button("Index PDF and select course", variant="primary")
            indexed = gr.Textbox(label="Indexing result", interactive=False)
            question = gr.Dropdown([(f"Q{q}. {QUESTIONS[q]}", q) for q in range(21,36)],
                                   value=21, label="Exam question")
            search = gr.Button("Find supporting passages")
            passages = gr.Dataframe(headers=["Source","PDF page","Passage","Score"],
                                    label="Top passages for the selected course and question", interactive=False)
            add.click(index_material,[material,name,library],[library,course,indexed,passages],api_name="index_material")
            search.click(preview_material,[question,library,course],passages,api_name="find_passages")
            course.change(lambda: [],None,passages,api_name=False)
            question.change(lambda: [],None,passages,api_name=False)
        with gr.Tab("Monitoring"):
            refresh = gr.Button("Refresh monitoring")
            alert = gr.Textbox(label="Drift alert",interactive=False)
            summary = gr.Dataframe(label="Request summary",interactive=False)
            timeline = gr.Dataframe(label="Hourly quality signals",interactive=False)
            refresh.click(monitoring,None,[alert,summary,timeline],api_name=False)
            demo.load(monitoring,None,[alert,summary,timeline],api_name=False)
    return demo

demo = build_app()
if __name__ == "__main__":
    logbook.start()
    demo.launch()
