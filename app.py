"""Local Component 2 workbench and course-compatible Gradio app."""
import math
from time import perf_counter

import gradio as gr
import pandas as pd

import logbook
import monitor
import pipeline

def process_once(raw_input):
    started = perf_counter()
    views = pipeline.run_views(raw_input)
    record = views["record"]
    record["latency_s"] = round(perf_counter() - started, 4)
    return views["teacher"], logbook.log_request(record)

def serve(raw_input):
    return process_once(raw_input)[1]

def display(raw_input):
    if not raw_input:
        raise gr.Error("Choose an OCR export or a PDF first.")
    teacher, record = process_once(raw_input)
    evidence, marks = [], []
    for item in teacher["answers"]:
        source = item["citation"] or {}
        evidence.append([item["question_id"], item["question"], " | ".join(item["readings"]), item["max_mark"], item["mark"],
                         item["reason"].replace("_", " "), item["feedback"],
                         source.get("title", ""), source.get("url", ""), source.get("quote", ""),
                         ", ".join(item.get("opened_passages",[])) or ""])
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
    context = {"id": record["id"], "in_scope": teacher["exam_status"] == "in_scope"}
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
    return {"request_id": context["id"], "teacher_signed_off": True, "maximum": 50,
            "total": sum(row["mark"] for row in final_marks), "marks": final_marks}

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
        with gr.Tab("App"):
            upload = gr.File(label="Exam PDF or vision JSON", file_types=[".pdf",".json"], type="filepath")
            with gr.Row():
                go = gr.Button("Grade uploaded exam", variant="primary")
                example = gr.Button("Try synthetic OCR example")
            status = gr.Textbox(label="Grading status", interactive=False)
            evidence = gr.Dataframe(headers=["Q","Question","OCR reading","Out of","Proposal","Decision","Feedback","Source","Source URL","Supporting quote","Passage opened by model"],
                                    interactive=False, label="Proposals and evidence")
            teacher_marks = gr.Dataframe(headers=["Q","Teacher mark"], type="array", datatype=["number","number"],
                                         interactive=True, label="Teacher marks · complete empty cells before sign-off")
            ctx = gr.State(None)
            approve = gr.Button("Sign off all 35 marks")
            signed = gr.JSON(label="Final teacher-approved sheet")
            outputs = [status,evidence,teacher_marks,ctx,signed]
            go.click(display,upload,outputs,api_name="run")
            example.click(lambda: display(pipeline.EXAMPLE),None,outputs,api_name=False)
            approve.click(sign_off,[ctx,teacher_marks],signed,api_name=False)
            with gr.Row():
                up = gr.Button("Helpful")
                down = gr.Button("Unhelpful")
            feedback_status = gr.Textbox(label="Feedback",interactive=False)
            up.click(lambda c: rate(c,1),ctx,feedback_status,api_name=False)
            down.click(lambda c: rate(c,-1),ctx,feedback_status,api_name=False)
            teacher_marks.input(lambda: None,None,signed,api_name=False)
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
