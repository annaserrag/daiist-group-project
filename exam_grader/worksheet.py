"""Local human source review; run python -m exam_grader.worksheet. No model or API key."""
import json
import re
import gradio as gr
from .annotations import PAGES, SHEET, decision, read, save
from .catalog import DATA, KEY, text

CORPUS=json.loads((DATA/"corpus.json").read_bytes())
BLOCKS={int(q):body.split("Part II:")[0].strip() for q,body in
        re.findall(r"(?ms)^(\d+)\. (.*?)(?=^\d+\. |\Z)",text)}
ORDER=[*range(21,36),*range(1,21)]  # written answers use retrieval at runtime
CHOICES=sorted([(title+" ["+sid+"]",sid) for sid,title in PAGES.items()])
GUIDE=json.loads(SHEET.with_name("retrieval_reading_guide.json").read_bytes())

def passages(page,query):
    words=(query or "").casefold().split()
    found=[]
    for r in CORPUS:
        if page and r["source"]!=page:
            continue
        if not page and not words:
            return "Choose a page or enter a keyword to read the frozen index."
        if all(w in (r["heading"]+" "+r["title"]+" "+r["text"]).casefold() for w in words):
            found.append(f'{r["source"]} | {r["heading"]}\n{r["text"]}\n{r["url"]}')
    return "\n\n".join(found) or "No matching indexed paragraphs. Try a shorter keyword or another page."

def load_row(q):
    rows=read()
    by_q={int(r["question_id"]):decision(r) for r in rows}
    status,ids,note=by_q[q]
    progress=sum(s!="open" for s,_,_ in by_q.values())
    return (f"Q{q}. {BLOCKS[q]}\n\nPublished key: {KEY[q]}", status,list(ids),note,
            (q,status,ids,note),f"{progress}/35 reviewed · {35-progress} open",None,"",passages(None,""),
            "Start with "+GUIDE[str(q)][0]+". "+GUIDE[str(q)][1])

def commit(q,state,ids,note,seen):
    if not seen or seen[0]!=q:
        raise gr.Error("Reload this question before saving.")
    try:
        save(q,state,ids,note,(seen[1],tuple(seen[2]),seen[3]))
    except ValueError as exc:
        raise gr.Error(str(exc)) from exc
    return load_row(q)

def next_open(q):
    order=ORDER[ORDER.index(q)+1:]+ORDER[:ORDER.index(q)+1]
    pending={int(r["question_id"]) for r in read() if decision(r)[0]=="open"}
    return next((i for i in order if i in pending),q)

def build():
    initial=next_open(20)
    with gr.Blocks(title="Retrieval worksheet") as ui:
        gr.Markdown("# Retrieval worksheet\nRead the question and key, then check the **indexed paragraphs**. "
                    "A page counts if it explains the answer. A shared keyword or a matching title alone does not count. "
                    "You can accept several pages. Use **gap** for missing teaching evidence, **missing_context** for an absent part of the exam (such as a figure), or **open** if unsure.")
        with gr.Row():
            q=gr.Dropdown([(f"Q{i} · {'written' if i>20 else 'multiple choice'}",i) for i in ORDER],
                          value=initial,label="Question",allow_custom_value=False)
            progress=gr.Textbox(label="Worksheet progress",interactive=False)
        prompt=gr.Textbox(label="Published question and answer",interactive=False,lines=5)
        with gr.Accordion("Optional: read the assistant’s suggestion",open=False):
            hint=gr.Textbox(label="Reading suggestion · never saved as a label",interactive=False,lines=3)
        with gr.Row():
            page=gr.Dropdown(CHOICES,label="Read a page (alphabetical)",value=None)
            query=gr.Textbox(label="Find passages by keyword",placeholder="e.g. generalization; clear the page to search all sources")
        excerpts=gr.Textbox(label="Indexed passages and original links",lines=8,interactive=False)
        page.input(passages,[page,query],excerpts)
        query.input(passages,[page,query],excerpts)
        state=gr.Radio(["supported","gap","missing_context","open"],label="Your decision",value="open")
        ids=gr.Dropdown(CHOICES,multiselect=True,value=[],label="Acceptable sources (for supported only)")
        note=gr.Textbox(label="Why? Required for a gap or missing exam context.",
                        placeholder="Which fact does the passage establish, or what evidence is missing?")
        with gr.Row():
            store=gr.Button("Save this decision",variant="primary")
            advance=gr.Button("Go to next open question")
        gr.Markdown("Save before changing questions. Saved decisions stay in eval/retrieval_sources.csv on this computer. "
                    "The worksheet does not send exam data or show the retriever’s rankings.")
        seen=gr.State()
        controls=[prompt,state,ids,note,seen,progress,page,query,excerpts,hint]
        ui.load(load_row,q,controls)
        q.change(load_row,q,controls)
        store.click(commit,[q,state,ids,note,seen],controls)
        advance.click(next_open,q,q)
    return ui

if __name__=="__main__":
    build().launch(server_name="127.0.0.1",server_port=7861,share=False,footer_links=[])
