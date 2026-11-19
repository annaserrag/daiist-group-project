"""
The presentation layer of Exam grader: the pieces of HTML the app shows, kept apart from the grading logic in app.py.

In ten seconds: a teacher drops in a student's scanned exam, sees every question at once on an answer map, and steps
only through the answers the product was not sure of (reading next to the key) until the paper is signed off.

Design rules (vision/README and the course's "not a generic template" advice):
- one dominant visual idea, the answer map: 35 questions, each showing who decides its mark;
- headings in a serif, text in a sans, numbers in tabular figures; colour only from the tokens in app.css;
- state is never shown by colour alone: every cell also carries a word and a symbol;
- keyboard first: every cell is a real button with a label, focus is always visible.
"""

from __future__ import annotations

from html import escape

import gradio as gr

HEAD = ('<link rel="preconnect" href="https://fonts.googleapis.com">'
        '<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>'
        '<link href="https://fonts.googleapis.com/css2?family=Source+Serif+4:opsz,wght@8..60,400;8..60,600&'
        'family=Source+Sans+3:wght@400;600;700&display=swap" rel="stylesheet">'
        '<script>document.addEventListener("keydown", (e) => {'
        'const t = e.target; if (t && (t.closest("input, textarea, select, [contenteditable]"))) return;'
        'const card = document.querySelector("#active-review"); if (!card || !card.offsetParent) return;'
        'const pills = card.querySelectorAll("#review-point label");'
        'if (/^[0-9]$/.test(e.key) && pills[Number(e.key)]) { pills[Number(e.key)].click(); e.preventDefault(); }'
        'else if (e.key === "Enter") { const b = document.querySelector("#save-next"); if (b && !b.disabled) { b.click(); e.preventDefault(); } }'
        '});</script>')

# What each state of a question means to the teacher, with a symbol so it never depends on colour.
STATES = {
    "review": ("Needs your mark", "?"),
    "teacher": ("Marked by you", "✎"),
    "proposed": ("Proposed, unchanged", "✓"),
    "changed": ("You changed the proposal", "↺"),
    "absent": ("Not on this paper", "–"),
}


def masthead(writing_on: bool) -> str:
    grader = ("Written answers are graded with cited course passages." if writing_on
              else "Written answers go to you: no grading service is configured.")
    return ('<header class="desk-mast"><div class="mast-brand"><span class="wordmark">Exam desk</span>'
            '<span class="mast-tag">Marks a scanned exam with you, not for you.</span></div>'
            f'<p class="mast-status">{escape(grader)}</p></header>')


def stepper(stage: int, to_review: int = 0) -> str:
    """Upload, review, sign off: where the teacher is, and what is left."""
    steps = [("Upload a paper", "PDF scan or reader export"),
             ("Review", f"{to_review} answer{'s' if to_review != 1 else ''} need you" if stage >= 1 else "only the uncertain answers"),
             ("Sign off", "the marks become final")]
    items = []
    for k, (title, detail) in enumerate(steps):
        state = "done" if k < stage else "current" if k == stage else "todo"
        current = ' aria-current="step"' if state == "current" else ""
        items.append(f'<li class="step {state}"{current}><span class="step-n">{k + 1}</span>'
                     f'<span class="step-t">{escape(title)}</span><span class="step-d">{escape(detail)}</span></li>')
    return f'<nav aria-label="Grading progress"><ol class="stepper">{"".join(items)}</ol></nav>'


class AnswerMap(gr.HTML):
    """Every question of the paper as a button. Its value is {"groups": [...], "selected": q}; clicking a question
    sets "selected" and fires change, which the app uses to open that question."""

    def __init__(self, value=None, **kwargs):
        html = """
        <div class="amap" role="group" aria-label="Questions on this paper">
        ${(value && value.groups ? value.groups : []).map(g => `
          <section class="amap-group"><h3>${g.title}<span>${g.note}</span></h3><div class="amap-grid">
          ${g.cells.map(c => `<button type="button" class="cell ${c.state} ${value.selected === c.q ? 'selected' : ''}"
               data-q="${c.q}" aria-pressed="${value.selected === c.q}"
               aria-label="Question ${c.q}, ${c.label}${c.mark ? ', ' + c.mark : ''}" title="${c.label}">
               <span class="cell-q">${c.q}</span><span class="cell-mark">${c.mark || c.symbol}</span></button>`).join('')}
          </div></section>`).join('')}
        ${value && value.groups && value.groups.length ? `<ul class="amap-legend">${value.legend.map(l =>
            `<li><span class="swatch ${l.state}">${l.symbol}</span>${l.label}</li>`).join('')}</ul>` : ''}
        </div>"""
        js = """element.addEventListener('click', (event) => {
            const cell = event.target.closest('button[data-q]');
            if (!cell) return;
            props.value = {...props.value, selected: Number(cell.dataset.q)};
            trigger('change');
        });"""
        super().__init__(value=value, html_template=html, js_on_load=js, **kwargs)


def map_value(context: dict | None, sheet: list, selected: int | None) -> dict:
    """The answer map's value from the grading context and the teacher's current mark sheet."""
    context = context or {}
    if not context.get("in_scope"):
        return {"groups": [], "selected": None, "legend": []}
    referrals = context.get("referrals", {})
    proposals = dict(zip((q for q, _ in context["limits"]), context.get("proposals", [])))
    marks = {row[0]: row[1] for row in sheet or [] if len(row) == 2}
    cells = []
    for q, maximum in context["limits"]:
        mark, proposal = marks.get(q), proposals.get(q)
        if q in referrals:
            state = "teacher" if mark not in (None, "") else "review"
        elif proposal is None:
            state = "absent" if mark in (None, "") else "teacher"
        else:
            state = "proposed" if mark == proposal else "changed"
        shown = f"{mark:g}/{maximum:g}" if isinstance(mark, (int, float)) else ""
        label, symbol = STATES[state]
        cells.append({"q": q, "state": state, "mark": shown, "label": label, "symbol": symbol, "max": maximum})
    if all(c["max"] == 1 for c in cells[:20]) and len(cells) == 35:  # the built-in exam: choices, then writing
        groups = [{"title": "Multiple choice", "note": "Q1–20, 1 point each", "cells": cells[:20]},
                  {"title": "Written answers", "note": "Q21–35, 2 points each", "cells": cells[20:]}]
    else:
        groups = [{"title": "Questions", "note": f"{len(cells)} on this exam", "cells": cells}]
    legend = [{"state": s, "label": l, "symbol": sym} for s, (l, sym) in STATES.items()
              if any(c["state"] == s for c in cells)]
    return {"groups": groups, "selected": selected, "legend": legend}


def summary_band(title: str, detail: str, completed: int, total: int, chips: list[tuple[str, str]]) -> str:
    """The slim bar above a graded paper: what is left, a thin progress line and a few facts as small pills."""
    share = round(100 * completed / total) if total else 0
    pills = "".join(f'<li><b>{escape(v)}</b> {escape(k)}</li>' for k, v in chips)
    return (f'<section class="paper-bar" aria-live="polite"><div class="bar-text"><h2>{escape(title)}</h2>'
            f'<p>{escape(detail)}</p></div>'
            f'<div class="bar-line" role="progressbar" aria-label="Questions with a mark" aria-valuemin="0" '
            f'aria-valuemax="{total}" aria-valuenow="{completed}"><span style="width:{share}%"></span></div>'
            f'<ul class="bar-pills">{pills}</ul></section>')


def proposal_card(q: int, detail: dict, mark) -> str:
    """A question the product marked: what it read, what it proposed and why, so the teacher can confirm or change."""
    source = detail.get("source") or ""
    quote = detail.get("quote") or ""
    evidence = (f'<blockquote><p>{escape(quote)}</p><cite>{escape(source)}</cite></blockquote>' if quote
                else f'<p class="muted">{escape(source)}</p>' if source else "")
    shown = f"{mark:g}" if isinstance(mark, (int, float)) else "–"
    return (f'<article class="review-question proposal"><h3>Question {q}, proposed {shown}/{detail["max"]:g}</h3>'
            f'<p class="question-prompt">{escape(detail["question"])}</p>'
            f'<div class="answer-pair"><div><h4>Student answer, as read</h4><p>{escape(detail["reading"] or "—")}</p></div>'
            f'<div><h4>Why this mark</h4><p>{escape(detail["feedback"] or detail["reason"])}</p></div></div>'
            f'{evidence}<p class="review-reason"><strong>Agree?</strong> Keep it, or choose another mark below. '
            'A change you make here also teaches the review rule.</p></article>')


def alert_banner(message: str, alerts: list[str]) -> str:
    state = "alert" if alerts else "quiet" if "no alert" in message else "pending"
    title = {"alert": "Alert: inputs or feedback look different",
             "quiet": "No alert: recent papers look like the reference set",
             "pending": "Not enough papers yet to check for drift"}[state]
    items = "".join(f"<li>{escape(a)}</li>" for a in alerts)
    return (f'<section class="banner {state}" role="status"><strong>{escape(title)}</strong>'
            f'<p>{escape(message)}</p>{f"<ul>{items}</ul>" if items else ""}</section>')


def tiles(summary) -> str:
    """The request summary as tiles: the numbers a teacher or the team checks first."""
    wanted = ["requests", "sent to review", "rated 👍/👎", "share 👎 (of rated)", "median latency (s)",
              "mean known request cost (€)"]
    values = dict(zip(summary["metric"], summary["value"])) if len(summary) else {}
    cells = "".join(f'<li><span class="tile-v">{escape(str(values.get(k, "–")))}</span>'
                    f'<span class="tile-k">{escape(k)}</span></li>' for k in wanted)
    return f'<ul class="tiles">{cells}</ul>'
