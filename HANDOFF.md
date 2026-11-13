# Anna ↔ Component 2: proposed handoff v1

Component 2 lives on `component-2-grading` in a checkout of the team's fork. Anna's published `vision_model` branch and `main` both pointed to the starter commit on 11 November. Agree which branch receives the integration PR and supply one vetted PDF with a separately checked answer export. [Team fork](https://github.com/annaserrag/daiist-group-project)

## Share the grading branch

To share the local branch before the integration PR, create a Git bundle from the Component 2 checkout. The bundle contains committed project files and history. The private response cache and local credentials stay on the owner's machine.

```powershell
git bundle create "$env:USERPROFILE/Documents/component-2-grading.bundle" component-2-grading
```

Anna can fetch the snapshot in her clone of the team fork and inspect it in a separate worktree. With the bundle saved in her Documents folder, these commands leave the branch in her current worktree untouched.

```powershell
git fetch "$env:USERPROFILE/Documents/component-2-grading.bundle" refs/heads/component-2-grading:refs/remotes/handoff/component-2-grading
git worktree add --detach ../grading-integration-check handoff/component-2-grading
```

## Reproduce Javi's feedback comparison

The grading branch includes all 23 approved labels and the frozen judge predictions for the earlier lexical grader. Run this from the grading checkout to reproduce 11 mutual passes, two mutual fails and ten judge passes that Javi failed. It runs offline with the project's dependencies installed. [Evaluation and review method](REPORT.md#written-feedback-and-judge-agreement)

```powershell
uv run --locked --no-dev python -m exam_grader.metrics agreement eval/human_labels.csv --judge-results eval/judge_results.csv
```

## Vision produces one object per exam

| Field | Meaning |
|---|---|
| schema_version | Literal "vision.exam.v1". Change it together if semantics change. |
| exam_id (optional for the built-in ML paper) | For another written exam, copy the definition fingerprint shown by the teacher app or returned by `python -m exam_grader.written_exam path/to/exam.json`. The reader must identify which exam was scanned before returning its fingerprint. |
| exam_status | "in_scope", "out_of_scope", or "unreadable". Only in-scope exams may receive proposals. |
| answers | List; any order; keep duplicate detections visible. Do not fill absent questions. |
| question_id | Integer 1–35, or null if the question cannot be assigned. Do not guess to satisfy the schema. |
| text | OCR text as read; for an MCQ, one selected letter; for a confirmed blank, the empty string. |
| confidence | Finite 0–1 score for the reading, or for the blank determination. Not a probability of a correct mark. |
| id_confidence | Separate finite 0–1 score for assigning the question number. Do not copy the text confidence. Until there is a usable signal, send 0 for uncertain IDs. |
| state | "read", "unreadable", or "confirmed_blank". Use the last only if Vision has positive evidence of a blank answer area for that specific question. |
| features (optional, at exam level) | ink_density (0–1) and line_count (nonnegative integer). Only these extra features are logged. |
| usage (optional, at exam level) | The reader can return `{"tokens":1200,"cost_eur":0.001}` for its full PDF request. An offline reader can return explicit zeros. Omitted or null measurements count as unknown. The course evaluation reports both components and a combined total when measured. Replaying a JSON file consumes zero reader tokens regardless of the historical usage recorded inside it. |

**Example:** Vision reads a clear “B” but cannot distinguish Q1 from Q7. Return an unassigned detection or flag low ID confidence. A high letter confidence does not justify giving Q1 a mark. A missing region is also different from a confidently empty region: one needs review, the other can receive a proposed zero.

Anna can validate an export with the [JSON Schema](exam_grader/data/vision.exam.v1.schema.json) or run the command below in the team checkout. It checks field types and reports duplicate, unassigned and missing question numbers without printing student text. A partial export can pass the format check and still need teacher review.

~~~powershell
uv run --no-dev python -m exam_grader.handoff exam_grader/data/vision-example.json
~~~

## Component 2 returns two views

- **teacher:** 35 ordered question rows containing a proposed mark or null, review reason, feedback and citation. An incomplete exam has a subtotal but no total. Every exam still requires teacher sign-off.
- **record:** exactly the pipeline data intended for the template logger. Contains counts, statuses, source IDs, versions, usage and numeric features; no scan, filename, OCR text, quote, or feedback. The teacher view is transient. Do not concatenate it into the record.
- A duplicate detection withholds that question. An unknown ID can belong anywhere and blocks the whole exam in v1. Weak retrieval or OCR never silently becomes zero. “differs_from_key” is a proposed incorrect response; “review” means the grader cannot propose a defensible mark.

## Connect the actual vision reader

The working pipeline accepts the example JSON now. For PDFs it imports the module named by GRADE_VISION_MODULE (default vision). Agree on this Python interface or add a small adapter to Anna's existing function.

~~~python
VERSION = "your-vision-model-and-weights-version"

def read_exam(path):
    # Read one PDF and return the agreed vision.exam.v1 dictionary.
    ...
~~~

The server passes a pathlib.Path. Convert it to str(path) inside the adapter if needed. Once the module is importable, the next PDF upload will use it. The /run endpoint processes the file once, records both component outputs in the course logbook, and shows proposals, teacher edits and sign-off in the app. The real OCR text and teacher feedback stay out of the log. The local HTTP test used the bundled synthetic export. The first test with Anna still needs a vetted PDF and her reader.

Coordinate changes to app.py, pipeline.py and evaluate.py when merging the component branch. Keep the professor's serve(raw_input), pipeline.run(raw_input) and /run endpoint in the shared app.

## First combined acceptance check

From the combined checkout, run this command on one PDF and its independently verified reference. It uses the same connector as the app, with external grading disabled. The reader still runs its own configured vision backend. The output identifies missing, unexpected and differing question IDs without printing the readings. Exit 0 requires the expected exam status and an exact match on the reference states and readings after whitespace normalization. Compare confidence calibration and live grading separately.

```powershell
uv run --locked --no-dev python -m exam_grader.check_reader "private/exam.pdf" "private/checked-export.json" --module vision
```

### Exercise the PDF route before the handwriting reader arrives

This optional engineering check generates an image-only printed PDF using three existing fixture readings. RapidOCR reads each question number and answer in a separate crop. The normal PDF entry point passes those readings into the grader. It runs locally. Adding `--live` permits Gemini to grade the single synthetic written answer. In the recorded live run, all three readings matched, Q1 earned 1/1, the invalid Q17 selection went to review, and Q21 earned 2/2. The PDF is temporary. The printed reader accepts only its generated fixture. [Measured run](eval/printed_pdf_integration.json)

```powershell
uv run --locked --no-dev --with rapidocr-onnxruntime==1.4.4 --with pypdfium2==5.15.2 python -m exam_grader.printed_smoke
```

1. Anna runs her exporter on a vetted sample; both of you inspect the question IDs, texts, blank states and confidence semantics. Have her run the offline CLI on that JSON before submitting a PR.
2. Pass that same PDF through read_exam → connect. Inspect Q1 and one short answer against the actual sheet. Without Gemini a readable written answer must route to review.
3. On a vetted file with the semantic index and Gemini enabled, inspect the cited passage and proposed mark. Confirm the same source ID appears in the safe log; the student's text must not.
4. Try a missed question and a dubious question number. Check that neither becomes a guessed grade. Try a different PDF which is not on the server's allowlist: no external model call.
5. For the component PR, confirm that the two sides agree on the export and the grading tests pass. When the team prepares the full submission, run check.py with the real evaluation files and try the same PDF through the deployed app.
