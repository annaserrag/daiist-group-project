# Anna ↔ Component 2 handoff

Component 2 lives on `component-2-grading`. Anna's `vision-finetune` branch provides the page reader. The branches can run together in separate local checkouts. [Team fork](https://github.com/annaserrag/daiist-group-project)

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

Anna can validate an export with the [JSON Schema](exam_grader/data/vision.exam.v1.schema.json) or run the command below in the team checkout. It checks field types and reports duplicate, unassigned and missing question numbers without printing student text. A partial export can pass the format check and still need teacher review. For a different written exam, add `--written-exam path/to/definition.json` to both the export check and the PDF comparison. The check uses the selected exam’s fingerprint and question numbers.

~~~powershell
uv run --no-dev python -m exam_grader.handoff exam_grader/data/vision-example.json
~~~

## Component 2 returns two views

- **teacher:** one row per question (35 on the built-in exam) containing a proposed mark or null, review reason, feedback and citation. An incomplete exam has a subtotal but no total. Every exam still requires teacher sign-off.
- **record:** exactly the pipeline data intended for the template logger. Contains counts, statuses, source IDs, versions, usage and numeric features; no scan, filename, OCR text, quote, or feedback. The teacher view is transient. Do not concatenate it into the record.
- A duplicate detection withholds that question. An unknown ID can belong anywhere and blocks the whole exam in v1. Weak retrieval or OCR never silently becomes zero. “differs_from_key” is a proposed incorrect response; “review” means the grader cannot propose a defensible mark.

## Connect the actual vision reader

In the integrated branch, follow [SETUP.md](SETUP.md) to run both components in one environment. The following optional workflow is for developers testing two separate checkouts. From the grading checkout, point the adapter at the local root of Anna's branch. The path below matches Javi's Windows worktree. On a new machine first install the reader environment with `uv sync --locked --no-dev` in Anna's checkout. The first nonblank page read downloads the pretrained model. On Windows, check that the reader's PyTorch build supports your GPU if you want CUDA acceleration.

```powershell
$env:GRADE_VISION_ROOT = "$env:LOCALAPPDATA/daiist-component2/worktrees/vision-finetune"
$env:GRADE_VISION_MODULE = 'exam_grader.vision_finetune'
uv run --locked --no-dev python -c "from pipeline import run_views; t=run_views('eval/inputs/c01.pdf')['teacher']; print('reader_failed', t['reader_failed']); print([(a['question_id'], a['readings'], a['mark'], a['reason']) for a in t['answers'][:5]])"
```

The same settings work for PDF uploads in the app. With both packages installed in one environment, omit `GRADE_VISION_ROOT`. Component 2 runs Anna's page model, prompt and line assignment, retaining the generated token scores for the question number and answer separately. When the original line score is below 0.85 and its digits clear that threshold, the bridge checks the label again. For a page with one detected label it requests a read of the question detector's crop. For a page with multiple labels it requests an ordered list and checks the count, position and digits. When confirmation is unavailable the original ID score applies. These checks never change the transcribed number. A validated multiple-choice response uses the letter score; written responses use the weakest token in the answer across its lines. Older native exports use their original confidence fields.

For a low-confidence written answer with a confident, unique question ID and an answer confined to one page, the reader rereads that page at a different scale. Its prompt contains the question number only, alongside instructions for verbatim transcription. When both readings agree on every word in order, including case and punctuation, the bridge uses the higher of their two whole-word scores for each word and the lowest resulting score across the answer. The same 0.80 text threshold applies. A disagreement, failed reread or continuation on another page leaves the first score in place. [Paired evaluation on 30 PDFs](eval/vision_finetune_word_confidence_30.md)

The adapter preserves written text, repeated question numbers and unassigned lines. Unknown numbers and subparts go to review. An empty transcription is unreadable. Local inference reports zero API cost and leaves unmeasured token usage unknown.

Use PDFs from the bundled Mendeley exam. The page reader rejects an entire PDF when every page has a confidently transcribed conflicting document heading and a second image pass confirms that document type. Ambiguous headings and unlabelled or mixed packets follow the ordinary reading and review rules. This does not authenticate a specific exam version or confirm a blank answer area. [Scope and regression checks](eval/vision_finetune_scope_30.md) The grader's existing ID and OCR confidence thresholds still apply. The training dataset and LoRA smoke run are separate from PDF inference. The pretrained page reader can run before training.

On 14 November the pretrained reader at `8086331` returned the five choices in `c01.pdf` as loose lines such as `1  D`. The adapter recognizes an isolated number and choice on a page carrying the printed exam header. It keeps the original line confidence and sends any additional ambiguous lines to review. All five letters and marks matched the checked references after this change. On the separate `c22.pdf` control, Q1 earned 1/1 and Q17's invalid D went to review.

Two further checks used the same PDF entry point. The separate handwritten sheet `c02.pdf` matched all five readings and marks. In the constructed written-answer control `c26.pdf`, Q32 reached the grader as “Learning rate” and the one-hyperparameter rule sent it to review. Across these four sheets, all 13 target readings, 11 numerical marks and two referrals matched the checked references. The first sheet was used to develop the adapter.

Javi's Windows reader environment uses the CUDA 13.0 wheels of PyTorch 2.14.1 and torchvision 0.29.1. The default locked installation on this machine supplied CPU-only wheels. To reproduce the GPU setup, run the following in the reader checkout after syncing its dependencies. The smoke run uses the prepared training data in the user's cache and discards its training updates.

```powershell
uv sync --locked --no-dev --group train
uv pip install --python .venv/Scripts/python.exe --index https://download.pytorch.org/whl/cu130 torch==2.14.1+cu130 torchvision==0.29.1+cu130
uv run --no-sync --group train python -m vision.finetune_reader --smoke
```

Coordinate changes to app.py, pipeline.py and evaluate.py when merging the component branch. Keep the professor's serve(raw_input), pipeline.run(raw_input) and /run endpoint in the shared app.

## First combined acceptance check

From the combined checkout, run this command on one PDF and its independently verified reference. It uses the same connector as the app, with external grading disabled. The reader still runs its own configured vision backend. The output identifies missing, unexpected and differing question IDs without printing the readings. Exit 0 requires the expected exam status and an exact match on the reference states and readings after whitespace normalization. Compare confidence calibration and live grading separately.

```powershell
uv run --locked --no-dev python -m exam_grader.check_reader "private/exam.pdf" "private/checked-export.json" --module exam_grader.vision_finetune
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
