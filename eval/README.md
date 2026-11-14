# Evaluation files

## Reproduce the Component 2 evidence

Run these commands from the repository root.

Recalculate the written generalization results from the released anonymized predictions and teacher marks. This command requires neither a key nor access to the original student papers. It reports proposal and review counts on all readable answers, then computes errors only for proposals with a numerical reference. It also calculates the error of awarding full credit on the same selected answers.

```powershell
uv run --locked --no-dev python -m exam_grader.generalization
```

The ML files identify each paper and question, the frozen policy and the checksums of the private exports. The UNT file records the public dataset archive hash, zero-based response line numbers, numeric scale, rubrics and source notes. Its new-question test uses different prompts from all development trials. The [physics demonstration](../README.md#use-a-different-written-exam) can be rerun live with the included definition and PDF.

After completing the 23-item worksheet and saving the downloaded CSV as `eval/human_labels.csv`, compare your labels with the already frozen judge predictions. This command runs locally without a key. It joins items by ID and prints the pass/fail counts, including cases the judge passed that you failed. You can run it before Anna's reader or the course case set is ready.

```powershell
$archive = "$env:LOCALAPPDATA/daiist-component2/heldout"
uv run --locked --no-dev python -m exam_grader.metrics agreement eval/human_labels.csv --judge-results "$archive/judge-results.csv"
```

Retrieval and drift replays use public material. The monitoring replay downloads the checksum-verified teacher CSV or accepts a local copy via `--teacher-csv`.

```powershell
uv run --locked --no-dev python -m exam_grader.metrics retrieval eval/retrieval_sources.csv --written-only --semantic --teaching-notes
uv run --locked --no-dev python -m exam_grader.monitor_replay
```

The component owner can reproduce the private feedback packet from the four redacted exports and their archived evaluation cache. On this workstation, the private archive lives in `$env:LOCALAPPDATA/daiist-component2/heldout`. Open `feedback-review.html`, choose Pass or Fail independently, and click **Download labels CSV**. Put the completed `human_labels.csv` in `eval/` and use the judge command below. The form can save progress in the browser. A blank `feedback-labels.csv` is also provided for offline editing. Rebuilding the packet preserves edits to that CSV. Keep the case manifest, marked-up texts and private paths outside the repository. The packet excludes the superseded model-based Q32 proposal.

```powershell
$archive = "$env:LOCALAPPDATA/daiist-component2/heldout"
uv run --locked --no-dev python -m exam_grader.feedback_packet "$archive"
# Set GRADE_APPROVED_HASHES to the four vetted export hashes before judging.
uv run --locked --no-dev python -m exam_grader.judge "$archive/feedback-items.csv" eval/human_labels.csv "$archive/judge-results.csv"
```

The judge can freeze its predictions with `--predict-only` before human annotation. It checkpoints and resumes successful requests by exact prompt and evidence fingerprint. Keep its output separate from the human worksheet until labeling is finished. Copy the final judge output to `eval/judge_results.csv` for the course runner. For a controlled threshold comparison on the private two-paper comparison cohort, run the following after populating the response cache.

```powershell
$env:GRADE_RETRIEVAL_MODE = "semantic"
$env:GRADE_EVIDENCE_RANKING = "original"
$env:GRADE_PASSAGE_STRATEGY = "original"
uv run --locked --no-dev python -m exam_grader.heldout_written "$archive" --cohort fresh --frozen --output "$archive/replayed-thresholds.json"
```

The original 0.40 independent test fixed two additional papers by deterministic ordering before grading. Q32 was excluded because it had already been examined across all 50 papers. Only redacted answer transcriptions went to the grader. Decisions were saved locally before joining to the published teacher marks. To reproduce a saved run, use its matching retrieval and passage settings. The current runtime defaults are `multipart` and `finalize`. A new `--run-label` preserves earlier private per-answer results when comparing versions. To reproduce the large-deduction review comparison, run `python -m exam_grader.review_policies "$archive"`. [Original independent run](written_independent_semantic.json)

Inputs are written the same way as in the App tab: text, or the path of a file under `eval/inputs/` (an image, a document…), written from the repo root, e.g. `eval/inputs/case_07.jpg`.

## Printed baseline and held-out written grading

For the direct-PDF comparison, run both modes on the built-in printed fixture. The direct mode sends its generated PDF and the public key to Gemini. Pipeline mode runs local OCR first. Both modes require the local Gemini key. The optional rate settings in the project setup enable the paid-tier cost estimate.

```powershell
uv run --locked --no-dev python -m exam_grader.printed_baseline --method direct --repeats 3
uv run --locked --no-dev --with rapidocr-onnxruntime==1.4.4 --with pypdfium2==5.15.2 python -m exam_grader.printed_baseline --method pipeline --repeats 3
```

The four-paper written validation is separate from the team's end-to-end cases. IDs 13, 18, 35 and 22 followed the two earlier independent papers in the same sampling order. Only Q21–35 excluding Q32 were eligible. The 56 original decisions were sealed before the comparison with teacher marks. On the owner's workstation, use the four private transcriptions and response cache to replay the frozen grader. The command checks the sealed per-answer files; write the new aggregate output to the private directory. [Archived results](written_validation_four.json)

```powershell
$env:GRADE_RETRIEVAL_MODE = "semantic"
$env:GRADE_INDEX_DESIGN = "section"
$env:GRADE_EVIDENCE_RANKING = "multipart"
$env:GRADE_PASSAGE_STRATEGY = "finalize"
uv run --locked --no-dev python -m exam_grader.heldout_written "$archive" --cohort validation --frozen --run-label sealed --output "$archive/four-paper-replay.json"
```

## `cases.csv`: the end-to-end test set

The prepared packet contains 30 distinct image-only PDF requests across 33 pages. Twelve are normal, eight are hard, seven require review and three are out of scope. Twenty-one PDFs contain actual handwriting cropped from eight public uncorrected Data Science exam papers. The other nine are constructed controls for situations such as an obscured answer, a positively identified blank, and a geography worksheet. A generic exam header was added to the handwritten excerpts. Only the answer regions are embedded. The two-page Q24 answer in c20 must be read as a single response. Case c16 contains all three answer-bearing pages of one submission; most other cases exercise particular parts of a sheet.

The team packet was assembled with Codex. Transcriptions were checked against the images during preparation, independently of the reader and grader under test. The case notes specify each intended behavior. Handwritten MCQ targets use recorded letters and the published key. Handwritten numeric scores are from the published teacher CSV, including partial credit. Review targets reflect the operating policy; their historical teacher marks, where available, are retained separately. The files reuse source papers and some responses across situations, and earlier Component 2 work used this dataset. Report the results as a targeted integration test, not a held-out estimate on 30 independent students. [Checked readings, source page regions, marks and PDF hashes](case_references.json) · [Dinesh K P, Mendeley Data v1, CC BY 4.0](https://doi.org/10.17632/sf3kvjwknt.1) · [License](https://creativecommons.org/licenses/by/4.0/)

#### Run with Anna's reader

Install the integrated checkout and run both acceptance PDFs as described in [SETUP.md](../SETUP.md). For the full evaluation, configure the grader's local key and use the frozen PDF hashes below. The PDFs contain no searchable answer layer. The evaluation runner keeps the reference readings and teacher marks separate from both components.

```powershell
$packet = Get-Content -Raw eval/case_references.json | ConvertFrom-Json
$env:GRADE_APPROVED_HASHES = ($packet.psobject.Properties.Value.sha256 -join ' ')
uv run --locked --no-dev python main.py evaluate
```

For grading, read `results.csv` and `summary.json` in this directory. The case pass requires the prewritten marks/referrals, document status, correct target IDs and states, and no unexpected or unassigned readings. Missing and unreadable answers have different expected behavior. A provider failure earns no credit for an expected substantive referral. A reader crash or unavailable reader fails even an unreadable test. Partial sheets must have no proposed total. The grader still returns 35 rows, so the per-sheet raw review count includes absent questions. Use `numeric_references` and `comparable_proposals` for coverage on labeled answers.

For the reader, `reader_exact_cases` and `exact_reading_slots` compare text after case and whitespace normalization. Punctuation and wording still count. A benign transcription difference can leave the numeric mark correct. Per-case missing, unexpected, state/count and text mismatch IDs use the same reader invocation that produced the mark. Reference regions use [PDF page, top, bottom], with vertical bounds in thousandths of page height. Final PDF inputs are the authoritative crops. For comparison, the executable baseline assumes perfect reading and scope recognition and awards full credit to every visible response. Report its case passes and exact numeric matches both overall and on the grader's proposals. The manual teacher workflow in the CSV is the operational baseline; its timing has not been measured here.

Before the reader arrives, validate the hashes, document types and scoring controls offline:

```powershell
uv run --locked --no-dev python -m unittest discover -s tests -p test_pdf_cases.py
```

30 or more cases written by the team, from realistic inputs, not generated by the model you are testing.

| Column | Content |
|---|---|
| `id` | Short unique id, e.g. `c01` |
| `input` | The input (see above) |
| `kind` | `normal`, `hard` (should work but is difficult), `review` (should be sent to a person) or `out_of_scope` (the product should refuse or say it cannot help) |
| `expected` | What a correct output is: a label, a value, or a short description the judge can check against |
| `baseline` | What happens to this case today, without the product |
| `notes` | Optional: why the case is in the set |

Include at least 5 `hard`, 5 `review` and 3 `out_of_scope` cases, and some `normal` ones: the drift demo uses them as normal traffic.

## `human_labels.csv`

If you use an LLM judge: your own labels for 20+ cases (`id`, `label`, `comment`). Compute your agreement with the judge in code.

## Reference and shifted scans for monitoring

`reference.csv` lists 30 one-page image-only PDFs, one answer-region crop per source paper. The first 30 numbered papers outside the eight source papers in the existing integration packet were selected. The crops include handwritten multiple-choice and written responses and omit student headers. Each uses the same answer-sheet heading and page layout as the normal packet. The manifest records the original file hash, page, vertical crop bounds in thousandths of page height, and final file hash. These extracts are adapted from Dinesh K P’s Mendeley Data release (v1, CC BY 4.0). [Source dataset](https://doi.org/10.17632/sf3kvjwknt.1) · [Licence](https://creativecommons.org/licenses/by/4.0/)

`shifted.csv` pairs each of ten synthetic faded scans with its unchanged original, c01 through c10. Each complete parent page was rasterised at 1.5 pixels per PDF point and placed back at the original size. For each grayscale pixel the output is `round(255 - retained_contrast * (255 - pixel))`. Fractions 0.12, 0.18, 0.24, 0.30 and 0.36 were applied twice in case order. The official normal window uses all 12 existing normal cases, including the two additional originals c11 and c12. All 40 new PDFs and the computed reference features are bundled.

From the integrated checkout, run:

```sh
uv run --locked --no-dev python main.py reference
uv run --locked --no-dev python main.py drift-demo
uv run --locked --no-dev python check.py
```

The shared drift function uses PSI (population stability index), with bin boundaries derived from the reference and half an observation added per bin. Each input signal is compared with the larger of 0.25 and twice the expected sampling noise for the two window sizes. The reference was generated with the integrated page reader on Windows/CUDA, with remote grading disabled. Only reader input features participate in the alert. Reference latency and grading fields are outside this calculation. A zero-ink reading is a valid measurement. The app requires 20 recent requests before it assesses live drift; the course demo evaluates the supplied smaller windows. [Measured results](../REPORT.md#operations)

## Held-out written grading pilot

`heldout_written_pilot.json` summarizes a paired threshold sweep on all 15 written questions from each of two papers. The transcriptions came from the uncorrected Student_17 and Student_42 PDFs. They preserve incomplete answers and normalize handwriting, line breaks and punctuation. They were fixed before joining the teacher CSV. Each reading and question assignment has confidence 1 to isolate Component 2. The baseline fits the rounded question mean on the other 48 exams. [Dinesh K P, Mendeley Data v1, CC BY 4.0](https://doi.org/10.17632/sf3kvjwknt.1)

The original pilot is preserved. The revised guide assigns two points across named criteria for each question and requires a reason for each deduction. The fresh test uses Student_8 and Student_36, selected before inspecting their answers or marks. All four papers are excluded from training the fresh-set baseline. The rubric, model and retrieval index are shared by the revised pilot and fresh runs. Run these commands from the project root in PowerShell. The private folder holds the four vetted transcriptions, cached responses and question-level comparisons with feedback. Identical requests reuse the same successful model response across thresholds. A lower threshold can change the selected evidence by skipping the question-only retry.

```powershell
uv run --no-dev python -m exam_grader.heldout_written "$env:LOCALAPPDATA/daiist-component2/heldout" --cohort pilot --output eval/written_current_pilot.json
uv run --no-dev python -m exam_grader.heldout_written "$env:LOCALAPPDATA/daiist-component2/heldout" --cohort fresh --output eval/written_current_fresh.json
uv run --no-dev python -m exam_grader.heldout_written "$env:LOCALAPPDATA/daiist-component2/heldout" --cohort q32 --output eval/q32_count_replay.json
```

Review share uses all 30 answers. Agreement and mean absolute error use only paired numeric proposals and teacher marks. A teacher review has no numeric proposal. Each model mark is out of two points. These two papers can guide development. Any later choice of threshold or scoring rubric needs confirmation on fresh papers.

The Q32 replay compares the four archived transcriptions with the previous Q32 decisions at 0.25. The count rule runs locally on confidently assigned, readable answers. It counts equivalent names once. Unrecognized wording that could change the award goes to review. The four-paper comparison with feedback is written to `q32-count-comparison.json` in the private evaluation folder. Earlier rules interpreted “max-node” as the terminal-node limit used by the R randomForest package. Version 4 asks the teacher whether “max-node” duplicates a separately named leaf limit when the distinction affects the mark. [Parameter definition](https://search.r-project.org/CRAN/refmans/randomForest/html/randomForest.html) · [50-paper comparison](q32_all50_ambiguous_duplicates.json)

The independent Q32 follow-up, `q32_fresh_count.json`, covers all 46 remaining papers. Manual transcriptions and production decisions were frozen before joining teacher marks. Of 24 numeric teacher marks only two had a proposal; one matched exactly. The other 22 labelled cases went to review. Read this as a coverage limitation, not as a reliable estimate of marking accuracy. The private evaluation folder contains `q32-fresh-blind.json` (transcriptions and frozen decisions) and `Q32 fresh papers.md` (the complete comparison).

To replay the revised Q32 rule on those same 46 readings, run `python -m exam_grader.q32_replay "$env:LOCALAPPDATA/daiist-component2/heldout"` from the project root. The command verifies the archived input digest, applies the production rule, then joins the published teacher marks. It writes aggregate results under `eval/` and the individual comparisons in the private folder. Version 2 recognizes singular "epoch" and affirmative definitions of the cluster count in K-means. It proposed 4 marks (previously 3); of 3 with numeric teacher references, 1 matched. This is a development replay because the cases informed the revision.
