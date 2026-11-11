# Exam grading assistant · Component 2

Start the local Gradio app from this folder in PowerShell.

~~~powershell
uv run --locked --no-dev python app.py
~~~

Open the localhost address printed in the terminal. Choose **Try synthetic OCR example** to see proposals, the text Vision read, evidence and questions requiring teacher review. Q1 scores 1, Q23 is a confirmed blank worth 0, and Q17 stays empty because D is outside its two options. Complete the teacher marks and click **Sign off all 35 marks** to produce the approved sheet for this session. Monitoring shows request counts, hourly review share and feedback. The built-in example is a partial OCR export and needs no key.

To run the test suite or inspect Anna's exported JSON from the command line:

~~~powershell
uv run --locked --no-dev python -m unittest discover -s tests -v
uv run --locked --no-dev python -m exam_grader
~~~

## Demonstrate the grading component

1. Start the app and click **Try synthetic OCR example**. Show why Q1 is a valid proposal and why Q17 needs a teacher to inspect its selected option. With Gemini enabled, inspect the Q21 mark, citation and quoted text.
2. Complete all 35 marks on this synthetic sheet and click **Sign off all 35 marks**. The signed sheet shows how many proposals the teacher accepted or changed, how many referrals they completed, and the elapsed time since the proposals appeared. Open Monitoring to check the single grading request.
3. Show the printed-PDF check and the independent written evaluation. One verifies that OCR can feed the grader. The other compares actual written proposals with published teacher marks, including the 13/28 referrals on the revised replay. For the final class demo, repeat the PDF path with Anna's reader and a vetted handwritten sample. [Printed check](HANDOFF.md) · [Written evidence](eval/written_independent_revised.json)

For a time-saving claim, have the team time manual grading and assisted sign-off on comparable exams. Record question counts, whether a teacher had seen each paper, and any interruptions. The app's sign-off timer measures elapsed time. The business calculation separately needs the teacher's actual hourly rate and the team's intended volume. [Business assumptions](REPORT.md#business-case)

## Your branch, separate from Anna's

This checkout uses `component-2-grading`, based on commit `490638f` shared by Anna's `vision_model` and `main` at setup. Confirm the target branch with Anna before opening the component PR and coordinate the changes to app.py, pipeline.py, evaluate.py and monitor.py. [Team fork](https://github.com/annaserrag/daiist-group-project)

## What happens to one answer

1. Match by question ID, never list position. Reject malformed or ambiguous observations. Only a positively identified blank gets a proposed zero without evaluating text.
2. Q1–20: compare an allowed letter with the published key. Q17 only offers A and B. Student marks never participate in inference.
3. Written answers use cached question embeddings and local lexical overlap with the reading. Weak retrieval triggers a question-only retry. Low reading confidence still requires review. Q32 uses its distinct-hyperparameter rule. Both saved indexes support local search without an API key.
4. A bounded LangChain create_agent call sees the official reference answer and the best passages. Its structured response contains feedback, passage ID, evidence quote, review flag and mark. A mark outside {0, 0.5, 1, 1.5, 2}, a fabricated ID or a quote absent from its passage is blocked. Quote matching verifies provenance, not whether the feedback is substantively correct; that needs human evaluation.
5. No model call is required for deterministic or review-only routes. One model call for a direct proposal, or two if the model requests a full passage; no automatic generation retries, up to 30 calls per exam. The teacher signs off; this package supplies a proposal, not a gradebook write.

The active grader offers one passage lookup on its first model call. After a lookup the second call can return a structured mark or request review. In a paired diagnostic on four earlier failures, the original configuration reached its limit on all four. The revised configuration produced four cited marks using the same eight calls and three matched the teacher. [Paired lookup trial](eval/written_lookup_comparison.json)

For questions asking about both outlier detection and handling, the ranker reserves one of the top three positions for treatment evidence if the initial three omit it and an explicit treatment occurs among the first 30 candidates. Q27 now retrieves both detection and handling. The other 14 written question-only top-three lists remained identical. [Retrieval results](eval/written_retrieval_multipart.json)

Teachers can try an optional second check for large deductions with `written_review_at_or_below` in their policy JSON. A value of 1 sends otherwise valid model proposals of 0, 0.5 or 1 out of 2 to review. The evidence and suggested mark remain visible in feedback, and the total waits for a teacher. This option applies to model-graded written answers. It defaults to disabled. The final signed sheet records elapsed review time, proposals accepted or changed, and referrals completed. Elapsed time includes interruptions. Use a separate timed manual run to estimate time saved. [Routing comparison](eval/written_review_policies.json)

## Run a real written-answer grade

Copy .env.example to .env and set GEMINI_API_KEY locally. The app and build commands read the course .env file. For a single synthetic exam question using the available index:

~~~powershell
uv run --no-dev python -m exam_grader.smoke_live
~~~

Both semantic indexes are bundled. To regenerate them after a corpus change, run the following commands. The builder reuses identical inputs across runs. Restart the app after rebuilding an index or changing configuration.

~~~powershell
uv run --no-dev python -m exam_grader.build vectors --design section
uv run --no-dev python -m exam_grader.build vectors --design paragraph
~~~

`GRADE_INDEX_DESIGN` selects section or paragraph chunks. `GRADE_RETRIEVAL_MODE` accepts `lexical`, `semantic` or `auto`. Auto prefers the selected design's saved vectors. The corpus contains 367 paragraphs from 25 pages, grouped into 157 section chunks. To refresh it, use `uv run --with lxml python -m exam_grader.build sources` and rebuild the vectors. The server-owned `GRADE_APPROVED_HASHES` list controls which vetted inputs may send their OCR text to Gemini. The bundled synthetic example is already approved.

Set GRADE_INPUT_EUR_PER_M and GRADE_OUTPUT_EUR_PER_M to your verified tariffs for cost estimates. Missing tariffs still permit live grading. The course logger requires a numeric cost field, so cost_known accompanies that field. Evaluation and monitoring exclude unknown costs. Build-time embeddings are a separate expense. Local request latency includes the vision handoff and grading. Teacher sign-off and manual grading times must be measured separately.

## Decisions to make together

- **Answer key versus labels:** the PDF says both “grade against the key” and “key only for evaluation.” This implementation uses the public instructor key as runtime reference and reserves teacher marks for evaluation. Confirm that interpretation. The key is not a learner outcome and using it does not establish agreement with the teacher.
- **Q22 vs Q34:** the published Q22 states 99% training and 30% testing accuracy. Q34 gives no three classifiers to compare. The conservative draft policy still flags both; decide whether to remove Q22 after checking the sheet used in evaluation. Do not assume every policy flag is an OCR failure.
- **Thresholds and pass mark:** Policy defaults just exercise the flow. Select thresholds on separate validation exams, then freeze and report held-out results. The default does not invent a pass mark. Supply the actual exam-level pass rule and margin if you use a boundary check.
- **Ownership:** Anna owns OCR. You own retrieval, structured grading, evidence and Component 2 evaluation. Agree who edits the small shared pipeline/app/monitor files once. Record actual work in CONTRIBUTIONS.md and disclose AI assistance in REPORT.md.

## Evaluation

**Public-data baseline.** Reproduce the benchmark with the command below. The evaluation-only reader checks the published file's SHA-256 and uses its anonymous IDs in memory. The runtime grader never reads teacher marks.

~~~powershell
uv run --no-dev python -m exam_grader.reference_marks
uv run --no-dev python -m exam_grader.reference_marks --mcq-replay
~~~

The published 50 exams contain 987 valid recorded MCQ choices, 564 valid written marks, 185 unmarked written slots and one written mark above the maximum. Each MCQ reference is derived from the recorded choice and the published key. The teacher's MCQ section total is a separate reference. For written answers, the leave-one-exam-out baseline uses the other 49 exams to estimate each question's average, excluding missing/invalid marks. On the 564 valid marks, its mean absolute error before rounding was 0.512 points out of 2. After rounding to the nearest half point it matched exactly on 18.4% and was within half a point on 84.0%. These are baseline results. For a final comparison fit the baseline on training/validation exams and compare it with the grader on the same held-out answers.

**Recorded-letter replay.** Across the 50 published exams the grader applied the published key to all 987 scorable entries and routed the other 13 to review. This checks the deterministic grading path with transcription and question assignment assumed correct. Only 13 of the 44 fully scorable sheets matched the teacher's MCQ total exactly. Substituting D for B on Q1 in a separate sensitivity calculation raised that count to 29 of 44. The runtime key remains B. Six sheets had unresolved entries. Four of their teacher totals were inside the possible score range under the published key. See [aggregate results](eval/published_mcq_replay.json).

**Retrieval.** Run the local worksheet and open the address it prints (port 7861). Start with the 15 written questions: the MCQ grader uses the official key without search. The worksheet shows the complete exam question, the published key and searchable paragraphs exactly as indexed. Expand the optional reading guide if you want a starting point. Suggestions never populate the evaluation labels. Accept any page that actually supports the answer; check both parts of multi-part questions. Use “gap” for missing teaching evidence, “missing_context” for an incomplete question, or “open” to defer. Save before changing questions. You can also edit the CSV directly, separating accepted IDs with a vertical bar.

~~~powershell
uv run --no-dev python -m exam_grader.retrieval 29
uv run --no-dev python -m exam_grader.worksheet
uv run --no-dev python -m exam_grader.metrics retrieval eval/retrieval_sources.csv --partial
uv run --no-dev python -m exam_grader.metrics retrieval eval/retrieval_sources.csv --semantic --written-only
~~~

Hit@k tests whether any of the first k **chunks** belongs to an approved page. The partial report separates unanswered rows, confirmed corpus gaps and missing exam context. It shows rates both on supported rows and with corpus gaps counted as misses. Missing exam context is excluded from both denominators. Use the per-question ranks to investigate misses; a page hit does not prove that its returned chunk supports the answer.

**Team cases.** evaluate.py runs the app pipeline on the standard eval/cases.csv. Structured expectations default to an in-scope exam. Use {"marks":{"1":1},"review_reasons":{"17":"invalid_choice"}} to test one clear answer alongside a review. A genuinely unreadable exam passes only when that status was explicitly expected and no marks or total were proposed. A failed/unavailable reader is recorded as an execution issue. Narrative expectations remain unscored until human assessment.

With team cases present, the course command also writes component_results.json: it compares the two lexical designs, includes semantic comparison when both vector files are present, and recomputes judge agreement when saved labels and results exist. It detects eval/case_links.csv, train_sources.csv and grade_policy.json when provided. Outputs include an evaluation run ID and input fingerprints. Run the documented independent judge first to create judge_results.csv. Run timed grading without the cache.

**Agent trace.** Python searches using the question and OCR text and retries with the question alone when needed. Gemini sees previews and may request one full indexed passage before returning its proposal. The teacher view records the opened ID. A citation must come from text made visible to the model. The bound is two model calls per eligible written answer and 30 per exam. Three uncached requests on the same five-observation synthetic handoff exercised live grading. [Measurements](eval/component2_live_smoke.json)

~~~powershell
uv run --no-dev python evaluate.py --cases eval/cases.csv --links eval/case_links.csv --out eval/results
~~~

For the final written comparison, add --baseline-train eval/train_sources.csv. List the team's training exams under a single source_id header. The evaluator rejects overlap with linked evaluation exams, fits one mean per written question and rounds to the nearest half point. summary.json reports both systems on the exact same answers, alongside review counts and baseline results on all scorable answers.

For repeat evaluations, add --cache ../grading-private/grades.sqlite. This optional private file stores model responses and token counts. An input fingerprint covers the question, reference, OCR text, retrieved evidence, model, prompt and response schema. The teacher-review and citation rules still run on every replay. results.csv reports billed calls, tokens, cost and cache hits separately from the estimated tokens and cost without reuse. The economics command uses the latter when estimating deployment cost. Measure latency with the cache disabled for the final timing comparison.

The optional links CSV contains id,source_id. The first is the team's case ID and the second is the published anonymous ID such as Student_1. Include links only when you know the input is from that source exam. The runner will additionally write paired_marks.csv and report MCQ agreement with the key and written agreement with the teacher separately. Use a human transcription and then an OCR export on the same held-out answers to separate grading errors from vision errors. Leave all pages and variants of an exam in one split.

**Feedback judge.** Run evaluation with --private-exports ../grading-private to save the teacher views outside the deployable repository. Keep the associated OCR exports and item CSV there. In the item CSV, select 20+ written proposals with id,question_id,ocr_json,teacher_json. Each OCR file must be a vetted export covered by the server hash list. Use the same chunk design for grading and judging. Label each item yourself in the course's id,label,comment format using pass/fail. The judge receives the answer, question, proposed mark, feedback and supporting source. It receives neither human labels nor comments. It checks ID coverage before making calls and saves progress so an interrupted run can resume.

~~~powershell
uv run --no-dev python -m exam_grader.judge ../grading-private/judge_items.csv eval/human_labels.csv eval/judge_results.csv
~~~

**Review policy and economics.** Supply a JSON object of Policy fields (for example min_read, min_id, min_similarity) using evaluate.py --policy PATH. Write each run to a different --out folder. Compare validation runs on the same complete set of linked exams with exam_grader.costs. Its assumptions JSON takes hourly_eur, signoff_seconds for every answer, review_seconds extra for flagged answers, manual_seconds per answer, wrong_eur, accepted_catch_rate, reviewed_error_rate, manual_error_rate and exams_per_day. Use measured times and document the assumptions behind error rates. A catch rate of 0.25 means routine teacher sign-off catches one quarter of disagreements among proposals. Missing references are excluded from agreement. The projection assumes the measured disagreement rate generalizes to other proposals, and treats label disagreement as a proxy for error. Audit that assumption.

~~~powershell
uv run --no-dev python -m exam_grader.costs eval/policy_a eval/policy_b --assumptions eval/cost_assumptions.json
~~~

The output includes teacher minutes, reviews per day and projected euros per 1,000 answers for each candidate and for manual grading. Freeze the team's chosen thresholds before the final test. Set GRADE_POLICY_FILE to that same JSON when serving the app. For whole-product drift, the brief requires separate normal reference inputs and shifted cases. The available 50 exams alone cannot cover 30 disjoint full-exam test cases and a further 30 independent full-exam reference inputs. Agree on additional independent vetted inputs with the team. Do not reuse scans under new names.

The course entry points main.py, check.py and deploy.py are included unchanged. Once the team fills its evaluation files, run uv run python main.py reference, uv run python main.py evaluate, uv run python main.py drift-demo and uv run python check.py. Publish only from the agreed team fork with its configured Space and private log dataset.

## Current verification

The local Gradio /run endpoint processed the synthetic fixture over HTTP, and the professor's unmodified check_app step validated the resulting record. Engineering tests exercise signing, feedback, safe logging, provider structure with a mocked transport, evaluation, judge inputs, grade comparisons, cost handling and a deliberately shifted confidence signal. Live Gemini, vector quality, teacher-calibrated feedback ratings, drift on actual scans, Anna's real reader and the deployed Space remain to be tested. Classroom evaluation cases and human labels are the team's work. See HANDOFF.md and SOURCES.md.
