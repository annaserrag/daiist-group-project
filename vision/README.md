# Vision component: reading a scanned exam

- `train_detector.py` trains the two detectors, the same way Assignment 2 trained the seal detector.
- `reader.py` turns any scanned exam into a list of questions and answers, the list the grading agent receives. It has four modes, from the rules baseline to a pretrained vision-language model that reads whole pages (the default).
- `devset.py` and `develop_reader.py` make every design choice of the reader on public handwriting that is not from the Mendeley exams.
- `evaluate_reader.py` evaluates the frozen reader once on the 50 Mendeley exams.

## The detectors

| Detector | What it finds | Trained on (Roboflow Universe, used as published) | Licence |
|---|---|---|---|
| question_no | the handwritten question number at the start of each answer | `answersheet/answer-sheet-component-detection` v1 | CC BY 4.0 |
| text_line | each handwritten line, so TrOCR can read it | `nucularvision/handwritten-text-line-dataset` v1 | CC BY 4.0 |

`train_detector.py` reads top to bottom in seven parts:
1. Download and describe the data.
2. Show **why we split it by booklet and by photo session**.
3. Build the datasets.
4. Run the baselines on validation: the rules in `rules.py` and COCO YOLO zero-shot.
5. Run the hyperparameter grid, choosing on validation only.
6. Save the chosen weights.
7. Test the chosen models and the baselines on the test split, once.

The split is train / validation / test by group, so no booklet, notebook or photo session is in two splits:

- **Answer sheets:** 8 booklets, split 5/2/1. Question numbers are scarce (about 280 in all), so most booklets go to train. Validation keeps two (one Science, one Social Science) because it chooses between the grid runs. Test keeps one Social Science booklet with 36 question numbers, so the detector test covers one booklet and one subject only.
- **Lines:** Roboflow's own split, except the WhatsApp session. That session is one writer's notebook photographed in 23 minutes, which Roboflow spread over all three splits, so all of it goes to train.

None of the 50 Mendeley exams is used here. They are the evaluation of the whole product, after everything has been chosen.

## Train it on a laptop

You need `ROBOFLOW_API_KEY` in `.env`. Downloads, built datasets and training runs go to `~/.cache/daiist-vision` (set `VISION_WORK` to change it), so they stay out of the repo and out of the Space. The script uses the Apple GPU (MPS) on a Mac, an NVIDIA GPU if there is one, and otherwise the CPU.

```bash
SMOKE=1 uv run --group train python vision/train_detector.py    # a few minutes: one tiny run per detector, to test the setup
uv run --group train python vision/train_detector.py            # the full grid, 11 runs: leave it overnight
REUSE=1 uv run --group train python vision/train_detector.py    # after an interruption: skips the runs that finished
```

The grid uses AdamW explicitly (lr0 0.002, the optimizer Ultralytics' `auto` mode would choose for runs this short). With `auto`, Ultralytics ignores `lr0`, so a learning-rate grid would train the same model several times. On an Apple M5 Pro an epoch takes roughly 13 to 21 s for question_no and 35 to 55 s for text_line (yolo11n to yolo11s), so the whole grid takes about 4 to 7 hours.

Keep the laptop plugged in and stop it from sleeping during the full run; on a Mac, `caffeinate -i uv run ...` does that. If MPS gives an error, `DEVICE=cpu` forces the CPU, but it is much slower. When it finishes, commit `weights/*_detector.pt` and `artifacts/vision/`.

## The reader

`reader.read_exam(path, mode)` takes a PDF or a photo and assumes nothing about the exam: not the number of questions, not which ones are multiple choice, not the pen colour. Every mode returns the same list of questions, each with its number, its text and a confidence.

| Mode | Finds the question numbers with | Reads with |
|---|---|---|
| `rules` | rules, no learning (`vision/rules.py`) | TrOCR base |
| `detectors` | the `question_no` detector | TrOCR base, on the `text_line` boxes |
| `ocr` | the text: a line that opens with a number | TrOCR base, on the `text_line` boxes |
| **`page`** (default) | the text, as in `ocr` | Qwen3-VL-2B, the whole page at once |

Which questions should be there, and which are multiple choice, comes from the teacher's answer key, not from the page. `answers_by_number()` and `choice_letter()` match the reader's output to the key. A choice letter is only taken when exactly one option letter stands on its own in the answer, otherwise the answer goes to review.

### How the design got here

1. **The detectors did not transfer.** The first run on the Mendeley exams (`reader_eval_*_v1_all50.csv`) found 12% of the questions with the detectors. The `question_no` detector scored AP50 0.79 on its own test split, but all its training groups come from one source ("13)"-style numbers on ruled answer sheets), and on the Mendeley scans its highest confidence on a page was 0.006.
2. **From then on, nothing was decided on Mendeley.** The 50 exams are evaluation-only, so every later choice was made on a development set of public handwriting (`devset.py`), and Mendeley was read once, at the end.
3. **The choices, on the development set** (`develop_reader.py`, `artifacts/vision/dev_*.csv`):

| Reader | Character error, SMHD pages | Character error, ASAP answers | Numbers found at line starts (SMHD) |
|---|---|---|---|
| TrOCR base on the detected lines | 27.6% | 22.7% | 23% |
| TrOCR large on the detected lines | 22.1% | 21.2% | 30% |
| Qwen3-VL-2B on the detected lines | 18.9% | 15.6% | 56% |
| **Qwen3-VL-2B on the whole page** | **7.3%** | **1.7%** | **89%** |
| Qwen3-VL-4B on the whole page | 6.5% | 1.3% | 85% |

4B reads text a little more accurately, but finds fewer question numbers, takes twice as long (about 10 s a page on an Apple GPU), and needs more memory than a free Space has. So the product uses 2B, and 4B stays as a documented try.

`Qwen/Qwen3-VL-2B-Instruct` (Apache-2.0, run locally, no key, never fine-tuned) transcribes each page line by line. Then:

- **Loops.** With plain greedy decoding it can fall into a loop on a page with little writing ("Cell", "Cell", "Cell"...): 703 lines that are not on the page, over 14 short and blank pages. `repetition_penalty=1.05` brought that to 12 without raising the error rate (`dev_decoding.csv`).
- **Blank pages.** A page whose ink (pixels 60 grey levels darker than its own background) is under 0.001 is not sent to the model. That is half the lightest written development page, and every blank page is at 0.
- **Question numbers.** A line that opens with a number and a separator (`26.`, `3b)`, `Q4:`) starts a question, after up to three stray marks (89% of the numbers found, 84% without them). Three more rules, which repaired misread numbers from the sequence, split a line at a later number, and accepted numbers with no separator, found no more numbers on the development set, so they were removed.
- **Line boxes for TrOCR (`ocr` mode).** A box much taller than a normal line is cut into rows at the emptiest band (error 27.6% vs 28.1%).

### What it assumes, and what it does not

It is meant to work for any teacher and any exam, so the exam-specific facts come from the teacher's key, never from the code:

- **Number of questions, and which are multiple choice:** from the key.
- **Question labels:** whatever the student wrote, as digits and an optional sub-part (`3`, `3b`, `3(b)`, `3.ii`). `answers_by_number(result, expected=key_labels)` matches them to the key. A sub-part the key doesn't have falls back to the plain number.
- **Options of a multiple-choice question:** `choice_letter(text, options)` takes them from the key (`ABCD`, `ABCDE`, `TF`...).

Limits that still apply to any exam:

- **English handwriting.** Both the development set and the evaluation are in English. Other languages, formulas and diagrams are untested.
- **Speed.** The page reader takes about 5 s a page on an Apple GPU, and about 85 s a page on 2 CPU threads (a free Hugging Face Space), with 13 GB of memory at its peak.
- **Invented text.** The repetition penalty stops loops, but the model can still write a line that is not on the page, with high confidence. Its token confidence does not tell those lines apart, so review must rely on checks against the key (missing or unexpected questions).
- **Upright pages.** Rotated photos are not straightened.
- **Printed question text** on the exam paper is read as if it were part of the answer.
- **Student names.** Anything written above the first question number (often the student's name) goes to `unassigned`. Don't log it or send it on, because the product must not store personal data.

```bash
uv run python -m vision.reader exam.pdf            # what it reads from one exam (page mode)
uv run python -m vision.reader exam.pdf ocr        # line detector + TrOCR
uv run python -m vision.reader exam.pdf detectors  # both YOLO models
uv run python -m vision.reader exam.pdf rules      # the no-learning baseline
uv run python -m vision.develop_reader             # every design choice, on the development set
uv run python -m vision.evaluate_reader            # all four modes on the 50 Mendeley exams, once the reader is frozen
```

## Every try, and why the product uses a pretrained vision-language model

The vision component started as two trained detectors plus TrOCR. It ended as a pretrained vision-language model (Qwen3-VL-2B, the kind of model Session 10 uses) that reads whole pages. `artifacts/vision/tries.csv` records all 16 tries in order: what each used, what it found, and what was decided. The column `uses_mendeley` marks the three tries that looked at Mendeley exams before the rule that they are evaluation-only. Those tries were superseded and are not reported as evaluation. In short:

1. **Training works on its own data.** The fine-tuned detectors beat the rules and COCO zero-shot by a wide margin on their test splits (AP50 0.79 for question numbers, 0.88 for lines).
2. **The question-number detector does not transfer.** It learnt one source of answer sheets and found 11% of the questions on the Mendeley exams. No public dataset with handwritten question numbers exists to retrain it on (step 5), and fine-tuning a reader on the same narrow source would repeat the problem (step 9).
3. **Reading is the bottleneck, and a pretrained page reader solves it.** On public development data, character error falls from 27.6% (TrOCR on the detected lines) to 7.3% (Qwen3-VL-2B on the page), and the question numbers found rise from 23% to 89%.
4. **It needs safeguards.** It invents text in a loop on almost empty pages until a repetition penalty stops it, its confidence does not flag its mistakes, and it is slow on a CPU. The grading agent's review rule has to take this into account.
5. **The trained line detector still plays a part.** It is the reading front end of the `ocr` mode, the baseline the page reader is compared against.

Every try's output is in `artifacts/vision/reads/`:

- `dev_reads.csv`: every line each reader and setting read on the development set.
- `mendeley_reads.csv`: what each mode found under each question number, without the text. The Mendeley pages carry the student's name and ID number in a handwritten header, which ends up inside the answers. So the text stays outside the repo, in `~/.cache/daiist-vision/reads/`.
- `superseded_mendeley_dev_reads.csv`: the same, for the superseded tries.

## The development set

Download, and extract each into its own folder under `~/.cache/daiist-vision/devsets/` (about 320 MB):

- `SMHD-forms.rar` from <https://doi.org/10.25439/rmt.24312715.v1> into `devsets/smhd/x/`: the Student Messy Handwritten Dataset (RMIT, CC BY-NC 4.0), student pages with line-by-line transcriptions.
- `Handwritten-ASAP-SAS-V1.0.zip` from <https://zenodo.org/records/8088866> into `devsets/asap/x/`: handwritten short answers with their text (Gold and Zesch 2020, non-commercial research use).

`devset.py` adds the Roboflow answer-sheet pages of the detectors' test booklet (question-number boxes, no text), real pages whitened below their second line, and blank pages. These parts check reading, question numbers, and invented text. No public dataset has whole exam pages with numbered answers and a key besides Mendeley, so handwritten multiple-choice lists like "7). B" are not in the development set.

## Evaluate it on the Mendeley exams

Download the dataset from <https://data.mendeley.com/datasets/sf3kvjwknt/1>. Put `answerkey.txt` and `Teacher_manual_marks_Anonymized.csv` in `~/.cache/daiist-vision/mendeley/`, and the raw `Student_Pdf/Student_<n>.pdf` files in `~/.cache/daiist-vision/mendeley/pdfs/`. Leave the rest of the download where it is:

- **`Corrected_Pdf/`** has the teacher's ticks and marks drawn on the pages, which would show the reader the grading.
- **`Student_MCQ.csv`** and **`file.txt`** hold names and student ID numbers.
- **The `Student_Name` column** of the marks file is not anonymised, despite the file's name. Drop it from your copy, because the evaluation never uses it.

In the teacher's sheet, `NC` (not corrected), `NM` and `-` mean the question was not answered.

`evaluate_reader.py` reads all 50 exams in all four modes, once the reader is frozen. It checks:
- Which questions in the key were found.
- For multiple choice, the letter read against the letter the teacher's sheet says each student wrote.
- For written answers, whether each answered question was found.
- How many numbers were read that aren't in the key.

Reads are cached, so an interrupted run continues where it stopped.

### Result: the frozen reader on all 50 exams, read once

| | rules | detectors | ocr | **page, 2B** (default) | page, 4B |
|---|---|---|---|---|---|
| Questions in the key found | 13.6% | 10.9% | 27.2% | **86.2%** | 83.4% |
| Written answers found | 3.7% | 20.2% | 24.8% | **83.0%** | 85.8% |
| MCQ letters right, of all answered | 4.6% | 0.5% | 11.8% | **81.4%** | 76.0% |
| MCQ letters right, when one was read | 75.0% | 55.6% | 83.6% | **98.9%** | 98.6% |
| Numbers read that are not in the key, per exam | 1.3 | 0.4 | 1.4 | 4.5 | 14.3 |
| Questions found twice, per exam | 0.9 | 0.6 | 0.6 | 2.1 | 2.1 |
| Seconds per exam (Apple GPU, median) | 17.3 | 11.5 | 6.2 | 18.3 | 54.9 |

The 4B result confirms the choice made on the development set.

The page reader's confidence barely separates right from wrong letters (median 1.00 vs 0.999), so it cannot decide review on its own. A question missing from the read, a duplicate, or a number not in the key are better review signals.

## What it writes

| File | What it is |
|---|---|
| `artifacts/vision/dataset_summary.csv` | images, photos, groups and boxes per Roboflow split |
| `artifacts/vision/similarity_summary.csv`, `similarity_by_pair_type.png` | the evidence for the split: how alike copies, same-group pairs and different-group pairs are |
| `artifacts/vision/closest_cross_split_pairs.csv`, `.png` | the closest different photos that Roboflow put in different splits |
| `artifacts/vision/split.csv`, `leakage_after_split.csv` | our split for every image, and the check that no photo or group is in two splits |
| `artifacts/vision/labels_question_no.png`, `labels_text_line.png` | a few built training images with their labels drawn on, to check them before training |
| `artifacts/vision/baselines.csv` | rules and COCO zero-shot on validation |
| `artifacts/vision/grid_results.csv`, `chosen.json` | every grid run (AP50 over all boxes, precision and recall at the reader's threshold, CPU ms per image), and the chosen one per detector |
| `artifacts/vision/conf_sweep_val.csv` | the chosen models' precision and recall on validation at several confidence thresholds |
| `artifacts/vision/test_results.csv` | the chosen models and the baselines on the test split |
| `artifacts/vision/tries.csv` | every try, in order: data, result, decision, evidence |
| `artifacts/vision/reads/` | what every try read (dev text; Mendeley without the text) |
| `artifacts/vision/dev_readers.csv`, `dev_line_boxes.csv`, `dev_number_rule.csv`, `dev_decoding.csv`, `dev_choices.json` | the reader's design choices on the development set |
| `artifacts/vision/reader_eval_summary.csv`, `reader_eval_questions.csv` | the frozen reader on the 50 Mendeley exams: rules vs detectors vs ocr vs page with 2B and 4B (the reported result) |
| `artifacts/vision/reader_eval_summary_v1_all50.csv`, `reader_eval_questions_v1_all50.csv` | the first run, rules vs detectors, which showed the detectors do not transfer |
| `weights/question_no_detector.pt`, `weights/text_line_detector.pt` | the chosen models |
