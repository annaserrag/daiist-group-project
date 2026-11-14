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

| Reader | Character error, SMHD scans | Character error, ASAP answers | Character error, notebook photos | Numbers found at line starts (SMHD) |
|---|---|---|---|---|
| TrOCR base on the detected lines | 27.6% | 22.7% | 50.9% | 23% |
| TrOCR large on the detected lines | 22.1% | 21.2% | 45.3% | 30% |
| Qwen3-VL-2B on the detected lines | 18.9% | 15.6% | 31.9% | 56% |
| **Qwen3-VL-2B on the whole page** | **7.3%** | **1.7%** | **5.1%** | **89%** |
| Qwen3-VL-4B on the whole page | 6.5% | 1.3% | 4.0% | 85% |

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

The vision component started as two trained detectors plus TrOCR. It ended as a pretrained vision-language model (Qwen3-VL-2B, the kind of model Session 10 uses) that reads whole pages. `artifacts/vision/tries.csv` records all 20 tries in order: what each used, what it found, and what was decided. The column `uses_mendeley` marks the three tries that looked at Mendeley exams before the rule that they are evaluation-only. Those tries were superseded and are not reported as evaluation. In short:

1. **Training works on its own data.** The fine-tuned detectors beat the rules and COCO zero-shot by a wide margin on their test splits (AP50 0.79 for question numbers, 0.88 for lines).
2. **The question-number detector does not transfer.** It learnt one source of answer sheets and found 11% of the questions on the Mendeley exams. No public dataset with handwritten question numbers exists to retrain it on (step 5), and fine-tuning a reader on the same narrow source would repeat the problem (step 9).
3. **Reading is the bottleneck, and a pretrained page reader solves it.** On public development data, character error falls from 27.6% (TrOCR on the detected lines) to 7.3% (Qwen3-VL-2B on the page), and the question numbers found rise from 23% to 89%.
4. **It needs safeguards.** It invents text in a loop on almost empty pages until a repetition penalty stops it, its confidence does not flag its mistakes, and it is slow on a CPU. The grading agent's review rule has to take this into account.
5. **The trained line detector still plays a part.** It is the reading front end of the `ocr` mode, the baseline the page reader is compared against.
6. **Fine-tuning the page reader did not pay off where it matters.** A LoRA adapter trained on public handwriting and synthetic exam pages (steps 18 to 20) read characters slightly better on the development set but found fewer question numbers on answer sheets, so the product keeps the pretrained reader.

Every try's output is in `artifacts/vision/reads/`:

- `dev_reads.csv`: every line each reader and setting read on the development set.
- `mendeley_reads.csv`: what each mode found under each question number, without the text. The Mendeley pages carry the student's name and ID number in a handwritten header, which ends up inside the answers. So the text stays outside the repo, in `~/.cache/daiist-vision/reads/`.
- `superseded_mendeley_dev_reads.csv`: the same, for the superseded tries.

## The development set

Download, and extract each into its own folder under `~/.cache/daiist-vision/devsets/` (about 690 MB):

- `SMHD-forms.rar` from <https://doi.org/10.25439/rmt.24312715.v1> into `devsets/smhd/x/`: the Student Messy Handwritten Dataset (RMIT, CC BY-NC 4.0), student pages with line-by-line transcriptions.
- `Handwritten-ASAP-SAS-V1.0.zip` from <https://zenodo.org/records/8088866> into `devsets/asap/x/`: handwritten short answers with their text (Gold and Zesch 2020, non-commercial research use).
- `images.zip` and `annotations_*.json` from <https://huggingface.co/datasets/ai-forever/school_notebooks_EN> into `devsets/school_notebooks/`, with the images extracted into `devsets/school_notebooks/x/`: phone photos of English school notebooks, every word transcribed and grouped by line (MIT). Teacher comments are left out of the text.

`devset.py` adds the Roboflow answer-sheet pages of the detectors' test booklet (question-number boxes, no text), real pages whitened below their second line, and blank pages. These parts check reading, question numbers, and invented text. Handwritten lists are rare in public data: SMHD has 21 pages with numbered lists, and School Notebooks 2. No public dataset has whole exam pages with numbered answers and a key besides Mendeley, so handwritten multiple-choice lists like "7). B" are not in the development set. A set of handwritten a/b/c/d answers exists on IEEE DataPort (CC BY 4.0, <https://ieee-dataport.org/documents/handwritten-text-dataset>) but needs a subscription.

## Fine-tuning the page reader (tried, not adopted)

This try fine-tunes the page reader (Qwen3-VL-2B, LoRA) on exam-like handwriting. `finetune_data.py` builds its training data, with nothing from the Mendeley exams or the development set, so the development set can judge the fine-tuned reader against the pretrained one and Mendeley stays a clean evaluation. `finetune_reader.py` trains the adapter. The adapter trained well, and it reads characters slightly better on the development set. But it finds fewer question numbers on answer sheets, so the product keeps the pretrained reader (see Result below).

Download into `~/.cache/daiist-vision/finetune/` (about 2 GB), then run `uv run --group train python -m vision.finetune_data`:

- `data/*.parquet` from <https://huggingface.co/datasets/radhakrishnadeshpande/gnhk-pagelevel> into `finetune/gnhk/`: GNHK (Lee, Chung and Lee, ICDAR 2021; CC BY 4.0), 687 phone photos of English handwriting from four continents (notes, diaries, shopping lists) with every word boxed and transcribed. The original page, goodnotes.com/gnhk, now redirects to the app's home page; this copy keeps the original images and annotations, and the text is rebuilt from the annotations here.
- `validation/*` from <https://huggingface.co/datasets/to-be/OpenHand-Synth> into `finetune/openhand/`: synthetic handwriting lines (CC BY 4.0). Only single English lines are used, split by writer (85% train, 15% validation).
- `emnist-balanced-*` from <https://huggingface.co/datasets/Heliosoph/EMNIST> into `finetune/emnist/`: handwritten digits and letters (Cohen et al. 2017, from NIST SD 19; CC BY 4.0 on this copy). EMNIST's own train and test files go to our train and validation pages.

| | pages, train | pages, validation | text lines (train) | numbered lines (train) |
|---|---|---|---|---|
| GNHK, real photos (its own train / test split, less leaks) | 512 | 161 | 6,842 | 236 |
| Synthetic exam pages | 500 | 100 | 11,630 | 7,368 |

A synthetic page is an A4 answer sheet, white or ruled. It has questions numbered from a random start, written with EMNIST digits and a pen-drawn '.' or ')', and about 15% of written answers get a sub-part ('2a)'). Each question is answered with an EMNIST letter A to D (40%), or with one to three OpenHand lines (60%). The target text follows the reader's prompt: one line per written line, numbers kept at the start. On GNHK, unreadable words (`%NA%`) and drawn symbols (`%math%`) are painted out of the image, crossed-out words (`%SC%`) stay on the page but not in the text, and printed words are left out, as the reader is asked for the handwriting only.

Checks: three sample pages with their text are in `artifacts/vision/finetune_samples/` (synthetic only; the GNHK photos are other people's notes and stay out of the repo). No text line of 25 or more characters is shared with the development set. Two problems were found and fixed by looking at the pages. 118 GNHK photos were stored on their side with a camera rotation tag, and their word boxes are drawn on the upright page, so the tag is now applied. Some OpenHand lines had grey scanner backgrounds that showed as boxes, so their backgrounds are now brought to white. A leakage check found that GNHK's own split is by photo, not by page: some notebook pages were photographed twice, once in each split (eng_AS_063 in test is eng_AS_044 in train, framed differently). The 11 test pages that share a line of 20 or more characters with a training page are dropped from validation, and 3 files found twice in train are kept once. Between our OpenHand train and validation lines no writer is shared, and only 5 short sentences (place names) are, written by different writers. One limit remains: EMNIST characters are 28 pixels, so the numbers look thicker and blurrier than pen writing, and a few are ambiguous (an EMNIST '8' can look like 'P').

### Training (`finetune_reader.py`)

Run `caffeinate -i uv run --group train python -m vision.finetune_reader` (`caffeinate -i` keeps the Mac awake), or add `--smoke` for 3 steps on the 24 heaviest pages, which check speed and memory and save nothing. Stop a run with Ctrl+C, not Ctrl+Z: a paused run keeps all its memory.

- **What trains.** LoRA of rank 16 on every attention and MLP projection of Qwen3-VL-2B's language model: 17.4 million parameters, 0.8% of the model. The image encoder and the model's own weights stay frozen. The adapter takes 67 MB.
- **How it trains.** One page at a time in bfloat16, with gradient checkpointing and 8 pages per optimiser step. AdamW at 1e-4, with 10 warm-up steps then cosine decay, for one epoch: 127 steps over the 1,012 pages. The loss is only on the text the reader should write, not on the prompt.
- **Validation loss.** It is measured on the same 60 validation pages (30 GNHK, 30 synthetic) before and after training.
- **Output.** The adapter goes to `weights/page_reader_lora/`. The loss log goes to `artifacts/vision/finetune_log.csv` and the settings and validation losses to `finetune_result.json`.
- **Memory.** A page takes 1,500 to 4,700 tokens, mostly image tokens: a full A4 page at the reader's width is 2,802 of them. The first full run kept the memory of every page in PyTorch's cache, grew to 57 GB on a 48 GB Mac and swapped for hours without finishing. Now the cache is emptied after every page, and the process may use at most 70% of the GPU's recommended maximum (26 GB on a 48 GB Mac). Past that it stops with an out-of-memory error instead of swapping. The log (`finetune_log.csv`, with each step's peak memory) is written as it trains, and the adapter is saved every 25 steps.
- **Smoke test on the 24 heaviest pages, on an M5 Pro with 48 GB.** It peaked at 21.1 GB and took about 17 s per page. A full epoch should take about 4 hours. The first smoke test ran on light pages only (7.5 s and 24 GB with no cache release) and underestimated both.
- **Fewer synthetic pages.** There are 500 synthetic pages instead of 1,000. The pretrained reader already reads them almost perfectly (validation loss 0.24, against 1.03 on GNHK), and halving them cut the run from about 6 to about 4 hours.
- **Loss before training.** The pretrained reader's validation loss is 1.03 on GNHK and 0.24 on the synthetic pages. It already reads the synthetic pages almost perfectly, so most of what the adapter can learn is on the real GNHK pages.

**How it is judged.** `reader.py` loads a model named `model+folder` as that model with the adapter merged in, so reading is as fast as before. When `weights/page_reader_lora/` exists, `develop_reader.py` adds "Qwen3-VL-2B + LoRA, page" to its comparison of readers. Whether the product switches is decided there, on the development set. Mendeley is read only after that choice is made.

### Result

**Training.** The full run was done on an NVIDIA RTX 5080 with the script unchanged: 127 steps in 19 minutes (the `peak_gb` column of the log is 0 there, as it measures Apple GPU memory only). The training loss fell from 0.53 (mean of the first 10 steps) to 0.40 (last 20 steps). The validation loss on the same 60 pages fell on both kinds of page:

| validation pages | before | after |
|---|---|---|
| GNHK, real photos | 1.03 | 0.62 |
| synthetic exam pages | 0.28 | 0.15 |

**Development set** (`artifacts/vision/dev_lora_comparison.json`, `vision/compare_lora.py` on the integration branch). The same 126 images and scoring as `develop_reader.py`, with the same pinned base model in both arms:

| | pretrained | LoRA |
|---|---|---|
| character error, SMHD | 7.2% | 6.7% |
| character error, ASAP | 1.65% | 1.57% |
| character error, notebooks | 4.9% | 4.5% |
| question numbers found, SMHD | 85% | 88% |
| question numbers found, notebooks | 100% | 100% |
| question numbers found, answer sheets | **43%** | **39%** |
| numbers read that are not on the page, per page | 0.20 | 0.07 |
| invented lines | 12 | 10 |

**Decision: keep the pretrained reader.** The adapter learnt what it was shown: GNHK's notes and clean synthetic pages, so it transcribes about half a point better and invents far fewer question numbers. But on the answer sheets, the pages closest to the exams the product grades, it finds fewer question numbers, and a missed number sends the answer to the teacher however well it is spelled. The answer-sheet set is small (21 pages), so the drop may be partly noise, but the fine-tuned reader shows no gain where the product needs one. The pretrained reader is also simpler (no adapter file to ship and check). The adapter is kept, and `reader.py` can still load it by name (`Qwen/Qwen3-VL-2B-Instruct+weights/page_reader_lora`).

The integration team also ran both readers through the grading pipeline on the 30 Mendeley PDFs. There, the pretrained reader also did better, but that comparison is not used for this decision: the bridge between reader and grader was built around the pretrained reader's output, and its LoRA-specific fixes were developed on those same PDFs.

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
| `artifacts/vision/finetune_data.json`, `finetune_samples/` | the fine-tuning data: pages and lines per source and split, and three synthetic pages with their text |
| `artifacts/vision/finetune_log.csv`, `finetune_result.json` | the LoRA run: loss and learning rate per step, and the settings and validation losses before and after |
| `artifacts/vision/dev_lora_comparison.json` | pretrained against LoRA reader on the development set: scores, and the question numbers each found on every page |
| `artifacts/vision/reader_eval_summary_v1_all50.csv`, `reader_eval_questions_v1_all50.csv` | the first run, rules vs detectors, which showed the detectors do not transfer |
| `weights/question_no_detector.pt`, `weights/text_line_detector.pt` | the chosen models |
