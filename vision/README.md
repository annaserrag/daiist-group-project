# Vision component: reading a scanned exam

The vision component turns a scanned exam (PDF or photo) into the list the grading agent receives: each answer under its question number, with a confidence. `reader.read_exam(path)` does it, and the product uses its default mode: **Qwen3-VL-2B, a pretrained vision-language model, reading each whole page**.

Getting there took 20 tries (`artifacts/vision/tries.csv`). The code tells that story in seven chapters. Each one asks a question, answers it with data, and ends with the conclusion that leads to the next chapter. Read them in order: as `.py` files, or as notebooks with every table and figure in `vision/notebooks/`.

## The story in seven chapters

| Chapter | The question | What we found | So next |
|---|---|---|---|
| [`step1_detectors`](step1_detectors.py) | Can we train detectors that find question numbers and lines? | Yes, on their own data: AP50 0.79 (numbers) and 0.88 (lines) on an unseen test booklet; rules 0.00 and 0.19. | Try them on real exams. |
| [`step2_first_reader`](step2_first_reader.py) | Do they work on real exams? | No: 11% of the questions found on the 50 Mendeley exams. No public data to retrain on. Three tries then tuned on Mendeley exams by mistake and are superseded. | Mendeley becomes evaluation-only; build a development set. |
| [`step3_development_set`](step3_development_set.py) | What can the reader's choices be made on? | 126 public images (SMHD, ASAP, School Notebooks, the step 1 answer sheets, short and blank pages), cleaned and explored. | Make every choice there. |
| [`step4_choose_reader`](step4_choose_reader.py) | Which reader, and with which rules? | Qwen3-VL-2B on the whole page: 7.3% character error and 89% of numbers found, against 19 to 28% and 23 to 56% for line readers. Safeguards for loops and blank pages. | It finds only 43% of numbers on answer sheets: try fine-tuning. |
| [`step5_finetune_data`](step5_finetune_data.py) | Is there data to fine-tune it on? | 1,012 training pages: real GNHK photos and synthetic exam pages, cleaned, with no leak into validation or the development set. | Train. |
| [`step6_finetune_reader`](step6_finetune_reader.py) | Does fine-tuning (LoRA) help? | Lower loss and slightly better text, but no more question numbers on answer sheets. | Keep the pretrained reader; freeze it. |
| [`step7_evaluate`](step7_evaluate.py) | How well does the frozen reader read real exams? | 86% of the answered questions found on 50 Mendeley exams, multiple-choice letters 99% right when read; detectors 11%, line reader 27%. | The grading agent's review rules (repo README). |

Two rules hold throughout:

- **Mendeley is evaluation-only.** Nothing is chosen by looking at the 50 Mendeley exams. Steps 2 and 7 are the only chapters that use them. Step 2 loads saved results; step 7 reads the exams once with the frozen reader.
- **Every dataset is cleaned and explored before it is used.** Each chapter that brings in data starts with what the source holds, what is wrong with it, what is removed or fixed (with counts), and how the result looks.

## Running the chapters

```bash
uv run --group train python -m vision.step1_detectors
uv run python -m vision.step2_first_reader
uv run python -m vision.step3_development_set
uv run python -m vision.step4_choose_reader
uv run --group train python -m vision.step5_finetune_data
uv run python -m vision.step6_finetune_reader
uv run python -m vision.step7_evaluate
uv run --group train --group dev python -m vision.make_notebooks   # all seven as executed notebooks
```

Downloads, built datasets, training runs and cached reads live in `~/.cache/daiist-vision/` (set `VISION_WORK` to move it). That keeps them out of the repo and out of the Space. The slow parts are cached, so a second run of any chapter takes minutes:

- **Step 1** reuses detector runs that already finished. `RETRAIN=1` trains the 11 runs again (4 to 7 hours on an Apple M5 Pro). `SMOKE=1` does one tiny run per detector to test the setup.
- **Steps 4 and 7** cache every read by image or exam and by reader version. A changed reader is never scored on old reads.
- **Step 6** shows the saved LoRA run. `RETRAIN=1` trains again (19 minutes on an RTX 5080, about 4 hours on an M5 Pro), and `RETRAIN=1 SMOKE=1` runs 3 steps on the heaviest pages to check memory. `RECOMPARE=1` reads the development set again with both readers. Stop a training run with Ctrl+C, not Ctrl+Z: a paused run keeps all its memory.

The notebooks are generated from the `.py` chapters, so change the chapter, never the notebook.

## The code

| File | Role |
|---|---|
| `step1_...` to `step7_...` | The chapters: the story, run top to bottom. |
| `reader.py` | The product: `read_exam(path, mode)` in four modes (`page`, the default; `ocr`; `detectors`; `rules`), and `answers_by_number()` / `choice_letter()` to match a read to the teacher's key. |
| `rules.py` | The no-learning baselines (step 1, and the reader's `rules` mode). |
| `datasets.py` | The two Roboflow datasets: download, index, the leakage evidence and the split by group (step 1). |
| `devset.py` | The development set: each source before selection, the cleaning, and `build()` (steps 3, 4, 6). |
| `metrics.py` | How a reader is scored on the development set, with the cache of reads (steps 4 and 6). |
| `eda.py` | Shared helpers so every chapter shows its tables and figures the same way. |
| `make_notebooks.py` | Turns the chapters into executed notebooks. |

`tests/test_vision_rules.py` checks the rules the chapters chose (question numbers, choice letters, blank pages, the scoring, the split) without any data.

## The data, and where to put it

| Data | Chapter | Licence | Put in `~/.cache/daiist-vision/` |
|---|---|---|---|
| Roboflow `answersheet/answer-sheet-component-detection` v1, `nucularvision/handwritten-text-line-dataset` v1 | 1, 3 | CC BY 4.0 | Downloaded by step 1 with `ROBOFLOW_API_KEY` from `.env`. |
| SMHD, Student Messy Handwritten Dataset (RMIT), `SMHD-forms.rar` from <https://doi.org/10.25439/rmt.24312715.v1> | 3 | CC BY-NC 4.0 | Extract into `devsets/smhd/x/`. |
| Handwritten ASAP-SAS (Gold and Zesch 2020), `Handwritten-ASAP-SAS-V1.0.zip` from <https://zenodo.org/records/8088866> | 3 | non-commercial research | Extract into `devsets/asap/x/`. |
| School Notebooks EN (ai-forever), `images.zip` and `annotations_*.json` from <https://huggingface.co/datasets/ai-forever/school_notebooks_EN> | 3 | MIT | Annotations in `devsets/school_notebooks/`, images extracted into `devsets/school_notebooks/x/`. |
| GNHK (Lee, Chung and Lee, ICDAR 2021), `data/*.parquet` from <https://huggingface.co/datasets/radhakrishnadeshpande/gnhk-pagelevel> | 5 | CC BY 4.0 | `finetune/gnhk/`. The original page, goodnotes.com/gnhk, now redirects to the app's home page; this copy keeps the original images and annotations. |
| OpenHand-Synth, `validation/*` from <https://huggingface.co/datasets/to-be/OpenHand-Synth> | 5 | CC BY 4.0 | `finetune/openhand/`. |
| EMNIST balanced (Cohen et al. 2017), `emnist-balanced-*` from <https://huggingface.co/datasets/Heliosoph/EMNIST> | 5 | CC BY 4.0 on this copy | `finetune/emnist/`. |
| Mendeley "A Dataset of Digitized Student Examination Papers" from <https://data.mendeley.com/datasets/sf3kvjwknt/1> | 2, 7 | CC BY 4.0 | `answerkey.txt` and `Teacher_manual_marks_Anonymized.csv` in `mendeley/`, the raw `Student_Pdf/Student_<n>.pdf` in `mendeley/pdfs/`. |

From the Mendeley download, leave three things where they are:

- **`Corrected_Pdf/`** has the teacher's ticks and marks drawn on the pages, which would show the reader the grading.
- **`Student_MCQ.csv`** and **`file.txt`** hold names and student ID numbers.
- **The `Student_Name` column** of the marks file is not anonymised, despite the file's name. Drop it from your copy. Step 7 also drops it when loading.

## What it writes

Every chapter writes to its own folder in `artifacts/vision/`, next to `tries.csv`, the log of all 20 tries. Its `chapter` column says where each try is told, and `uses_mendeley` marks the tries that looked at Mendeley exams.

| Folder | Main files |
|---|---|
| `step1_detectors/` | `dataset_summary.csv`, `eda_*.png`, `similarity_*` and `closest_cross_split_pairs.*` (the leakage evidence), `split.csv`, `leakage_after_split.csv`, `labels_*.png`, `baselines.csv`, `grid_results.csv`, `chosen.json`, `conf_sweep_val.csv`, `test_results.csv` |
| `step2_first_reader/` | the first run on Mendeley (`reader_eval_*_v1_all50.csv`) and what the superseded tries read (numbers only) |
| `step3_development_set/` | `cleaning.csv`, `composition.csv`, `devset.csv`, `eda_*.png` |
| `step4_choose_reader/` | `dev_readers.csv`, `dev_line_boxes.csv`, `dev_number_rule.csv`, `dev_decoding.csv`, `dev_choices.json`, `error_per_page.png`, and `dev_reads.csv` (every line every try read) |
| `step5_finetune_data/` | `finetune_data.json`, `eda_*.png`, and three synthetic pages with their text (GNHK photos are other people's notes and stay off the repo) |
| `step6_finetune_reader/` | `finetune_log.csv`, `finetune_result.json`, `training_curve.png`, `dev_lora_comparison.json` |
| `step7_evaluation/` | `reader_eval_summary.csv`, `reader_eval_questions.csv`, `pages.csv`, `eda_answers.png`, `found_rates.png`, and `mendeley_reads.csv` (without the text: every Mendeley page has the student's name in a handwritten header, so the text stays in `~/.cache/daiist-vision/reads/`) |

Weights: `weights/question_no_detector.pt` and `weights/text_line_detector.pt` (step 1), and `weights/page_reader_lora/` (step 6, kept but not used by default).

## Limits the grading agent must take into account

- **Confidence does not flag mistakes.** The page reader's confidence barely separates right from wrong letters (median 1.00 against 0.999). Better review signals are a question missing from the read, a question found twice, or a number not in the key.
- **Invented numbers.** The page reader reads about 4.5 numbers per exam that are not in the key.
- **English handwriting only.** Both the development set and the evaluation are in English; formulas and diagrams are untested.
- **Upright pages only.** Rotated photos are not straightened.
- **Printed question text** on the exam paper is read as if it were part of the answer.
- **Student names.** Anything above the first question number (often the student's name) goes to `unassigned`. It must not be logged or sent on.
- **Speed.** About 18 s an exam on an Apple GPU. On 2 CPU threads (a free Hugging Face Space) it takes about 85 s a page, with 13 GB of memory at its peak.
