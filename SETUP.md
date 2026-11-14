# Setup

## Run the integrated exam grader from a fresh clone

Open the repository at the version you want to grade and run the commands below from its root. Use [uv](https://docs.astral.sh/uv/getting-started/installation/) and Python 3.12 (uv can install Python). The committed lock contains both the handwriting reader and the grader. Leave `GRADE_VISION_MODULE` and `GRADE_VISION_ROOT` unset. No second checkout, training set or fine-tuning is needed.

1. Install the inference dependencies.

   ```sh
   uv sync --locked --no-dev
   ```

2. Copy `.env.example` to `.env` (PowerShell: `Copy-Item .env.example .env`; macOS/Linux: `cp .env.example .env`). In this local file, enter your own active Gemini key and the following approved digest for the bundled, de-identified c06 PDF. c24 is marked locally. No Hugging Face token or log repository is needed for these checks. Keep secrets out of Git.

   ```dotenv
   GEMINI_API_KEY=<your key>
   GRADE_APPROVED_HASHES=ff35ee20848842d1d4e76031eceb282582cb30a217d21c9e654f18b360caeb73
   ```

3. Run the acceptance check. The first nonblank PDF downloads the pinned public Qwen3-VL-2B-Instruct checkpoint from Hugging Face (several GB). Allow disk space for the model and PyTorch, an internet connection, and Gemini quota for Q21. The YOLO detector weights and course index are in this repo. The reader selects CUDA, Apple MPS or CPU depending on availability. Windows x64 uses the locked CUDA 13 wheels; a compatible NVIDIA driver enables GPU acceleration. Other hardware can have different runtimes and readings. The live Gemini response may vary.

   ```sh
   uv run --locked --no-dev python -m exam_grader.check_integration
   ```

   The check uses the same PDF processing and request logger as the app, looks up expectations after grading and exits nonzero if either case fails. **c06** should propose Q18 = 1, Q19 = 0, Q20 = 1 and Q21 = 2 with a course citation. **c24** should propose Q1 = 1 and refer Q2 as unreadable. Both PDFs are excerpts, so neither should receive a final total. An unavailable written-grading provider or an unapproved c06 leaves Q21 for review.

4. Start Gradio and open the local address it prints. In **App**, upload `eval/inputs/c06.pdf` and then `eval/inputs/c24.pdf`. The app also has **Materials** and **Monitoring** tabs.

   ```sh
   uv run --locked --no-dev python app.py
   ```

The two-PDF check covers integration. For the course-wide evaluation and drift requirements see [the evaluation guide](eval/README.md) and `uv run --locked --no-dev python check.py`. The detector training commands and the optional two-checkout development workflow are in [the reader handoff](HANDOFF.md).

## API keys

You may use these services. If you need another one, ask first: the professor has to run your app with his own keys.

| Service | Variable | Free key |
|---|---|---|
| Gemini (LLMs, embeddings, vision) | `GEMINI_API_KEY` | <https://aistudio.google.com/apikey> |
| Roboflow (datasets and hosted models) | `ROBOFLOW_API_KEY` | <https://app.roboflow.com> → Settings → API Keys |
| Hugging Face (Space, logs, models) | `HF_TOKEN` | <https://huggingface.co/settings/tokens>, with *write* permission |

Local models (Hugging Face `transformers`, Ultralytics YOLO, scikit-learn, PyTorch) need no key.

```bash
cp .env.example .env    # paste your keys into .env
```

Never commit `.env`. Read keys only from the environment (`os.environ`). In `.env.example`, list **every** variable your app needs, with a comment saying what it is for. If a key is missing, the app should say which one instead of crashing with a long error.

Each member can use their own free keys. Free tiers have daily limits (for example, about 1,000 Gemini embeddings a day per project), so cache what you can: embeddings, model outputs in the evaluation, and anything else you would otherwise recompute.

## Your Space

1. One member creates a Hugging Face account and a new Space: <https://huggingface.co/new-space>, SDK **Gradio**, hardware **CPU basic** (free), visibility **public**. Add the others as members of the Space or of an organisation.
2. In your `.env`, set `HF_TOKEN` and `HF_SPACE=<user>/<space-name>`, then:

   ```bash
   uv run python deploy.py --secrets
   ```

   `--secrets` copies every variable in `.env.example` that is set in your environment into the Space's secrets (you can also add them by hand in the Space's *Settings → Variables and secrets*). Next time, `uv run python deploy.py` alone is enough unless a key changed.

   This uploads the app, writes the `requirements.txt` that Spaces needs from your `uv.lock`, and adds the Space's configuration. Do not edit the Space's files by hand: always change the repo and run `deploy.py` again.
3. Open the Space and check the build log (*Logs*) if it does not start.

A Space on free hardware **goes to sleep when nobody uses it** and takes a minute or more to wake up. Open it before your presentation.

A model that needs a GPU will not run on CPU basic. Use a smaller model, an API, or precompute its outputs for the cases you demo and say so in REPORT.md.

## Logs

A Space's disk is wiped every time it restarts, so the logger in the template (`logbook.py`) writes each request to `logs/` and, if `HF_LOG_REPO` is set, pushes that folder every few minutes to a **private** Hugging Face dataset.

1. Create the dataset: <https://huggingface.co/new-dataset>, visibility **private**, for example `<user>/<space-name>-logs`.
2. Add `HF_LOG_REPO=<user>/<space-name>-logs` to `.env` and to the Space's secrets.

Locally, without `HF_LOG_REPO`, logs stay in `logs/`, which is git-ignored. The Monitoring tab reads them from wherever they are.

Do not change the log fields the template defines. `check.py` and the grading rely on them. You may add your own.

## Dependencies

`uv add <package>`, then commit `pyproject.toml` and `uv.lock`. `deploy.py` takes the Space's `requirements.txt` from them.

## Check your submission

```bash
uv run python check.py                    # keys, evaluation files, one request from start to finish, logs, drift demo
uv run python check.py --space <url>      # also wakes your Space and sends it one request
```

`check.py` uses the keys in your environment, exactly as the professor will use his. If it fails for you, it will fail when grading too. Before you tag `final`, run it in a **fresh clone** of your fork, with a `.env` holding only the keys listed in `.env.example`. That is the quickest way to find a file you forgot to commit.
