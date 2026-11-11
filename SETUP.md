# Setup

**Exam grading project:** from the repo root, run `uv sync --locked --no-dev` and `uv run --locked --no-dev python app.py`. “Try synthetic OCR example” works without a key. For the private retrieval worksheet run `uv run --locked --no-dev python -m exam_grader.worksheet`. For live written marking set GEMINI_API_KEY locally; only server-approved, vetted inputs may be sent. The public corpus is included. See [Component 2 instructions](README.md) for the vector build, approval hashes, course evaluation and test commands, and [the integration contract](HANDOFF.md) to configure the team's PDF reader. Until that reader is connected, queued PDFs cannot receive marks. The general course setup follows.

## Install

Install [uv](https://docs.astral.sh/uv/getting-started/installation/). One member forks this repo and adds the rest of the team as collaborators (*Settings → Collaborators*). Each of you clones the fork and runs `uv sync`.

Then:

```bash
uv run python app.py
```

You should see the template's placeholder app at <http://127.0.0.1:7860>, with its App and Monitoring tabs.

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
