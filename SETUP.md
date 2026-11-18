# Setup

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
| Hugging Face (models; only if one needs a token) | `HF_TOKEN` | <https://huggingface.co/settings/tokens>, with *read* permission |

Local models (Hugging Face `transformers`, Ultralytics YOLO, scikit-learn, PyTorch) need no key, but only small ones like scikit-learn fit in the online app (see "Put it online").

```bash
cp .env.example .env    # paste your keys into .env
```

Never commit `.env`. Read keys only from the environment (`os.environ`). In `.env.example`, list **every** variable your app needs, with a comment saying what it is for. If a key is missing, the app should say which one instead of crashing with a long error.

Each member can use their own free keys. Free tiers have daily limits (for example, about 1,000 Gemini embeddings a day per project), so cache what you can: embeddings, model outputs in the evaluation, and anything else you would otherwise recompute.

## Put it online

The app runs as a free web service on [Render](https://render.com). Render builds it from your fork using `render.yaml`, and every push to `main` puts the new version online.

1. Before the first deploy, change `name` in `render.yaml` to your product's name (lowercase, with hyphens) and push. Your app's address will be `https://<name>.onrender.com`. If someone already has that name, Render adds a few characters to it.
2. The member who owns the fork creates a free Render account: <https://dashboard.render.com/register>, with **GitHub**. No credit card is needed.
3. In Render, click **New → Blueprint**, then **Connect GitHub**. GitHub asks which repositories Render may see: choose **Only select repositories** and pick your fork. Back in Render, select the fork. Do not paste the fork's address as a public repository instead: Render would deploy it once, but not again when you push.
4. Give the Blueprint any name, paste the keys your app uses in the values Render asks for (leave the others empty), and click **Deploy Blueprint**.
5. The first build takes a few minutes. Your app's address is at the top of the service's page. If the build or the app fails, the service's **Logs** page says why.

Only the member who created the service sees it on Render. The others don't need to: pushing to `main` is enough to put a new version online. On the free plan, a new version can take up to 15 minutes to go online. If it hasn't after 20 minutes, open the service on Render, cancel the deploy and click **Manual Deploy → Deploy latest commit**. Don't push a fix just before your presentation.

Keys: if your app needs a new one later, add it in the service's **Environment** page on Render (and to `.env.example`). Render does not read new keys from `render.yaml` after the first deploy.

The free service has limits that matter for your design:

- **512 MB of memory and a small share of one CPU.** PyTorch, `transformers` and YOLO don't fit. Use an API for those components, or precompute their outputs for the cases you demo and say so in REPORT.md. Locally, you can still run them in your evaluation.
- **It goes to sleep after 15 minutes without visits**, and the first visit after that takes about a minute. Open it a few minutes before your presentation.
- **Its disk is wiped** every time it restarts or goes to sleep. That is why the logs go to a database (next section).

## Logs

`logbook.py` saves every request and every 👍/👎. Where it saves them depends on where the app runs:

- **Online**, in a free Postgres database that Render creates with your app (it is declared in `render.yaml`, and Render gives the app its address in `DATABASE_URL`). The app's own disk is wiped on every restart, sleep and redeploy; the database is not.
- **On your laptop**, in JSON files in `logs/`, which is git-ignored. Leave `DATABASE_URL` empty in `.env`, so your tests don't mix with real use.

The Monitoring tab reads the logs from wherever the app is running.

To see the online logs on your laptop (for your analysis, or before the database expires):

1. On Render, open the database (`daiist-group-project-logs` or the name you gave it) → **Connect** → **External**, and copy the **External Database URL**.
2. Put it in `.env` as `LOGS_DATABASE_URL=...` (never commit it), then:

   ```bash
   uv run python main.py export-logs
   ```

   This writes `logs/requests-export.jsonl` and `logs/feedback-export.jsonl`. `uv run python app.py` then shows them in the Monitoring tab.

**Render deletes free databases 30 days after they are created.** A database created now lasts beyond the presentations, but export the logs after the presentation if you want to keep them. A Render account can have only one free database at a time.

Do not change the log fields the template defines. `check.py` and the grading rely on them. You may add your own.

## Dependencies

`uv add <package>`, then commit `pyproject.toml` and `uv.lock`. Render installs exactly what is in `uv.lock`.

## Check your submission

```bash
uv run python check.py                    # keys, evaluation files, one request from start to finish, logs, drift demo
uv run python check.py --url <url>        # also wakes your online app and sends it one request
```

`check.py` uses the keys in your environment, exactly as the professor will use his. If it fails for you, it will fail when grading too. Before you tag `final`, run it in a **fresh clone** of your fork, with a `.env` holding only the keys listed in `.env.example`. That is the quickest way to find a file you forgot to commit.
