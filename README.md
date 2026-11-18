# Group assignment: Build and run an AI product

As a team, build a small AI product for one business process, put it online, and run it like a product: it logs its own use, watches itself for drift, and you decide, with evidence, whether anyone should rely on it.

The product chains **two AI components from different modules of the course**: the output of one is the input of the other. Getting each piece to run is the easy part, and you have done most of it in assignments 1, 2, and 3. The hard part is making the chain work end to end, measuring it, and keeping an eye on it once it is live.

Start with [SETUP.md](SETUP.md).

## Teams

Teams are the Blackboard groups. Record who built what in [CONTRIBUTIONS.md](CONTRIBUTIONS.md) as you go, not on the last day.

The larger the team, the more it is expected to deliver. Mostly that means a better final product: a more careful evaluation, a more solid app, a better-argued decision. It also means more work in some parts, for example more models or settings tried and compared, more cases in the evaluation, or more useful views in the Monitoring tab.

## The task

1. **Business process and decision.** Describe a situation your application would be a good fit for: who uses the product, which decision or task it supports, what happens today without it (a **baseline** to beat), how you measure success, and roughly what it is worth in euros. Write this before you choose any model.
2. **Two connected AI components** from different modules:
   - tabular ML
   - vision
   - agentic LLM

   One component's output must be the next one's input, in one flow. Two demos in two tabs do not count. Pretrained models and APIs are fine, and you do not have to train anything from scratch. You may reuse your own code from assignments 1–3; say whose and where in REPORT.md.

3. **Deploy it** as a public Gradio app on Render's free plan, with an **App** tab and a **Monitoring** tab. API keys go in the service's environment variables on Render, never in the repo.
4. **Log every request**: a summary of the input, the output of each component, the model and prompt versions, how long it took, what it cost, and the user's 👍/👎. **No personal data** (names, emails, faces, ID numbers…) anywhere: not in the logs, and not in the examples you use to test and demo the app. The template's logger saves the online app's logs in a free Postgres database that Render creates with the app, because the app's disk is wiped when it restarts.
5. **Monitor it.** The Monitoring tab shows, from the logs:
   - at least one **quality signal over time** (share of 👎, share of cases sent to review, confidence…);
   - one **drift check**: are today's inputs different from the ones you tested with? For example, compare the distribution of a feature or of the model's confidence (you will see how in the monitoring session);
   - an **alert rule**, with a threshold you justify.

   `main.py drift-demo` replays inputs that you shifted on purpose (darker photos, another language, another customer segment…). Your alert must fire on them and stay quiet on normal inputs.
6. **Evaluate.**
   - Each component, with the metric that fits it (as in assignments 1–3).
   - The whole product, on **30+ cases** you write in `eval/cases.csv`, compared with the baseline from step 1.
   - Where the output is free text, an LLM judge that you check against 20+ of your own labels, as in assignment 3.
   - The cost per request, in tokens, euros and seconds.
7. **Decide.** Back to step 1, now with the numbers:
   - which cases go to a person: the rule that sends a case to review, and how many cases a day that would be;
   - **go / no-go / go with conditions**, the conditions, and the evidence behind them.
8. **Write REPORT.md, submit, and present.**

### Bonus: reinforcement learning (up to +10 %)

Add an RL layer that **changes how the app behaves based on the feedback the app itself collects**. A bandit is enough, for example:
- choosing among three prompt variants or answer styles and learning from 👍/👎;
- choosing the confidence threshold or the review rule and learning from reviewers' corrections;
- choosing which offer to show and learning from accept/reject.

The bonus counts only if REPORT.md and the demo show all of this:
1. **The loop closes:** feedback in the app → reward → the policy is updated → the app visibly behaves differently. The update can happen after every request or every so often from the logs.
2. **The reward is justified:** what it rewards, what it could reward by accident (think of reward hacking), and how long the reward takes to arrive.
3. **Exploration** (ε-greedy, UCB or Thompson sampling) is chosen and justified, including what exploring costs the users who get the worse option.
4. **Some of the learning comes from your app's real logs.** You will have little traffic, so you may warm the policy up with a simulator. Say which part is simulated and which part is real.
5. **The risks:** feedback loops, who gives feedback and who doesn't, and calibration.

A bandit that is wired in but never learns from real feedback earns nothing.

## Code layout

```bash
uv run python app.py                 # the app, at http://127.0.0.1:7860
uv run python main.py reference      # features of your reference inputs, for the drift check
uv run python main.py evaluate       # component and end-to-end evaluation
uv run python main.py drift-demo     # normal and shifted inputs; the alert must fire only on the shifted ones
uv run python check.py               # checks that everything works, as the professor will
```

There is no deploy command: once your fork is connected to Render (SETUP.md), every push to `main` puts the new version online.

The template works from day one with placeholder components (simple stand-ins that return fixed answers), so you can put it online straight away and replace the pieces one by one:

| File | What it does | Do you change it? |
|---|---|---|
| `pipeline.py` | Your two components and how the output of one goes into the other | Yes: replace the placeholders. Keep what `run()` returns, because the rest of the app reads it |
| `app.py` | The App and Monitoring tabs that users see | Yes, the screens. Keep `serve()` and the `/run` address, which `check.py` uses |
| `evaluate.py` | Runs your test cases through the product; you add the measures | Yes (or use `evaluate.ipynb`) |
| `monitor.py` | The charts and the alert of the Monitoring tab | Yes: what you track and when the alert fires |
| `logbook.py` | Saves every request and its 👍/👎 | You may add fields; don't remove any |
| `env.py`, `main.py`, `check.py` | Internal pieces that make the rest work | No |
| `render.yaml` | How Render builds and starts the app | Only `name` (and nothing else unless you know why) |

The input files for the evaluation and the drift check are described in [eval/README.md](eval/README.md).

**Your app must run on someone else's computer with someone else's keys**, because that is how it is graded. Use only the services listed in SETUP.md (ask first if you need another one), never write a key in the code (SETUP.md shows where keys go), and include in the repo every file the app needs, or explain in SETUP.md how to download it.

## What you submit

- **Final submission, Mon 30 Nov, 23:59**, once per team. One member forks this repo and adds the rest of the team as collaborators. Tag the final version `final` (`git tag final && git push --tags`), open a PR to https://github.com/ami232/daiist-group-project, and submit on Blackboard: the PR link, the app's link, and a zip of the repo as a backup. After the deadline you may only fix things that keep the app running.
- **Presentation** in class, 1–3 Dec. The order is published after the final submission.
- **Peer assessment**, each student on their own, on Blackboard after the presentations. Your teammates do not see it.


## Presentation

- 6 minutes, then 4 minutes of questions.
- Show the **live product online**, not screenshots: one normal case, one case it gets wrong or sends to review, and the Monitoring tab. A recorded backup video is allowed if the network fails.
- Every team member must be able to explain every part: the professor chooses who answers each question.
- While the other teams present, you rate each of them as a potential user or buyer: did you understand the problem, does it work, would you trust it, would you buy it. You submit your ratings on Blackboard at the end of each presentation session, and doing so for every team counts towards class participation.


## Grading

Anyone must be able to run your app from the `final` version, with their own keys, by following SETUP.md. If your online app happens to be down when it is graded, that is not a problem as long as this works.

| Part | Weight |
|---|---|
| Business case, baseline, value and go/no-go decision | 15% |
| The two components working together | 30% |
| Evaluation: each component, the whole product, cost | 20% |
| Running it: logging, monitoring, drift alert | 15% |
| Presentation, live demo and questions | 20% |
| *Bonus: reinforcement learning* | *up to +10%* |

The team grade cannot go above 100%. Each member's grade also takes into account their own part in the project: feedback from teammates and classmates, how they present and answer questions, and the evidence of their work in the repository and in CONTRIBUTIONS.md.

## AI use

Allowed for code; disclose it in REPORT.md. The decisions (the problem, the components and how they connect, the metrics, the drift threshold, the review rule, etc.), the test cases, your own labels and the report's analysis must be the team's.
