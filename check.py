"""
Submission check. `uv run python check.py [--url URL]`

This is what the professor runs, with their own API keys. It checks the variables
listed in .env.example, the files in eval/, that the app starts and serves one
request end to end with a valid log record, and that the drift demo passes.
With --url it also wakes your deployed app and sends it one request.

It never writes to your logs: the test request goes to a temporary folder.
Don't edit this file to make it pass.
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
EVAL = ROOT / "eval"
CASE_COLUMNS = ["id", "input", "kind", "expected", "baseline", "notes"]
KINDS = {"normal": 1, "hard": 5, "review": 5, "out_of_scope": 3}  # minimum per kind
MIN_CASES, MIN_REFERENCE, MIN_SHIFTED, MIN_HUMAN_LABELS = 30, 30, 10, 20
DRIFT_TIMEOUT = 900
URL_TIMEOUT = 600
SKIP_DIRS = {".git", ".venv", "logs", "__pycache__", ".ipynb_checkpoints"}

TOTAL_STEPS = 5


def fail(message: str) -> None:
    print(f"\nFAIL: {message}", file=sys.stderr)
    sys.exit(1)


def warn(message: str) -> None:
    print(f"WARNING: {message}")


def step(n: int, title: str) -> None:
    print(f"\n=== {n}/{TOTAL_STEPS}  {title} ".ljust(72, "=") + "\n", flush=True)


def read_csv(path: Path) -> tuple[list[str], list[dict]]:
    if not path.is_file():
        fail(f"{path.relative_to(ROOT)} is missing. See eval/README.md.")
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader.fieldnames or []), [r for r in reader if any((v or "").strip() for v in r.values())]


def check_env() -> None:
    import env

    env.load_env()
    declared = env.declared()
    if not declared:
        fail(".env.example lists no variables. Keep it: it is how the professor knows which keys to set.")
    for name, optional in declared:
        state = "set" if os.environ.get(name, "").strip() else "not set"
        print(f"{name:<20} {state}{'  (optional)' if optional else ''}")
    if missing := env.missing():
        fail(
            f"Not set: {', '.join(missing)}.\n"
            "  Put them in .env (copy .env.example) or in the environment. If one is not\n"
            "  needed to run the app, mark its line in .env.example with (optional)."
        )


def check_eval() -> str:
    """Check the CSVs in eval/. Returns the first `normal` input, for the test request."""
    columns, cases = read_csv(EVAL / "cases.csv")
    if missing := [c for c in CASE_COLUMNS if c not in columns]:
        fail(f"eval/cases.csv is missing the column(s): {', '.join(missing)}. See eval/README.md.")
    kinds = [r["kind"].strip().lower() for r in cases]
    problems = []
    if len(cases) < MIN_CASES:
        problems.append(f"{len(cases)} cases; at least {MIN_CASES} are required")
    for kind, minimum in KINDS.items():
        if kinds.count(kind) < minimum:
            problems.append(f"{kinds.count(kind)} `{kind}` cases; at least {minimum} are required")
    if unknown := sorted(set(kinds) - set(KINDS)):
        problems.append(f"unknown kind(s) {', '.join(unknown)}; use {', '.join(KINDS)}")
    if len({r['id'] for r in cases}) != len(cases):
        problems.append("case ids are not unique")
    for r in cases:
        value = r["input"].strip()
        if value.startswith("eval/") and not (ROOT / value).is_file():
            problems.append(f"case {r['id']}: file {value} does not exist")
    if problems:
        fail("eval/cases.csv: " + "; ".join(problems) + ".")
    print(f"eval/cases.csv: {len(cases)} cases, " + ", ".join(f"{kinds.count(k)} {k}" for k in KINDS) + ".")

    for name, minimum in (("reference.csv", MIN_REFERENCE), ("shifted.csv", MIN_SHIFTED)):
        columns, rows = read_csv(EVAL / name)
        if "input" not in columns:
            fail(f"eval/{name} needs an `input` column.")
        if len(rows) < minimum:
            fail(f"eval/{name} has {len(rows)} inputs; at least {minimum} are required.")
        print(f"eval/{name}: {len(rows)} inputs.")

    labels = EVAL / "human_labels.csv"
    graded = len(read_csv(labels)[1]) if labels.is_file() else 0
    if graded < MIN_HUMAN_LABELS:
        warn(f"eval/human_labels.csv has {graded} labels. If you use an LLM judge, the brief asks for {MIN_HUMAN_LABELS}+.")
    return next(r["input"].strip() for r in cases if r["kind"].strip().lower() == "normal")


def check_app(test_input: str) -> None:
    log_dir = tempfile.mkdtemp(prefix="check-logs-")
    os.environ["LOG_DIR"] = log_dir
    # LOG_DIR set means files, so the test request never reaches your online database.
    started = time.perf_counter()
    try:
        import app
        import logbook
    except Exception as error:  # noqa: BLE001 - any failure to import is the finding
        fail(f"app.py does not start: {type(error).__name__}: {error}\n  Run `uv run python app.py` to see the full error.")
    print(f"app.py loads ({time.perf_counter() - started:.1f}s).")

    print(f"Test request: {test_input[:100]!r}")
    started = time.perf_counter()
    try:
        record = app.serve(test_input)
    except Exception as error:  # noqa: BLE001
        fail(f"app.serve() failed on the first `normal` case: {type(error).__name__}: {error}")
    print(f"Served in {time.perf_counter() - started:.1f}s. Review: {record['review']}. Outputs:")
    print("  " + str(record["outputs"])[:600])

    logs = logbook.load()
    if logs.empty or record["id"] not in set(logs["id"]):
        fail("The request was not logged. app.serve() must call logbook.log_request().")
    if problems := logbook.validate(record):
        fail("The log record does not match logbook.REQUIRED_FIELDS: " + "; ".join(problems))
    if len(record["outputs"]) < 2:
        warn("The log shows fewer than two component outputs in `outputs`. Log each component's output.")
    print(f"Logged with all required fields. Features: {', '.join(record['features']) or 'none'}.")
    if not record["features"]:
        fail("The record has no features, so the drift check has nothing to watch.")


def check_drift() -> None:
    if not (ROOT / "monitoring" / "reference_features.csv").is_file():
        fail("monitoring/reference_features.csv is missing. Run `uv run python main.py reference` and commit it.")
    try:
        result = subprocess.run(
            [sys.executable, "main.py", "drift-demo"], cwd=ROOT, capture_output=True, text=True, timeout=DRIFT_TIMEOUT,
        )
    except subprocess.TimeoutExpired:
        fail(f"`main.py drift-demo` did not finish within {DRIFT_TIMEOUT}s.")
    print(result.stdout.strip()[-2500:])
    if result.returncode != 0:
        fail(f"`main.py drift-demo` failed:\n{(result.stderr or result.stdout).strip()[-1500:]}")


def check_repo() -> None:
    size = 0
    for path in ROOT.rglob("*"):
        if path.is_file() and not SKIP_DIRS.intersection(path.relative_to(ROOT).parts) and path.name != ".env":
            size += path.stat().st_size
    print(f"Repo size without logs and .venv: {size / 1e6:.1f} MB.")
    report = (ROOT / "REPORT.md").read_text(encoding="utf-8")
    url_line = next((line for line in report.splitlines() if line.startswith("- **App URL**")), "")
    if "http" not in url_line:
        warn("REPORT.md has no App URL yet.")
    contributions = (ROOT / "CONTRIBUTIONS.md").read_text(encoding="utf-8")
    if "| | | |\n| | | |" in contributions:
        warn("CONTRIBUTIONS.md still has empty member rows.")


def check_url(url: str, test_input: str) -> None:
    from gradio_client import Client, handle_file

    url = url.strip().rstrip("/") + "/"
    if not url.startswith("http"):
        url = "https://" + url
    deadline = time.time() + URL_TIMEOUT
    print(f"Waking {url} (a sleeping app takes about a minute)...")
    while True:
        try:
            client = Client(url, verbose=False)
            break
        except Exception as error:  # noqa: BLE001 - keep trying until the deadline
            if time.time() > deadline:
                fail(f"Could not reach {url} within {URL_TIMEOUT}s: {type(error).__name__}: {error}")
            time.sleep(15)
    argument = handle_file(str(ROOT / test_input)) if (ROOT / test_input).is_file() else test_input
    started = time.perf_counter()
    try:
        result = client.predict(argument, api_name="/run")
    except Exception as error:  # noqa: BLE001
        fail(f"The deployed app's /run endpoint failed: {type(error).__name__}: {error}")
    print(f"The deployed app answered in {time.perf_counter() - started:.1f}s:\n  {str(result)[:600]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--url", metavar="URL", help="also send one request to your deployed app (https://<name>.onrender.com)")
    args = parser.parse_args()
    if args.url:
        global TOTAL_STEPS
        TOTAL_STEPS += 1

    step(1, "Environment variables")
    check_env()
    step(2, "Evaluation files")
    test_input = check_eval()
    step(3, "The app serves one request and logs it")
    check_app(test_input)
    step(4, "Drift demo")
    check_drift()
    step(5, "Repo")
    check_repo()
    if args.url:
        step(6, "Your deployed app")
        check_url(args.url, test_input)
    print("\nAll checks passed.")


if __name__ == "__main__":
    main()
