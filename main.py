"""
Entry point. Don't rename it or its commands.

    uv run python main.py reference    # features of eval/reference.csv -> monitoring/reference_features.csv
    uv run python main.py evaluate     # evaluate.py or evaluate.ipynb
    uv run python main.py drift-demo   # normal and shifted inputs against the reference
    uv run python main.py export-logs  # the online app's logs (LOGS_DATABASE_URL) -> logs/
"""

import argparse
import csv
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

import env
import logbook
import monitor
import pipeline

ROOT = Path(__file__).resolve().parent
CASES = ROOT / "eval" / "cases.csv"
REFERENCE_INPUTS = ROOT / "eval" / "reference.csv"
SHIFTED_INPUTS = ROOT / "eval" / "shifted.csv"


def read_inputs(path: Path, kind: str | None = None) -> list[str]:
    """The `input` column of a CSV, optionally only the rows of one `kind`."""
    if not path.is_file():
        raise SystemExit(f"{path.relative_to(ROOT)} is missing. See eval/README.md.")
    with path.open(newline="", encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if (r.get("input") or "").strip()]
    if kind:
        rows = [r for r in rows if (r.get("kind") or "").strip().lower() == kind]
    return [r["input"].strip() for r in rows]


def features_of(inputs: list[str]) -> pd.DataFrame:
    """Run each input through the pipeline (not logged) and keep its features."""
    rows = []
    for i, raw in enumerate(inputs, 1):
        print(f"  {i}/{len(inputs)}", end="\r", flush=True)
        rows.append(pipeline.run(raw)["features"])
    return pd.DataFrame(rows)


def reference() -> None:
    inputs = read_inputs(REFERENCE_INPUTS)
    if len(inputs) < 30:
        print(f"WARNING: eval/reference.csv has {len(inputs)} inputs; use at least 30.")
    print(f"Computing the features of {len(inputs)} reference inputs...")
    table = features_of(inputs)
    monitor.REFERENCE_FILE.parent.mkdir(exist_ok=True)
    table.to_csv(monitor.REFERENCE_FILE, index=False)
    print(f"Wrote {monitor.REFERENCE_FILE.relative_to(ROOT)}: {len(table)} rows, features {', '.join(table.columns)}.")
    print("Commit it: the app and the drift demo read it.")


def drift_demo() -> None:
    ref = monitor.load_reference()
    if ref is None:
        raise SystemExit("No reference set. Run `uv run python main.py reference` first.")
    normal, shifted = read_inputs(CASES, kind="normal"), read_inputs(SHIFTED_INPUTS)
    if not normal or not shifted:
        raise SystemExit("Need `normal` cases in eval/cases.csv and inputs in eval/shifted.csv.")

    results = {}
    for name, inputs in (("normal", normal), ("shifted", shifted)):
        print(f"\n{name}: {len(inputs)} inputs")
        report = monitor.drift(ref, features_of(inputs))
        print(report.to_string(index=False))
        results[name] = bool(report["alert"].any())

    print(f"\nAlert rule: PSI > {monitor.PSI_ALERT} and > {monitor.NOISE_MULTIPLE} x the PSI expected by chance")
    print(f"normal inputs : {'ALERT (false alarm)' if results['normal'] else 'quiet'}")
    print(f"shifted inputs: {'ALERT' if results['shifted'] else 'quiet (missed)'}")
    if results["normal"] or not results["shifted"]:
        raise SystemExit("The alert must stay quiet on normal inputs and fire on shifted ones.")
    print("OK: quiet on normal inputs, fires on shifted ones.")


def export_logs() -> None:
    url = os.environ.get("LOGS_DATABASE_URL", "").strip()
    if not url:
        raise SystemExit(
            "Set LOGS_DATABASE_URL in .env to your database's External Database URL\n"
            "(Render: your database -> Connect -> External). See SETUP.md, \"Logs\"."
        )
    counts = logbook.export(url)
    print(f"Copied {counts['requests']} requests and {counts['feedback']} feedback rows to {logbook.LOG_DIR}.")
    print("Run `uv run python app.py` to see them in the Monitoring tab on your laptop.")


def run_entrypoint(name: str) -> None:
    py_path, nb_path = ROOT / f"{name}.py", ROOT / f"{name}.ipynb"
    if py_path.exists() and nb_path.exists():
        raise SystemExit(f"Found both {py_path.name} and {nb_path.name}. Delete the one you are not using.")
    if py_path.exists():
        subprocess.run([sys.executable, str(py_path)], cwd=ROOT, check=True)
        return
    if not nb_path.exists():
        raise SystemExit(f"Expected {py_path.name} or {nb_path.name} in {ROOT}.")
    with tempfile.NamedTemporaryFile(suffix=".py", dir=ROOT, delete=False) as tmp:
        tmp_path = Path(tmp.name)
    try:
        with tmp_path.open("w") as tmp_file:
            subprocess.run(
                ["jupyter", "nbconvert", "--to", "script", "--stdout", str(nb_path)],
                cwd=ROOT, check=True, stdout=tmp_file,
            )
        subprocess.run([sys.executable, str(tmp_path)], cwd=ROOT, check=True)
    finally:
        tmp_path.unlink(missing_ok=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("command", choices=["reference", "evaluate", "drift-demo", "export-logs"])
    args = parser.parse_args()
    env.load_env()
    {"reference": reference, "evaluate": lambda: run_entrypoint("evaluate"), "drift-demo": drift_demo,
     "export-logs": export_logs}[args.command]()
