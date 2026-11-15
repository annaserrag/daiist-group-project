"""
Turn each chapter (vision/step*.py) into a notebook and run it, so its tables and figures can be read without
running anything: `uv run --group train --group dev python -m vision.make_notebooks [step1 step4 ...]`.

The .py chapters are the only source. Each is written in the percent format (`# %%` starts a code cell,
`# %% [markdown]` a text cell), which jupytext converts to a notebook cell by cell; nbconvert then runs it from
the top. Never edit a notebook by hand: change the chapter and run this again.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHAPTERS = sorted((ROOT / "vision").glob("step*.py"))
NOTEBOOKS = ROOT / "vision" / "notebooks"


def build(chapter: Path) -> Path:
    notebook = NOTEBOOKS / f"{chapter.stem}.ipynb"
    NOTEBOOKS.mkdir(exist_ok=True)
    subprocess.run([sys.executable, "-m", "jupytext", "--to", "ipynb", "--output", str(notebook), str(chapter)], check=True)
    # Run it in place from the repository root, with no time limit per cell (step 1 evaluates 11 detector runs).
    subprocess.run([sys.executable, "-m", "jupyter", "nbconvert", "--to", "notebook", "--execute", "--inplace",
                    "--ExecutePreprocessor.timeout=-1", f"--ExecutePreprocessor.cwd={ROOT}", str(notebook)], check=True)
    return notebook


if __name__ == "__main__":
    wanted = sys.argv[1:]
    for chapter in CHAPTERS:
        if not wanted or any(chapter.stem.startswith(w) for w in wanted):
            print(f"{chapter.stem} -> {build(chapter).relative_to(ROOT)}", flush=True)
