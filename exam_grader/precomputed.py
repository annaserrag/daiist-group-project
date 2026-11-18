"""
Reader outputs computed ahead of time, for the online app.

Why. The online app runs on Render's free plan, which has 512 MB of memory. The page reader (Qwen3-VL-2B through
torch and transformers) needs several GB, so it cannot run there, and the course allows precomputing a component's
outputs for the cases shown in the demo as long as the report says so. On a laptop with the reader installed the app
reads every PDF live, exactly as in the evaluation; online, where torch and transformers are not installed, it can
only use the readings stored here and sends any other PDF back with a message saying so.

Each file in data/precomputed_reads/ is the bridge's vision.exam.v1 payload for one demo PDF, named by the PDF's
SHA-256 and tagged with the reader version that produced it, so a reading from an older reader is never used. They
are made with `uv run --group reader python -m exam_grader.precomputed <pdf> ...` and hold the same text the reader
produces live, from PDFs that are already in the repo and already de-identified.

GRADE_READER=precomputed uses the stored readings even when the reader is installed, to test the online path locally.
"""

import json
import os
from importlib.util import find_spec
from pathlib import Path

from .integration import file_sha256

DIR = Path(__file__).resolve().parent / "data" / "precomputed_reads"
READER_PACKAGES = ("torch", "transformers")


def reader_installed() -> bool:
    """True when the live page reader can run here (a laptop), False online."""
    if os.getenv("GRADE_READER", "").strip().lower() == "precomputed":
        return False
    return all(find_spec(name) is not None for name in READER_PACKAGES)


def lookup(path, version: str) -> dict | None:
    """The stored reading of this exact PDF by this reader version, or None."""
    stored = DIR / f"{file_sha256(path)}.json"
    if not stored.is_file():
        return None
    data = json.loads(stored.read_text(encoding="utf-8"))
    return data["payload"] if data.get("reader_version") == version else None


def save(path, version: str, payload: dict, source: str) -> Path:
    DIR.mkdir(parents=True, exist_ok=True)
    out = DIR / f"{file_sha256(path)}.json"
    out.write_text(json.dumps({"source": source, "reader_version": version, "payload": payload},
                              indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def main(paths: list[str]) -> None:
    from . import vision_finetune

    if not reader_installed():
        raise SystemExit("The page reader is not installed. Run with `uv run --group reader ...`.")
    root = Path(__file__).resolve().parents[1]
    for raw in paths:
        pdf = Path(raw).resolve(strict=True)
        payload = vision_finetune.read_exam(pdf)
        out = save(pdf, vision_finetune.VERSION, payload, str(pdf.relative_to(root)))
        print(f"{pdf.relative_to(root)} -> {out.relative_to(root)} ({len(payload.get('answers', []))} answers)", flush=True)


if __name__ == "__main__":
    import sys

    import env

    env.load_env()
    main(sys.argv[1:])
