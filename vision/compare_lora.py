"""Compare the two page readers using Anna's development images and scoring rules."""
import gc
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch

from vision import develop_reader as dev
from vision import devset, reader


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run():
    rows = devset.build()
    counts = rows.groupby("kind").size().to_dict()
    expected = json.loads((dev.ARTIFACTS / "dev_choices.json").read_bytes())["dev_set"]
    if counts != expected:
        raise SystemExit(f"Development inputs incomplete: expected {expected}, found {counts}.")
    adapter = dev.ROOT / "weights/page_reader_lora/adapter_model.safetensors"
    adapter_sha = digest(adapter)
    model_names = {"base": "Qwen/Qwen3-VL-2B-Instruct", "lora": "Qwen/Qwen3-VL-2B-Instruct+weights/page_reader_lora"}
    report = {"base_revision": reader.PAGE_READER_REVISION, "adapter_sha256": adapter_sha,
              "counts": rows.groupby("kind").size().to_dict(), "scores": {}, "pages": []}
    print(report["counts"], flush=True)
    pages = []
    for row in rows.itertuples():
        pages.append({"kind": row.kind, "image": Path(row.image).name, "sha256": digest(Path(row.image)),
                      "expected": row.numbers, "boxes": int(row.n_boxes) if row.kind == "answersheet" else None})
    for arm, model in model_names.items():
        dev.READS = devset.DEVSETS / "paired_reads" / (reader.PAGE_READER_REVISION if arm == "base" else adapter_sha)
        def read(image):
            return dev.read_page(image, reader.VLM_GENERATE, model)
        for i, row in enumerate(rows.itertuples()):
            found = dev.numbers_read(read(row.image))
            pages[i][arm] = found
            if i % 10 == 0 or i + 1 == len(rows):
                print(arm, i+1, "/", len(rows), flush=True)
        scores = dev.score(rows, lambda row: read(row.image))
        report["scores"][arm] = {k: None if v != v else v for k,v in scores.items()}
        reader._vlm.cache_clear()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    for page in pages:
        if page["kind"] in ("smhd", "notebook"):
            expected = Counter(page["expected"])
            base, lora = (Counter(page[arm]) & expected for arm in model_names)
            page["gained"] = sorted((lora-base).elements())
            page["lost"] = sorted((base-lora).elements())
    report["pages"] = pages
    output = dev.ARTIFACTS / "dev_lora_comparison.json"
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(report["scores"], flush=True)


if __name__ == "__main__":
    run()
