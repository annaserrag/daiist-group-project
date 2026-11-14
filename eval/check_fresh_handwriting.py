"""Measure frozen LoRA transcription on unused public Handwritten ASAP-SAS short answers."""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from vision import develop_reader as dev
from vision import devset, reader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    info = pd.read_csv(devset.ASAP/"information"/"prompt-3.txt", sep="\t", comment="#", header=None,
                       names=["file", "id", "text", "pen", "status"], quoting=3)
    digest = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    dev_pages = devset.asap_answers()
    dev_files = {Path(r["image"]).name for r in dev_pages}
    frozen = json.loads((reader.ROOT/"artifacts/vision/dev_lora_comparison.json").read_bytes())
    excluded = {p["sha256"] for p in frozen["pages"]}
    dev_text = {dev.normalise(r["truth"]) for r in dev_pages}
    pool = []
    for r in info.itertuples():
        path = devset.ASAP/"prompt-3"/r.file
        if r.status != "ok" or r.file in dev_files or not path.is_file() or dev.normalise(r.text) in dev_text:
            continue
        fingerprint = digest(path)
        if fingerprint not in excluded:
            pool.append((fingerprint, path, r.text))
    # Fixed selection independent of legibility and results.
    selected = sorted(pool)[:12]
    if len(selected) != 12 or len({h for h, _, _ in selected}) != 12:
        raise ValueError("Incomplete or duplicate held-out images.")
    observations = []
    for fingerprint, path, truth in selected:
        page = reader.read_exam_page(path)
        ordered = [row["text"] for row in page["unassigned"]]
        ordered.extend(f"{row['number_text']} {row['text']}".strip() for row in page["questions"])
        expected = dev.normalise(truth)
        errors = dev.edit_distance(dev.normalise(" ".join(ordered)), expected)
        observations.append({"image": path.name, "sha256": fingerprint, "chars": len(expected), "edits": errors,
                             "unexpected_question_ids": [q["number"] for q in page["questions"]]})
    chars = sum(p["chars"] for p in observations)
    edits = sum(p["edits"] for p in observations)
    result = {"reader": reader.PAGE_READER, "adapter_sha256": digest(reader.WEIGHTS/"page_reader_lora/adapter_model.safetensors"),
              "source": "Handwritten ASAP-SAS, prompt 3, public transcriptions", "source_url": "https://zenodo.org/records/8088866",
              "selection": "First 12 unique eligible SHA-256 values in sorted order. Excludes development filenames, image hashes and normalized reference texts.",
              "eligible_images": len(pool), "images": observations, "reference_characters": chars, "character_edits": edits,
              "pooled_character_error_rate": edits/chars, "exact_transcriptions": sum(p["edits"] == 0 for p in observations)}
    args.output.write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(f"{edits}/{chars} character edits; {result['exact_transcriptions']}/{len(observations)} exact")


if __name__ == "__main__":
    main()
