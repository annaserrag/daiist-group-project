"""
Fine-tune the page reader (Qwen3-VL-2B) on exam-like handwriting with LoRA, on a laptop.

LoRA leaves the model's own weights as they are and trains small low-rank matrices added to the language model's
attention and MLP layers (about 1% of the parameters); the image encoder is not touched. The data is built by
vision/finetune_data.py: GNHK photos and synthetic exam pages, none from the development set or Mendeley.

Run with `uv run --group train python -m vision.finetune_reader` (add `--smoke` for 3 steps on the heaviest pages,
to check speed and memory). Stop it with Ctrl+C, not Ctrl+Z: a paused run keeps all its memory. The script is meant
to be read top to bottom, like train_detector.py:

    Part 1  Data        each page becomes the reader's prompt and the text it should write; only the answer's
                        tokens are trained on
    Part 2  Model       Qwen3-VL-2B in bfloat16, LoRA on the language model, gradient checkpointing, and a
                        memory ceiling
    Part 3  Train       AdamW, warm-up then cosine decay, gradient accumulation; the log and the adapter are saved
                        as it trains
    Part 4  Validation  loss on the validation pages before and after, the same pages each time
    Part 5  Save        the adapter (WORK/finetune/adapter and weights/page_reader_lora/) and the result

Whether the fine-tuned reader is better is decided on the development set (vision/develop_reader.py), not here
and not on Mendeley.
"""

# %% Setup
import gc
import json
import math
import random
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
from PIL import Image

from vision import datasets, reader

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "vision"
BUILT = datasets.WORK / "finetune" / "built"
ADAPTER = datasets.WORK / "finetune" / "adapter"
WEIGHTS = ROOT / "weights" / "page_reader_lora"
LOG = ARTIFACTS / "finetune_log.csv"
BASE = reader.PAGE_READER
SMOKE = "--smoke" in sys.argv
SEED = 0

# LoRA: rank 16 on every attention and MLP projection of the language model. The image encoder stays frozen: it
# already sees handwriting well (the errors on the development set are in what is written down, not in seeing).
LORA = {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05,
        "target_modules": r".*language_model.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)"}
EPOCHS = 1
LR = 1e-4
ACCUMULATE = 8     # pages per optimiser step (one page at a time fits in memory)
WARMUP = 10        # optimiser steps
N_VAL = 60         # validation pages scored before and after (30 GNHK, 30 synthetic)
SMOKE_STEPS = 3    # optimiser steps in a --smoke run, on the heaviest pages
SAVE_EVERY = 25    # optimiser steps between saves of the adapter, so a stopped run is not lost
# Memory. A page takes 1,500 to 4,700 tokens (a full A4 page at the reader's width is 2,802 image tokens), and one
# page at a time peaks between 8 and 21 GB on an Apple GPU. The first full run did not hand freed memory back, so
# it grew to 57 GB and the Mac swapped: now the cache is emptied after every page, and the process may use at most
# this share of the GPU's recommended maximum (about 26 GB on a 48 GB Mac). Past it, PyTorch stops with an
# out-of-memory error instead of swapping.
MEMORY_FRACTION = 0.7
LINE = "=" * 78


def section(title: str) -> None:
    print(f"\n{LINE}\n{title}\n{LINE}")


def release() -> None:
    """Give the GPU memory of the last page back, so pages of different sizes do not pile up in the cache."""
    gc.collect()
    if reader.device() == "mps":
        torch.mps.empty_cache()


def gpu_memory() -> float:
    return torch.mps.driver_allocated_memory() / 2**30 if reader.device() == "mps" else float("nan")


# %% Part 1. Data
def load(split: str) -> list[dict]:
    return [json.loads(line) for line in open(BUILT / f"{split}.jsonl")]


def encode(processor, example: dict) -> dict:
    """The prompt the reader sends (page image and VLM_PAGE_PROMPT) followed by the text it should answer. Labels
    are -100 on the prompt, so the loss is only on the answer."""
    image = Image.open(example["image"]).convert("RGB")
    user = [{"role": "user", "content": [{"type": "image", "image": image},
                                         {"type": "text", "text": reader.VLM_PAGE_PROMPT}]}]
    full = user + [{"role": "assistant", "content": [{"type": "text", "text": example["text"]}]}]
    prompt = processor.apply_chat_template(user, tokenize=True, add_generation_prompt=True, return_dict=True,
                                           return_tensors="pt")
    inputs = processor.apply_chat_template(full, tokenize=True, return_dict=True, return_tensors="pt")
    n = prompt["input_ids"].shape[1]
    assert torch.equal(inputs["input_ids"][0, :n], prompt["input_ids"][0]), "the prompt is not a prefix"
    labels = inputs["input_ids"].clone()
    labels[0, :n] = -100
    return {**inputs, "labels": labels}


def page_loss(model, processor, example: dict) -> torch.Tensor:
    batch = {k: v.to(reader.device()) for k, v in encode(processor, example).items()}
    return model(**batch).loss


def validation_loss(model, processor, pages: list[dict]) -> dict:
    model.eval()
    losses = {"gnhk": [], "exam": []}
    with torch.no_grad():
        for example in pages:
            losses[example["kind"]].append(page_loss(model, processor, example).item())
            release()
    model.train()
    return {kind: round(sum(v) / len(v), 4) for kind, v in losses.items() if v}


def heaviest(pages: list[dict], n: int) -> list[dict]:
    """The tallest pages with the longest text: the most tokens, so the most memory."""
    return sorted(pages, key=lambda p: (Image.open(p["image"]).height, len(p["text"])), reverse=True)[:n]


if __name__ == "__main__":
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    random.seed(SEED)
    torch.manual_seed(SEED)
    section("Part 1. Data")
    train, val = load("train"), load("val")
    val_pages = ([p for p in val if p["kind"] == "gnhk"][:N_VAL // 2] + [p for p in val if p["kind"] == "exam"][:N_VAL // 2])
    if SMOKE:
        val_pages = val_pages[:2] + val_pages[-2:]
    print(f"train {len(train)} pages, validation {len(val)} pages ({len(val_pages)} scored)")

    # %% Part 2. Model
    section("Part 2. Model")
    if reader.device() == "mps":
        torch.mps.set_per_process_memory_fraction(MEMORY_FRACTION)
        print(f"GPU memory ceiling: {MEMORY_FRACTION * torch.mps.recommended_max_memory() / 2**30:.1f} GB")
    processor = AutoProcessor.from_pretrained(BASE)
    model = AutoModelForImageTextToText.from_pretrained(BASE, dtype=torch.bfloat16).to(reader.device())
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(task_type="CAUSAL_LM", **LORA))
    model.print_trainable_parameters()

    # %% Part 4 (before). Validation loss of the pretrained reader
    section("Part 4a. Validation loss before training")
    started = time.perf_counter()
    before = validation_loss(model, processor, val_pages)
    print(before, f"({time.perf_counter() - started:.0f} s)")

    # %% Part 3. Train
    section("Part 3. Train")
    params = [p for p in model.parameters() if p.requires_grad]
    optimiser = torch.optim.AdamW(params, lr=LR, weight_decay=0.0)
    total = SMOKE_STEPS if SMOKE else EPOCHS * math.ceil(len(train) / ACCUMULATE)
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimiser, lambda s: min(1.0, (s + 1) / WARMUP) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))
    order = (heaviest(train, SMOKE_STEPS * ACCUMULATE) if SMOKE
             else [p for _ in range(EPOCHS) for p in random.sample(train, len(train))])
    if not SMOKE:
        LOG.write_text("step,loss,lr,minutes,peak_gb\n")
    step, running, peak, started = 0, [], 0.0, time.perf_counter()
    model.train()
    for k, example in enumerate(order[:total * ACCUMULATE], start=1):
        loss = page_loss(model, processor, example)
        (loss / ACCUMULATE).backward()
        running.append(loss.item())
        del loss
        peak = max(peak, gpu_memory())
        release()
        if k % ACCUMULATE == 0 or k == len(order):
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optimiser.step()
            schedule.step()
            optimiser.zero_grad()
            step += 1
            seconds = time.perf_counter() - started
            row = {"step": step, "loss": round(sum(running) / len(running), 4), "lr": schedule.get_last_lr()[0],
                   "minutes": round(seconds / 60, 1), "peak_gb": round(peak, 1)}
            running, peak = [], 0.0
            if not SMOKE:
                with open(LOG, "a") as f:
                    f.write(",".join(str(v) for v in row.values()) + "\n")
                if step % SAVE_EVERY == 0:
                    model.save_pretrained(ADAPTER)
            if step % 10 == 0 or SMOKE or step == total:
                print(f"step {step}/{total}  loss {row['loss']:.3f}  {seconds / step:.0f} s/step  "
                      f"eta {(total - step) * seconds / step / 60:.0f} min  peak {row['peak_gb']:.1f} GB")

    # %% Part 4 (after). Validation loss of the fine-tuned reader
    section("Part 4b. Validation loss after training")
    after = validation_loss(model, processor, val_pages)
    print(pd.DataFrame({"before": before, "after": after}).to_string())

    # %% Part 5. Save
    section("Part 5. Save")
    if SMOKE:
        print("smoke run: nothing saved")
    else:
        model.save_pretrained(ADAPTER)
        shutil.copytree(ADAPTER, WEIGHTS, dirs_exist_ok=True)
        (ARTIFACTS / "finetune_result.json").write_text(json.dumps(
            {"base": BASE, "lora": LORA, "epochs": EPOCHS, "lr": LR, "accumulate": ACCUMULATE, "steps": total,
             "train_pages": len(train), "val_pages": len(val_pages), "val_loss_before": before,
             "val_loss_after": after, "minutes": round((time.perf_counter() - started) / 60, 1)}, indent=2))
        print(f"adapter saved to {ADAPTER} and {WEIGHTS}; log in {LOG}")
