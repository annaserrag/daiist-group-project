"""
Vision component, training pipeline: the two detectors that find where answers are on an exam page.

    question_no detector   finds the handwritten question number at the start of each answer
    text_line detector     finds each handwritten line, so TrOCR can read it

Run with `uv run --group train python vision/train_detector.py` on a laptop: it uses the Apple GPU (MPS) or an
NVIDIA GPU if there is one, else the CPU. The full grid takes several hours, so leave it running (see vision/README.md).
The script is meant to be read top to bottom, like Assignment 2:

    Part 1  The data             download the two public datasets and describe them
    Part 2  Why this split       the leakage evidence, then a split by booklet and by photo session
    Part 3  YOLO datasets        one dataset per detector, built from our split
    Part 4  Baselines            simple rules, and COCO YOLO zero-shot
    Part 5  Finetuning grid      a small planned hyperparameter grid, chosen on validation only
    Part 6  Save                 the chosen weights and every table and figure, in artifacts/vision/
    Part 7  Test                 the chosen models and the baselines on the test split, once

Nothing from the Mendeley exams is used here: they are the final evaluation of the whole product
(vision/evaluate_reader.py and evaluate.py).

Switches (environment variables):
    SMOKE=1    one tiny run per detector on CPU, to check the code end to end (numbers are meaningless)
    REUSE=1    skip training for any grid run whose weights already exist (runs live in VISION_WORK),
               so an interrupted run can be restarted where it stopped
    DEVICE=cpu force a device (default: mps on an Apple laptop, cuda if there is an NVIDIA GPU, else cpu)
"""

# %% Setup
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import env  # noqa: E402
from vision import datasets  # noqa: E402

env.load_env()
pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "artifacts" / "vision"
WEIGHTS = ROOT / "weights"
RUNS = datasets.WORK / "runs"          # outside the repo (see vision/datasets.py)
BUILT = datasets.WORK / "built"

SMOKE = os.environ.get("SMOKE") == "1"
REUSE = os.environ.get("REUSE") == "1"
SEED = 0
EPOCHS, PATIENCE = (1, 1) if SMOKE else (40, 10)
IMGSZ = 320 if SMOKE else 640
# Parts 1 to 3 are real in every mode. In SMOKE mode, the baselines, grid and weights go to VISION_WORK
# instead of the repo, so a test run can never be committed by mistake.
RESULTS = datasets.WORK / "smoke" if SMOKE else ARTIFACTS
if SMOKE:
    WEIGHTS = datasets.WORK / "smoke"
    RUNS = datasets.WORK / "smoke" / "runs"  # its own folder: a 1-epoch run must never be picked up by REUSE=1
for folder in (ARTIFACTS, RESULTS, WEIGHTS, RUNS):
    folder.mkdir(parents=True, exist_ok=True)


def pick_device() -> str:
    if os.environ.get("DEVICE"):
        return os.environ["DEVICE"]
    import torch
    if torch.cuda.is_available():
        return "0"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = "cpu" if SMOKE else pick_device()
print(f"Training device: {DEVICE}")
os.chdir(datasets.WORK)  # Ultralytics downloads yolo11n.pt / yolo11s.pt into the working folder; keep them out of the repo


def section(title: str) -> None:
    print(f"\n{'=' * 100}\n{title}\n{'=' * 100}")


# %% Part 1. The data
section("Part 1. The data: two public Roboflow datasets, used as published")

folders = {name: datasets.download(name) for name in datasets.SOURCES}
index = pd.concat([datasets.index_images(n, f) for n, f in folders.items()], ignore_index=True)

describe = (index.groupby(["dataset", "roboflow_split"])
            .agg(images=("file", "size"), source_photos=("source", "nunique"), groups=("group", "nunique"),
                 images_without_boxes=("n_boxes", lambda s: int((s == 0).sum())), boxes=("n_boxes", "sum"))
            .reset_index())
print(describe.to_string(index=False))
copies = (index.groupby(["dataset", "source"]).size().rename("copies").reset_index()
          .groupby(["dataset", "copies"]).size().rename("photos").reset_index())
print("\nHow many exported images each source photo became (augmented copies, and tiles for the lines set):")
print(copies.to_string(index=False))
describe.to_csv(ARTIFACTS / "dataset_summary.csv", index=False)

# %% Part 2. Why this split
section("Part 2. Why we split by booklet and by photo session, not by image")

vectors = datasets.embed(index["path"].tolist(), datasets.WORK / "embeddings.npz")
summary, crossing = datasets.similarity_evidence(index, vectors)
print("Cosine similarity of image features, by type of pair (higher = more alike):")
print(summary.round(3).to_string(index=False))
print("\nHow to read this: two augmented copies of the same photo are the reference for 'the model has already seen this'.")
for name, part in summary.groupby("dataset", sort=False):
    med = part.set_index("pair")["median"]
    copy, group, other = med.get("copy of the same photo"), med.get("same group, different photo"), med.get("different group")
    verdict = ("so the group, not the photo, must stay on one side of the split"
               if group is not None and other is not None and group > other else "so groups add little beyond photos here")
    print(f"  {name}: copies {copy:.3f}, same group {group if group is not None else float('nan'):.3f}, "
          f"different groups {other:.3f} (medians), {verdict}.")

print("\nClosest pairs of DIFFERENT photos that Roboflow put in DIFFERENT splits:")
print(crossing[["dataset", "similarity", "a", "a_split", "b", "b_split", "same_group"]]
      .assign(a=lambda d: d["a"].str[:38], b=lambda d: d["b"].str[:38]).round(3).to_string(index=False))

sessions = index[index["group"] == datasets.WHATSAPP_SESSION].groupby("roboflow_split")["source"].nunique()
print(f"\nThe WhatsApp session (one writer, one notebook, 23 minutes) in Roboflow's split: {sessions.to_dict()}")
booklets = index[index["dataset"] == "answersheet"].groupby("group")["source"].nunique().sort_values(ascending=False)
print(f"Answer-sheet pages per booklet (Roboflow puts all of them in train):\n{booklets.to_string()}")

datasets.plot_similarity(index, vectors, ARTIFACTS / "similarity_by_pair_type.png")
datasets.contact_sheet(crossing, ARTIFACTS / "closest_cross_split_pairs.png")
summary.to_csv(ARTIFACTS / "similarity_summary.csv", index=False)
crossing.drop(columns=["a_path", "b_path"]).to_csv(ARTIFACTS / "closest_cross_split_pairs.csv", index=False)

split = datasets.make_split(index)
check = datasets.leakage_after_split(split)
print("\nOur split, by group:")
print(split[split["used"]].groupby(["dataset", "our_split"])
      .agg(images=("file", "size"), photos=("source", "nunique"), groups=("group", "nunique"), boxes=("n_boxes", "sum"))
      .reset_index().to_string(index=False))
print("\nSources and groups found in more than one of train, val and test (must be 0):")
print(check.to_string(index=False))
assert check["in_more_than_one_split"].sum() == 0, "a photo or group is in more than one split"
split.drop(columns=["path", "label"]).to_csv(ARTIFACTS / "split.csv", index=False)
check.to_csv(ARTIFACTS / "leakage_after_split.csv", index=False)

# %% Part 3. One YOLO dataset per detector
section("Part 3. One YOLO dataset per detector")
# Two detectors, not one with two classes: the answer-sheet pages have lines that nobody labelled, and the
# notebook photos have no question numbers labelled. One combined model would learn that unlabelled lines
# and numbers are background. Each detector is trained only on data where its class is fully labelled.
DETECTORS = {
    "question_no": dict(dataset="answersheet", class_map={4: 0}),  # class 4 is question_no in the export
    "text_line": dict(dataset="lines", class_map={0: 0}),
}
yamls = {}
for det, spec in DETECTORS.items():
    yamls[det] = datasets.write_yolo(split, spec["dataset"], spec["class_map"], [det], BUILT / det)
    datasets.label_sheet(BUILT / det, ARTIFACTS / f"labels_{det}.png")  # look at the labels before training
    print(f"{det:12s} -> {yamls[det]}   (labels drawn in {(ARTIFACTS / f'labels_{det}.png').relative_to(ROOT)})")

# %% Part 4. Baselines


def gt_boxes(label_path: Path, w: int, h: int) -> np.ndarray:
    rows = [list(map(float, l.split()[1:5])) for l in label_path.read_text().splitlines() if l.strip()] if label_path.is_file() else []
    out = [[(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h] for cx, cy, bw, bh in rows]
    return np.array(out).reshape(-1, 4)


def iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x0 = np.maximum(box[0], boxes[:, 0]); y0 = np.maximum(box[1], boxes[:, 1])
    x1 = np.minimum(box[2], boxes[:, 2]); y1 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    return inter / (area(box) + area(boxes) - inter + 1e-9)


def ap50(predictions: dict, truths: dict) -> dict:
    """All-point AP at IoU 0.5, one class. predictions: image -> [(box, score)], truths: image -> boxes."""
    flat = sorted(((s, img, b) for img, preds in predictions.items() for b, s in preds), key=lambda t: -t[0])
    n_true = sum(len(t) for t in truths.values())
    used = {img: np.zeros(len(t), bool) for img, t in truths.items()}
    tp = np.zeros(len(flat))
    for k, (_, img, box) in enumerate(flat):
        t = truths.get(img, np.zeros((0, 4)))
        if len(t):
            overlaps = iou(np.asarray(box), t)
            j = int(np.argmax(overlaps))
            if overlaps[j] >= 0.5 and not used[img][j]:
                used[img][j] = True
                tp[k] = 1
    if not n_true:
        return dict(ap50=float("nan"), precision=float("nan"), recall=float("nan"))
    cum_tp = np.cumsum(tp)
    recall = cum_tp / n_true
    precision = cum_tp / np.arange(1, len(flat) + 1) if len(flat) else np.array([])
    envelope = np.maximum.accumulate(precision[::-1])[::-1] if len(flat) else np.array([])
    ap = float(np.sum(np.diff(np.concatenate([[0], recall])) * envelope)) if len(flat) else 0.0
    return dict(ap50=ap, precision=float(precision[-1]) if len(flat) else 0.0, recall=float(recall[-1]) if len(flat) else 0.0)


def scores(predictions: dict, truths: dict) -> dict:
    """AP50 over every box (to compare models), and precision and recall at the confidence the reader
    actually uses (to say what the reader will do). As in Session 9: mAP picks a model, the threshold
    table configures it."""
    at_conf = {img: [(b, s) for b, s in preds if s >= DETECT_CONF] for img, preds in predictions.items()}
    kept = ap50(at_conf, truths)
    return dict(ap50=ap50(predictions, truths)["ap50"], precision=kept["precision"], recall=kept["recall"])


def cpu_ms_per_image(model, det: str, n: int = 20) -> float:
    """Inference time on the CPU, which is what the Space has (the training device says nothing about it).
    The reader runs question_no once per page and text_line on 4 tiles per page."""
    paths = [str(p) for p in split_images(det, "val")[:n]]
    model.predict(paths[0], imgsz=IMGSZ, device="cpu", verbose=False)  # warm-up
    started = time.perf_counter()
    for p in paths:
        model.predict(p, imgsz=IMGSZ, device="cpu", verbose=False)
    return (time.perf_counter() - started) * 1000 / len(paths)


def split_images(det: str, split: str = "val") -> list[Path]:
    return sorted((BUILT / det / split / "images").iterdir())


def truths_for(det: str, split: str = "val") -> dict:
    from PIL import Image
    out = {}
    for path in split_images(det, split):
        w, h = Image.open(path).size
        out[path.name] = gt_boxes(BUILT / det / split / "labels" / f"{path.stem}.txt", w, h)
    return out


def predict_boxes(model, det: str, split: str) -> dict:
    preds = {}
    for path in split_images(det, split):
        r = model.predict(str(path), conf=0.001, imgsz=IMGSZ, device=DEVICE, verbose=False)[0]
        preds[path.name] = [(b, float(s)) for b, s in zip(r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    return preds


def rule_boxes(det: str, split: str) -> dict:
    rule = rule_question_numbers if det == "question_no" else rule_lines
    return {p.name: [(b, 1.0) for b in rule(cv2.imread(str(p), cv2.IMREAD_GRAYSCALE))] for p in split_images(det, split)}


section("Part 4. Baselines on the validation split")
# The rules live in vision/rules.py because the reader uses them too, as its no-learning baseline.
import cv2  # noqa: E402
from ultralytics import YOLO  # noqa: E402

from vision.reader import DETECT_CONF  # noqa: E402
from vision.rules import rule_lines, rule_question_numbers  # noqa: E402

baselines = []
coco = YOLO("yolo11n.pt")
for det in DETECTORS:
    truths = truths_for(det)
    for name, preds in (("rules (no learning)", rule_boxes(det, "val")),
                        ("COCO YOLO11n zero-shot, any box", predict_boxes(coco, det, "val"))):
        baselines.append(dict(detector=det, model=name, **scores(preds, truths)))
baselines = pd.DataFrame(baselines)
print(f"AP50 over all boxes; precision and recall at confidence {DETECT_CONF}, the reader's threshold "
      "(the rules have no score, so all their boxes count):")
print(baselines.round(3).to_string(index=False))
print("COCO has no class for question numbers or handwritten lines, so zero-shot can only guess with unrelated boxes.")
baselines.to_csv(RESULTS / "baselines.csv", index=False)

# %% Part 5. Finetuning grid
section("Part 5. Finetuning: a small planned grid, chosen on validation only")
# What we vary, and why (everything else stays at the Ultralytics defaults):
#   model      yolo11n vs yolo11s: is the bigger model worth its extra latency on a CPU Space?
#   aug        'default' flips images left to right, which never happens to writing; 'scan' turns that off
#              and adds the small rotations, brightness changes and perspective of a phone scan
#   freeze     question_no only, around the best of the above: 10 frozen backbone layers may generalise
#              better with about 140 training pages
#   lr0        question_no only, around the best so far, 2 extra runs: half and double the base rate
# That is 7 runs for question_no and 4 for text_line, sized to finish overnight on a laptop.
# Not varied: image size. Both exports were resized to 640 x 640 by their authors, so a larger imgsz only
# upsamples the same pixels.
#
# The optimizer is set explicitly. With Ultralytics' default `optimizer="auto"`, lr0 is IGNORED: for a short
# run like ours (under 10,000 iterations) it picks AdamW at lr 0.002 and momentum 0.9 by itself, so an lr0
# grid would train the same model three times. We fix that same choice (AdamW, 0.002, 0.9) as the base,
# which keeps the base runs identical to the default, and the lr0 runs then really move the learning rate.
OPTIMIZER, BASE_LR0, MOMENTUM = "AdamW", 0.002, 0.9
LR0_SWEEP = (0.001, 0.004)
AUGS = {
    "default": dict(),
    "scan": dict(fliplr=0.0, degrees=5.0, hsv_v=0.5, translate=0.05, scale=0.3, perspective=0.0005),
}


def grid_for(det: str) -> list[dict]:
    base = [dict(model=m, aug=a, freeze=0, lr0=BASE_LR0) for m in ("yolo11n", "yolo11s") for a in AUGS]
    return base[:1] if SMOKE else base


def run_name(det: str, cfg: dict) -> str:
    return f"{det}_{cfg['model']}_{cfg['aug']}_f{cfg['freeze']}_lr{cfg['lr0']}"


def train_one(det: str, cfg: dict) -> dict:
    name = run_name(det, cfg)
    best = RUNS / name / "weights" / "best.pt"
    started = time.time()
    if not (REUSE and best.is_file()):
        # deterministic=False as in the Session 9 notebook: MPS has no deterministic backward for some ops,
        # and seed=0 still fixes the shuffling and the augmentation.
        YOLO(f"{cfg['model']}.pt").train(
            data=str(yamls[det]), epochs=EPOCHS, patience=PATIENCE, imgsz=IMGSZ, batch=16, seed=SEED,
            deterministic=False, freeze=cfg["freeze"] or None, optimizer=OPTIMIZER, lr0=cfg["lr0"],
            momentum=MOMENTUM, project=str(RUNS), name=name, exist_ok=True, plots=False, verbose=False,
            workers=0 if SMOKE else 4, fraction=0.15 if SMOKE else 1.0, device=DEVICE, **AUGS[cfg["aug"]],
        )
    minutes = (time.time() - started) / 60
    model = YOLO(str(best))
    metrics = model.val(data=str(yamls[det]), split="val", imgsz=IMGSZ, device=DEVICE, verbose=False, plots=False)
    ours = scores(predict_boxes(model, det, "val"), truths_for(det))
    epochs_run = len(pd.read_csv(RUNS / name / "results.csv")) if (RUNS / name / "results.csv").is_file() else None
    return dict(detector=det, run=name, **cfg, mAP50_95=float(metrics.box.map), mAP50=float(metrics.box.map50),
                **ours, cpu_ms_per_image=round(cpu_ms_per_image(model, det), 1), epochs_run=epochs_run,
                train_minutes=round(minutes, 1))


results = []
for det in DETECTORS:
    for cfg in grid_for(det):
        print(f"\n--- {run_name(det, cfg)}")
        results.append(train_one(det, cfg))
    if det == "question_no" and not SMOKE:  # freezing, then learning rate, around the best so far
        best = max((r for r in results if r["detector"] == det), key=lambda r: r["mAP50_95"])
        cfg = dict(model=best["model"], aug=best["aug"], freeze=10, lr0=BASE_LR0)
        print(f"\n--- {run_name(det, cfg)}")
        results.append(train_one(det, cfg))
        best = max((r for r in results if r["detector"] == det), key=lambda r: r["mAP50_95"])
        for lr in LR0_SWEEP:
            cfg = dict(model=best["model"], aug=best["aug"], freeze=best["freeze"], lr0=lr)
            print(f"\n--- {run_name(det, cfg)}")
            results.append(train_one(det, cfg))
grid = pd.DataFrame(results).sort_values(["detector", "mAP50_95"], ascending=[True, False])
print("\n" + grid.drop(columns=["run"]).round(3).to_string(index=False))
grid.to_csv(RESULTS / "grid_results.csv", index=False)

# %% Part 6. Save the chosen models
section("Part 6. The chosen models (best mAP50-95 on validation)")
chosen = {}
for det, part in grid.groupby("detector"):
    top = part.iloc[0]
    target = WEIGHTS / f"{det}_detector.pt"
    shutil.copy2(RUNS / top["run"] / "weights" / "best.pt", target)
    rules_ap = baselines[(baselines["detector"] == det) & baselines["model"].str.startswith("rules")]["ap50"].iloc[0]
    chosen[det] = dict(run=top["run"], weights=str(target.relative_to(ROOT)) if not SMOKE else str(target), mAP50_95=float(top["mAP50_95"]),
                       mAP50=float(top["mAP50"]), ap50_minus_rules=float(top["ap50"] - rules_ap),
                       reader_conf=DETECT_CONF, cpu_ms_per_image=float(top["cpu_ms_per_image"]))
    print(f"{det:12s} {top['run']}  mAP50-95 {top['mAP50_95']:.3f}  mAP50 {top['mAP50']:.3f}  -> {target}")
(RESULTS / "chosen.json").write_text(json.dumps(dict(smoke=SMOKE, chosen=chosen), indent=2))

# The confidence threshold is a dial (Session 9, 6.3): the chosen models on validation at several cuts, so the
# reader's threshold can be revisited on validation, never on the test split or the Mendeley exams.
# For question numbers a miss is the worse error (its answer is merged into the previous question).
sweep = []
for det in DETECTORS:
    preds, truths = predict_boxes(YOLO(str(WEIGHTS / f"{det}_detector.pt")), det, "val"), truths_for(det)
    for cut in (0.05, 0.1, 0.15, 0.25, 0.4, 0.5, 0.6, 0.75):
        kept = ap50({img: [(b, s) for b, s in p if s >= cut] for img, p in preds.items()}, truths)
        sweep.append(dict(detector=det, conf=cut, precision=kept["precision"], recall=kept["recall"]))
sweep = pd.DataFrame(sweep)
print(f"\nPrecision and recall of the chosen models on validation, by confidence threshold (the reader uses {DETECT_CONF}):")
print(sweep.pivot(index="conf", columns="detector", values=["precision", "recall"]).round(3).to_string())
sweep.to_csv(RESULTS / "conf_sweep_val.csv", index=False)

# %% Part 7. Test, once
section("Part 7. The chosen models and the baselines on the test split (used once, nothing is chosen here)")
rows = []
for det in DETECTORS:
    truths = truths_for(det, "test")
    model = YOLO(str(WEIGHTS / f"{det}_detector.pt"))
    metrics = model.val(data=str(yamls[det]), split="test", imgsz=IMGSZ, device=DEVICE, verbose=False, plots=False)
    for name, preds in (("finetuned (chosen on val)", predict_boxes(model, det, "test")),
                        ("rules (no learning)", rule_boxes(det, "test")),
                        ("COCO YOLO11n zero-shot, any box", predict_boxes(coco, det, "test"))):
        rows.append(dict(detector=det, model=name, images=len(truths), **scores(preds, truths)))
    rows[-3].update(mAP50_95=float(metrics.box.map), mAP50=float(metrics.box.map50))
test = pd.DataFrame(rows)
print(f"AP50 over all boxes; precision and recall at confidence {DETECT_CONF}, the reader's threshold:")
print(test.round(3).to_string(index=False))
print("A good test number says the detector finds these labels on these booklets and notebooks. Whether it works on")
print("real exams is the product evaluation on the Mendeley exams (vision/evaluate_reader.py).")
test.to_csv(RESULTS / "test_results.csv", index=False)
