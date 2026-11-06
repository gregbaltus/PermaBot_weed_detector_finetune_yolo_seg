"""
Step 02: turn a Roboflow export into a clean YOLO-seg dataset, and optionally
push it to the Hugging Face hub.

What it does:
  1. Reads the Roboflow zip (format "YOLOv11", Instance Segmentation project).
  2. Checks that every label is a polygon (not a box).
  3. Remaps Roboflow's class IDs (alphabetical) to our fixed CLASSES list, by name.
  4. Puts every image back in OUR split, read from the manifest.csv files written
     by step 01 (Roboflow's own train/valid/test split is ignored).
  5. Draws the polygons on each image in previews/, to check the labels by eye.
  6. Writes data.yaml (for Ultralytics) and a README.md (dataset card).
  7. If PUSH_TO_HUB is True, uploads everything except previews/ to the hub.

Output (in OUTPUT_DIR):
  images/{train,val,test}/   labels/{train,val,test}/   data.yaml   README.md
  previews/{train,val,test}/  (local only)

Run with PUSH_TO_HUB = False first, look at the previews, then set it to True.
You must be logged in to the hub once on this machine: `hf auth login`.
"""

import csv
import glob
import os
import shutil
import zipfile
from collections import Counter

import cv2
import numpy as np
import yaml

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
EXPORT_ZIP = os.path.join(HERE, "dataset", "roboflow_exports", "Onion and Weed Segmentation.v2i.yolov11.zip")
FRAMES_DIR = os.path.join(HERE, "dataset", "frames")  # contains <video tag>/manifest.csv from step 01

DATASET_NAME = "permabot_onions_weed_seg_first_test"
OUTPUT_DIR = os.path.join(HERE, "dataset", "prepared", DATASET_NAME)

# Fixed class list: the index is the class ID. Never reorder, only append.
CLASSES = ["weed", "onion", "leek", "carrot", "bean", "pea"]

PUSH_TO_HUB = True
HF_REPO_ID = f"Greynar/{DATASET_NAME}"
PRIVATE = True
HF_TAG = "v2"           # version tag on the hub (e.g. v1, v2...); None = no tag

PREVIEW_ALPHA = 0.4     # transparency of the filled polygons in the previews

# ----------------------------------------------------------------------------

SPLITS = ["train", "val", "test"]
COLORS = {  # BGR
    "weed": (0, 0, 255), "onion": (255, 0, 255), "leek": (255, 255, 0),
    "carrot": (0, 165, 255), "bean": (0, 255, 0), "pea": (255, 0, 0),
}


def load_manifests(frames_dir):
    """image stem -> our split, from every manifest.csv written by step 01."""
    split_of = {}
    for path in glob.glob(os.path.join(frames_dir, "*", "manifest.csv")):
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                split_of[os.path.splitext(row["filename"])[0]] = row["split"]
    if not split_of:
        raise SystemExit(f"No manifest.csv found in {frames_dir}, run step 01 first.")
    return split_of


def original_stem(roboflow_filename):
    """'X_f001895_jpg.rf.4ecb....jpg' -> 'X_f001895' (Roboflow appends _jpg.rf.<hash>)."""
    base = os.path.basename(roboflow_filename)
    if "_jpg.rf." not in base:
        raise SystemExit(f"Unexpected Roboflow file name: {base}")
    return base.split("_jpg.rf.")[0]


def read_roboflow_classes(zf):
    yaml_names = [n for n in zf.namelist() if os.path.basename(n) == "data.yaml"]
    if not yaml_names:
        raise SystemExit("No data.yaml in the Roboflow zip.")
    names = yaml.safe_load(zf.read(yaml_names[0]))["names"]
    if isinstance(names, dict):
        names = [names[i] for i in sorted(names)]
    unknown = [n for n in names if n not in CLASSES]
    if unknown:
        raise SystemExit(f"Roboflow classes {unknown} are not in CLASSES {CLASSES}.")
    return names


def convert_labels(text, roboflow_names, label_name):
    """Remap class IDs and check that every line is a polygon. Returns (lines, class names)."""
    lines, names = [], []
    for line_no, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if not parts:
            continue
        coords = parts[1:]
        if len(coords) == 4:
            raise SystemExit(f"{label_name}:{line_no} is a box, not a polygon. "
                             "Is the Roboflow project an Instance Segmentation project?")
        if len(coords) < 6 or len(coords) % 2:
            raise SystemExit(f"{label_name}:{line_no} is not a valid polygon ({len(coords)} coordinates).")
        name = roboflow_names[int(parts[0])]
        lines.append(" ".join([str(CLASSES.index(name))] + coords))
        names.append(name)
    return lines, names


def draw_preview(image, label_lines):
    h, w = image.shape[:2]
    overlay = image.copy()
    polygons = []
    for line in label_lines:
        parts = line.split()
        name = CLASSES[int(parts[0])]
        pts = (np.array(parts[1:], dtype=np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
        polygons.append((name, pts))
        cv2.fillPoly(overlay, [pts], COLORS[name])
    preview = cv2.addWeighted(overlay, PREVIEW_ALPHA, image, 1 - PREVIEW_ALPHA, 0)
    for name, pts in polygons:
        cv2.polylines(preview, [pts], True, COLORS[name], 2)
        x, y = pts[np.argmin(pts[:, 1])]
        cv2.putText(preview, name, (int(x), max(15, int(y) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLORS[name], 2)
    return preview


def write_data_yaml(output_dir):
    # No "path" key: Ultralytics then resolves the splits relative to this file.
    data = {"train": "images/train", "val": "images/val", "test": "images/test",
            "names": dict(enumerate(CLASSES))}
    with open(os.path.join(output_dir, "data.yaml"), "w") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def write_dataset_card(output_dir, counts, instances):
    rows = "\n".join(f"| {s} | {counts[s]} | " + " | ".join(str(instances[s][c]) for c in CLASSES) + " |"
                     for s in SPLITS)
    card = f"""---
license: other
task_categories:
- image-segmentation
tags:
- yolo
- agriculture
- weed-detection
---

# {DATASET_NAME}

Instance segmentation of crops and weeds in top-down videos of market-garden rows
(PermaBot project). YOLO-seg format (Ultralytics): one polygon per plant.

Source export: `{os.path.basename(EXPORT_ZIP)}`

## Classes

{", ".join(f"{i} {c}" for i, c in enumerate(CLASSES))}

Dead or dry plant material is not labelled (background).

## Split

Images are split by position along the crop row (blocks of the row go entirely to
train, val or test), not at random, so the same plant never appears in two splits.

| split | images | {" | ".join(CLASSES)} |
|---|---|{"---|" * len(CLASSES)}
{rows}

## Use

```python
from huggingface_hub import snapshot_download
path = snapshot_download("{HF_REPO_ID}", repo_type="dataset")
# then train with data=f"{{path}}/data.yaml"
```
"""
    with open(os.path.join(output_dir, "README.md"), "w") as f:
        f.write(card)


def push_to_hub(output_dir):
    from huggingface_hub import HfApi

    api = HfApi()
    api.create_repo(HF_REPO_ID, repo_type="dataset", private=PRIVATE, exist_ok=True)
    api.upload_folder(folder_path=output_dir, repo_id=HF_REPO_ID, repo_type="dataset",
                      ignore_patterns=["previews/*"],
                      commit_message=f"Dataset from {os.path.basename(EXPORT_ZIP)}")
    if HF_TAG:
        api.create_tag(HF_REPO_ID, tag=HF_TAG, repo_type="dataset", exist_ok=True)
    print(f"Pushed to https://huggingface.co/datasets/{HF_REPO_ID}" + (f" (tag {HF_TAG})" if HF_TAG else ""))


def main():
    split_of = load_manifests(FRAMES_DIR)

    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)  # always regenerated from the zip
    for sub in ["images", "labels", "previews"]:
        for split in SPLITS:
            os.makedirs(os.path.join(OUTPUT_DIR, sub, split))

    counts = Counter()
    instances = {split: Counter() for split in SPLITS}
    moved, skipped = 0, []

    with zipfile.ZipFile(EXPORT_ZIP) as zf:
        roboflow_names = read_roboflow_classes(zf)
        print(f"Roboflow classes {roboflow_names} -> IDs {[CLASSES.index(n) for n in roboflow_names]}")

        image_names = sorted({n for n in zf.namelist() if "/images/" in n and n.lower().endswith(".jpg")})
        for image_name in image_names:
            stem = original_stem(image_name)
            split = split_of.get(stem)
            if split not in SPLITS:
                skipped.append(f"{stem} ({'not in any manifest' if split is None else split})")
                continue
            roboflow_split = image_name.split("/")[0]
            moved += roboflow_split.replace("valid", "val") != split

            label_name = image_name.replace("/images/", "/labels/").rsplit(".", 1)[0] + ".txt"
            text = zf.read(label_name).decode() if label_name in zf.namelist() else ""
            label_lines, names = convert_labels(text, roboflow_names, label_name)

            image_bytes = zf.read(image_name)
            with open(os.path.join(OUTPUT_DIR, "images", split, stem + ".jpg"), "wb") as f:
                f.write(image_bytes)
            with open(os.path.join(OUTPUT_DIR, "labels", split, stem + ".txt"), "w") as f:
                f.write("\n".join(label_lines) + ("\n" if label_lines else ""))

            image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
            cv2.imwrite(os.path.join(OUTPUT_DIR, "previews", split, stem + ".jpg"), draw_preview(image, label_lines))

            counts[split] += 1
            instances[split].update(names)

    write_data_yaml(OUTPUT_DIR)
    write_dataset_card(OUTPUT_DIR, counts, instances)

    for split in SPLITS:
        per_class = ", ".join(f"{c}: {instances[split][c]}" for c in CLASSES if instances[split][c])
        print(f"  {split:5s}: {counts[split]:3d} images  ({per_class or 'no instances'})")
    print(f"{moved} image(s) were in a different split in Roboflow (our split from manifest.csv was used).")
    if skipped:
        print(f"WARNING, {len(skipped)} image(s) skipped: " + ", ".join(skipped))
    print(f"Dataset written to {OUTPUT_DIR}")
    print(f"Check the label drawings in {os.path.join(OUTPUT_DIR, 'previews')}")

    if PUSH_TO_HUB:
        push_to_hub(OUTPUT_DIR)
    else:
        print("PUSH_TO_HUB is False: nothing uploaded.")


if __name__ == "__main__":
    main()
