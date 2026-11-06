"""
Step 04: evaluate a trained model on the test set (images it has never seen),
and draw its predictions next to the ground truth for every image.

What it does:
  1. Downloads the dataset from the hub (same repo / version as for training).
  2. Runs Ultralytics validation on SPLIT: precision, recall and mAP, per class,
     for boxes and masks, plus the confusion matrix.
  3. For each image, saves a comparison picture: ground truth (top) and
     prediction (bottom), with the class and confidence of each detection.

Output (in eval/<run name>_<SPLIT>/):
  metrics.csv            per-class metrics
  comparisons/           one picture per image (ground truth vs prediction)
  confusion_matrix*.png  and the other Ultralytics validation plots

The run's training settings (image size) are read from its args.yaml.
"""

import csv
import glob
import os

import cv2
import numpy as np
import yaml
from huggingface_hub import snapshot_download
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

RUN_DIR = os.path.join(HERE, "model", "permabot_weed_seg_yolo",
                       "permabot_onions_weed_seg_first_test_v2_yolo11s-seg_1280_20261106_1559")

HF_DATASET_REPO = "Greynar/permabot_onions_weed_seg_first_test"
DATASET_REVISION = "v2"

SPLIT = "test"          # "test" for the final score, "val" to compare with the training logs
CONF = 0.25             # minimum confidence for a prediction to be drawn on the comparison pictures
DEVICE = None           # None = GPU if available, else CPU
PREVIEW_ALPHA = 0.4
COMPARISON_WIDTH = 1280  # width of each half of the comparison pictures

# ----------------------------------------------------------------------------

COLORS = {  # BGR, same as in 02_prepare_and_push_dataset.py
    "weed": (0, 0, 255), "onion": (255, 0, 255), "leek": (255, 255, 0),
    "carrot": (0, 165, 255), "bean": (0, 255, 0), "pea": (255, 0, 0),
}


def download_dataset():
    local_dir = os.path.join(HERE, "dataset", "hub", HF_DATASET_REPO.split("/")[1], DATASET_REVISION)
    snapshot_download(HF_DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION, local_dir=local_dir)
    return local_dir


def draw_polygons(image, polygons, title):
    """polygons: list of (class name, list of Nx2 pixel contours, confidence or None)."""
    overlay = image.copy()
    for name, contours, _ in polygons:
        cv2.fillPoly(overlay, contours, COLORS[name])
    out = cv2.addWeighted(overlay, PREVIEW_ALPHA, image, 1 - PREVIEW_ALPHA, 0)
    for name, contours, conf in polygons:
        cv2.polylines(out, contours, True, COLORS[name], 2)
        pts = np.vstack(contours)
        x, y = pts[np.argmin(pts[:, 1])]
        text = name if conf is None else f"{name} {conf:.2f}"
        cv2.putText(out, text, (int(x), max(20, int(y) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, COLORS[name], 2)
    counts = ", ".join(f"{n}: {sum(p[0] == n for p in polygons)}" for n in sorted({p[0] for p in polygons}))
    cv2.rectangle(out, (0, 0), (out.shape[1], 50), (0, 0, 0), -1)
    cv2.putText(out, f"{title}   ({counts or 'nothing'})", (10, 35), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                (255, 255, 255), 2)
    return out


def ground_truth_polygons(label_path, names, w, h):
    polygons = []
    if os.path.exists(label_path):
        for line in open(label_path):
            parts = line.split()
            if parts:
                pts = (np.array(parts[1:], dtype=np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
                polygons.append((names[int(parts[0])], [pts], None))
    return polygons


def predicted_polygons(result, names):
    """Contours of the predicted pixel masks (needs retina_masks=True, so masks have the image size).
    A mask can have several separate pieces (e.g. crossed leaves): each piece gets its own contour."""
    polygons = []
    if result.masks is not None:
        masks = result.masks.data.cpu().numpy().astype(np.uint8)
        for mask, cls, conf in zip(masks, result.boxes.cls.tolist(), result.boxes.conf.tolist()):
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contours = [c.reshape(-1, 2) for c in contours if len(c) >= 3]
            if contours:
                polygons.append((names[int(cls)], contours, conf))
    return polygons


def main():
    with open(os.path.join(RUN_DIR, "args.yaml")) as f:
        imgsz = yaml.safe_load(f)["imgsz"]
    run_name = os.path.basename(RUN_DIR.rstrip("/"))
    out_dir = os.path.join(HERE, "eval", f"{run_name}_{SPLIT}")
    os.makedirs(os.path.join(out_dir, "comparisons"), exist_ok=True)

    dataset_dir = download_dataset()
    model = YOLO(os.path.join(RUN_DIR, "weights", "best.pt"))
    names = model.names

    # 1. Metrics
    metrics = model.val(data=os.path.join(dataset_dir, "data.yaml"), split=SPLIT, imgsz=imgsz, device=DEVICE,
                        project=os.path.dirname(out_dir), name=os.path.basename(out_dir), exist_ok=True)
    rows = []
    for i, class_id in enumerate(metrics.seg.ap_class_index):
        bp, br, b50, b5095 = metrics.box.class_result(i)
        mp, mr, m50, m5095 = metrics.seg.class_result(i)
        rows.append([names[int(class_id)], bp, br, b50, b5095, mp, mr, m50, m5095])
    header = ["class", "box_P", "box_R", "box_mAP50", "box_mAP50-95", "mask_P", "mask_R", "mask_mAP50",
              "mask_mAP50-95"]
    with open(os.path.join(out_dir, "metrics.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows([[r[0]] + [f"{v:.3f}" for v in r[1:]] for r in rows])

    # 2. Comparison pictures
    image_paths = sorted(glob.glob(os.path.join(dataset_dir, "images", SPLIT, "*.jpg")))
    for image_path in image_paths:
        image = cv2.imread(image_path)
        h, w = image.shape[:2]
        label_path = image_path.replace(f"images{os.sep}{SPLIT}", f"labels{os.sep}{SPLIT}")[:-4] + ".txt"
        result = model.predict(image, imgsz=imgsz, conf=CONF, device=DEVICE, retina_masks=True, verbose=False)[0]

        gt = draw_polygons(image, ground_truth_polygons(label_path, names, w, h), "Ground truth")
        pred = draw_polygons(image, predicted_polygons(result, names), f"Prediction (conf >= {CONF})")
        size = (COMPARISON_WIDTH, round(COMPARISON_WIDTH * h / w))
        comparison = np.vstack([cv2.resize(gt, size), cv2.resize(pred, size)])
        cv2.imwrite(os.path.join(out_dir, "comparisons", os.path.basename(image_path)), comparison)

    print(f"\nPer-class metrics on '{SPLIT}' ({len(image_paths)} images):")
    print("  " + "  ".join(f"{h:>13s}" for h in header))
    for r in rows:
        print("  " + f"{r[0]:>13s}  " + "  ".join(f"{v:13.3f}" for v in r[1:]))
    print(f"\nResults in {out_dir}")
    print(f"Comparison pictures in {os.path.join(out_dir, 'comparisons')}")


if __name__ == "__main__":
    main()
