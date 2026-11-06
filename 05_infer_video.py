"""
Step 05: run a trained model on a video and write a new video with the predicted
masks drawn on every frame.

What it does:
  1. Reads the video from START_TIME for DURATION seconds.
  2. Runs the model only on distinct frames (the camera really records at
     ~5 fps, each frame is repeated ~6 times); repeated frames reuse the
     prediction. The output keeps the original frame rate and duration.
  3. Draws the masks (onion magenta, weed red) on each frame.
  4. If BANNER is True, adds a banner with the time, the number of plants per
     class, the inference time and the ZONE of the row: TRAIN / VAL / TEST /
     BUFFER, read from the manifest.csv of step 01. Only the TEST zones (and
     videos never used for the dataset: UNSEEN VIDEO) show honest performance.

Output: inference/<video tag>_<run name>_<start>-<end>s.mp4

The run's training image size is read from its args.yaml.
"""

import csv
import os
import re
import time
import unicodedata

import cv2
import numpy as np
import yaml
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

VIDEO_DIR = "/home/gregory/Documents/Recherche/PermaBot/videoMaraicherBackup/Obs_video"
VIDEO_NAME = "OignonRouge80cmHauteurRange110x60_retourmkv.mkv"
START_TIME = 25.0       # seconds
DURATION = 25.0         # seconds; None = until the end of the video

RUN_DIR = os.path.join(HERE, "model", "permabot_weed_seg_yolo",
                       "permabot_onions_weed_seg_first_test_v2_yolo11s-seg_1280_20261106_1559")

CONF = 0.25             # minimum confidence for a prediction to be drawn
SHOW_CONF = True        # write the confidence next to each mask
BANNER = True           # banner with time, counts, inference time and train/val/test zone
DEVICE = None           # None = GPU if available, else CPU

FRAMES_DIR = os.path.join(HERE, "dataset", "frames")  # manifest.csv files from step 01 (for the banner)
OUTPUT_DIR = os.path.join(HERE, "inference")

PREVIEW_ALPHA = 0.4
DUPLICATE_THRESHOLD = 1.0  # same value as in step 01

# ----------------------------------------------------------------------------

COLORS = {  # BGR, same as in 02 and 04
    "weed": (0, 0, 255), "onion": (255, 0, 255), "leek": (255, 255, 0),
    "carrot": (0, 165, 255), "bean": (0, 255, 0), "pea": (255, 0, 0),
}
ZONE_COLORS = {  # BGR
    "TRAIN": (0, 140, 255), "VAL": (0, 215, 255), "TEST": (0, 200, 0),
    "BUFFER": (160, 160, 160), "UNSEEN VIDEO": (0, 200, 0),
}


def video_tag(video_name):
    """Same naming as step 01, e.g. 'OignonRouge80cm_retour'."""
    stem = os.path.splitext(video_name)[0]
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    match = re.match(r"^(.*?\d+cm).*_(alle|retour)", stem, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)}_{match.group(2).lower()}"
    return re.sub(r"[^A-Za-z0-9_-]", "_", stem)


class ZoneLookup:
    """Zone of any frame = split of the closest image extracted by step 01."""

    def __init__(self, tag):
        path = os.path.join(FRAMES_DIR, tag, "manifest.csv")
        self.frames, self.zones = None, None
        if os.path.exists(path):
            with open(path, newline="") as f:
                rows = list(csv.DictReader(f))
            self.frames = np.array([int(r["frame_idx"]) for r in rows])
            self.zones = ["BUFFER" if r["split"] == "discarded" else r["split"].upper() for r in rows]

    def __call__(self, frame_idx):
        if self.frames is None:
            return "UNSEEN VIDEO"
        return self.zones[int(np.argmin(np.abs(self.frames - frame_idx)))]


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


def draw_predictions(frame, polygons):
    overlay = frame.copy()
    for name, contours, _ in polygons:
        cv2.fillPoly(overlay, contours, COLORS[name])
    out = cv2.addWeighted(overlay, PREVIEW_ALPHA, frame, 1 - PREVIEW_ALPHA, 0)
    for name, contours, conf in polygons:
        cv2.polylines(out, contours, True, COLORS[name], 2)
        if SHOW_CONF:
            pts = np.vstack(contours)
            x, y = pts[np.argmin(pts[:, 1])]
            cv2.putText(out, f"{name} {conf:.2f}", (int(x), max(80, int(y) - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        COLORS[name], 2)
    return out


def draw_banner(frame, time_s, polygons, inference_ms, zone):
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 60), (0, 0, 0), -1)
    counts = "  ".join(f"{n}: {sum(p[0] == n for p in polygons)}" for n in ["onion", "weed"])
    text = f"t = {time_s:6.1f} s    {counts}    inference {inference_ms:4.0f} ms"
    cv2.putText(frame, text, (15, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    label = f"{zone} ZONE" if zone in ("TRAIN", "VAL", "TEST", "BUFFER") else zone
    (w, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
    x = frame.shape[1] - w - 25
    cv2.rectangle(frame, (x - 12, 6), (frame.shape[1] - 10, 54), ZONE_COLORS[zone], -1)
    cv2.putText(frame, label, (x, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (0, 0, 0), 3)


def main():
    with open(os.path.join(RUN_DIR, "args.yaml")) as f:
        imgsz = yaml.safe_load(f)["imgsz"]
    run_name = os.path.basename(RUN_DIR.rstrip("/"))
    tag = video_tag(VIDEO_NAME)

    cap = cv2.VideoCapture(os.path.join(VIDEO_DIR, VIDEO_NAME))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {VIDEO_NAME}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    start_frame = int(START_TIME * fps)
    if start_frame >= total_frames:
        raise SystemExit(f"START_TIME {START_TIME}s is beyond the video length ({total_frames / fps:.1f}s).")
    end_frame = total_frames if DURATION is None else min(total_frames, start_frame + int(DURATION * fps))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    end_time = end_frame / fps

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, f"{tag}_{run_name}_{START_TIME:g}-{end_time:g}s.mp4")
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))

    model = YOLO(os.path.join(RUN_DIR, "weights", "best.pt"))
    zone_of = ZoneLookup(tag) if BANNER else None
    print(f"{VIDEO_NAME}: {START_TIME:g}s -> {end_time:g}s ({end_frame - start_frame} frames)")
    if BANNER and zone_of.frames is None:
        print("No manifest.csv for this video: the banner will show UNSEEN VIDEO.")

    prev_thumb, annotated, polygons, inference_ms = None, None, [], 0.0
    inference_times = []
    for frame_idx in range(start_frame, end_frame):
        ok, frame = cap.read()
        if not ok:
            print(f"Video ended early at frame {frame_idx}.")
            break

        thumb = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (160, 90)).astype(np.int16)
        is_new = prev_thumb is None or np.abs(thumb - prev_thumb).mean() >= DUPLICATE_THRESHOLD
        if is_new:
            prev_thumb = thumb
            t0 = time.perf_counter()
            result = model.predict(frame, imgsz=imgsz, conf=CONF, device=DEVICE, retina_masks=True,
                                   verbose=False)[0]
            inference_ms = (time.perf_counter() - t0) * 1000
            inference_times.append(inference_ms)
            polygons = predicted_polygons(result, model.names)
            annotated = draw_predictions(frame, polygons)
            if BANNER:
                draw_banner(annotated, frame_idx / fps, polygons, inference_ms, zone_of(frame_idx))

        writer.write(annotated)  # repeated frames reuse the last annotated image

        done = frame_idx - start_frame + 1
        if is_new and len(inference_times) % 20 == 0:
            print(f"  {done}/{end_frame - start_frame} frames, {len(inference_times)} distinct")

    cap.release()
    writer.release()
    print(f"\n{len(inference_times)} distinct frames, inference {np.mean(inference_times):.0f} ms per frame "
          f"on average (median {np.median(inference_times):.0f} ms).")
    print(f"Video written to {output_path}")


if __name__ == "__main__":
    main()
