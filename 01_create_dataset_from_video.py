"""
Step 01: extract still images from a field video and split them into
train / val / test, ready to be uploaded to Roboflow for labelling.

What it does:
  1. Reads the video and drops duplicated frames (the camera really records at
     ~5 fps, the container repeats each frame ~6 times).
  2. Estimates how far the camera has moved between distinct frames (ORB
     feature matching + similarity transform; the soil is roughly planar).
  3. Saves a new image each time the camera has moved FRAME_STEP image widths,
     so images do not overlap much and cart stops do not produce copies.
  4. Splits images by POSITION ALONG THE ROW, not at random: the row is cut into
     N_BLOCKS blocks, each block goes entirely to train, val or test, and images
     too close to a boundary between two different splits are discarded.
     Positions are normalised to 0..1 from the start of the row (reversed for
     "retour" videos), and the block -> split assignment only depends on
     N_BLOCKS and SEED. So a 50cm and an 80cm video of the same row, or the
     "allé" and "retour" passes, put the same plants in the same split.

Output (in OUTPUT_DIR/<video tag>/):
  train/ val/ test/   the images
  manifest.csv        one row per extracted image (including discarded ones)

Edit the variables below, then run:
    python 01_create_dataset_from_video.py
"""

import csv
import os
import re
import shutil
import unicodedata

import cv2
import numpy as np

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
VIDEO_DIR = "/home/gregory/Documents/Recherche/PermaBot/videoMaraicherBackup/Obs_video"
VIDEO_NAME = "OignonRouge80cmHauteurRange110x60_retourmkv.mkv"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset", "frames")

FRAME_STEP = 0.5        # save one image every FRAME_STEP image widths of camera travel
MAX_IMAGES = None       # keep at most this many images (evenly spread along the row); None = all

DIRECTION = "auto"      # "alle", "retour" or "auto" (guessed from the file name)
N_BLOCKS = 10           # the row is cut into this many blocks for the split
SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
BUFFER = 0.5            # discard images closer than BUFFER image widths to a boundary between splits
SEED = 0                # same seed + same N_BLOCKS => same block assignment for every video

JPEG_QUALITY = 95
DUPLICATE_THRESHOLD = 1.0  # mean abs pixel difference (thumbnail) below which a frame is a duplicate
MOTION_WIDTH = 640         # frames are downscaled to this width for motion estimation

# ----------------------------------------------------------------------------


def video_tag(video_name):
    """Short ASCII name for a video, e.g. 'OignonRouge80cm_retour'."""
    stem = os.path.splitext(video_name)[0]
    stem = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode()
    match = re.match(r"^(.*?\d+cm).*_(alle|retour)", stem, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)}_{match.group(2).lower()}"
    return re.sub(r"[^A-Za-z0-9_-]", "_", stem)


def guess_direction(video_name):
    tag = video_tag(video_name).lower()
    if tag.endswith("_retour"):
        return "retour"
    if tag.endswith("_alle"):
        return "alle"
    raise SystemExit(f"Cannot guess direction from '{video_name}', set DIRECTION by hand.")


def assign_blocks_to_splits(n_blocks, ratios, seed):
    """Deterministic list split_of_block[i], with at least one val and one test block."""
    n_val = max(1, round(ratios["val"] * n_blocks))
    n_test = max(1, round(ratios["test"] * n_blocks))
    n_train = n_blocks - n_val - n_test
    if n_train < 1:
        raise SystemExit(f"N_BLOCKS={n_blocks} is too small for the requested split ratios.")
    splits = ["train"] * n_train + ["val"] * n_val + ["test"] * n_test
    rng = np.random.default_rng(seed)
    rng.shuffle(splits)
    return splits


class MotionEstimator:
    """Camera displacement between consecutive distinct frames, in full-resolution pixels."""

    def __init__(self, frame_width):
        self.scale = frame_width / MOTION_WIDTH
        self.orb = cv2.ORB_create(2000)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        self.prev = None
        self.last_shift = np.zeros(2)
        self.failures = 0

    def update(self, gray_small):
        keypoints, descriptors = self.orb.detectAndCompute(gray_small, None)
        shift = None
        if self.prev is not None and descriptors is not None and self.prev[1] is not None:
            matches = self.matcher.match(self.prev[1], descriptors)
            if len(matches) >= 20:
                pts_prev = np.float32([self.prev[0][m.queryIdx].pt for m in matches])
                pts_new = np.float32([keypoints[m.trainIdx].pt for m in matches])
                transform, _ = cv2.estimateAffinePartial2D(pts_prev, pts_new, method=cv2.RANSAC,
                                                           ransacReprojThreshold=3)
                if transform is not None:
                    # The image content moves by +t, so the camera moved by -t.
                    shift = -transform[:, 2] * self.scale
        if self.prev is not None and shift is None:
            self.failures += 1
            shift = self.last_shift  # assume constant speed over a failed frame
        self.prev = (keypoints, descriptors)
        if shift is None:
            return np.zeros(2)
        self.last_shift = shift
        return shift


def extract_frames(video_path, staging_dir, tag):
    """Pass over the video, save candidate images to staging_dir, return their metadata."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    small_size = (MOTION_WIDTH, round(MOTION_WIDTH * height / width))

    motion = MotionEstimator(width)
    trajectory = []  # camera position (x, y) for every distinct frame
    saved = []
    prev_thumb = None
    position = np.zeros(2)
    last_saved_position = None
    n_distinct = 0

    frame_idx = -1
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1

        gray_small = cv2.cvtColor(cv2.resize(frame, small_size), cv2.COLOR_BGR2GRAY)
        thumb = cv2.resize(gray_small, (160, 90)).astype(np.int16)
        if prev_thumb is not None and np.abs(thumb - prev_thumb).mean() < DUPLICATE_THRESHOLD:
            continue
        prev_thumb = thumb
        n_distinct += 1

        position = position + motion.update(gray_small)
        trajectory.append(position.copy())

        if last_saved_position is None or np.linalg.norm(position - last_saved_position) >= FRAME_STEP * width:
            filename = f"{tag}_f{frame_idx:06d}.jpg"
            cv2.imwrite(os.path.join(staging_dir, filename), frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
            saved.append({"filename": filename, "frame_idx": frame_idx,
                          "time_s": round(frame_idx / fps, 2), "xy": position.copy()})
            last_saved_position = position.copy()

        if (frame_idx + 1) % 600 == 0:
            print(f"  frame {frame_idx + 1}/{total_frames}, {len(saved)} images saved")

    cap.release()
    print(f"Read {frame_idx + 1} frames, {n_distinct} distinct, {len(saved)} images saved, "
          f"{motion.failures} motion-estimation failures.")
    return saved, np.array(trajectory), width


def main():
    video_path = os.path.join(VIDEO_DIR, VIDEO_NAME)
    tag = video_tag(VIDEO_NAME)
    direction = guess_direction(VIDEO_NAME) if DIRECTION == "auto" else DIRECTION
    out_dir = os.path.join(OUTPUT_DIR, tag)
    if os.path.exists(out_dir):
        raise SystemExit(f"{out_dir} already exists, delete it first to regenerate this video's images.")
    staging_dir = os.path.join(out_dir, "_staging")
    os.makedirs(staging_dir)
    print(f"Video: {VIDEO_NAME}  ->  tag '{tag}', direction '{direction}'")

    saved, trajectory, width = extract_frames(video_path, staging_dir, tag)

    # Position along the row: projection on the main direction of travel, normalised to 0..1.
    travel = trajectory[-1] - trajectory[0]
    axis = travel / np.linalg.norm(travel)
    row_coords = trajectory @ axis
    start, length = row_coords.min(), row_coords.max() - row_coords.min()
    print(f"Row length: {length / width:.1f} image widths.")
    for item in saved:
        p = (item["xy"] @ axis - start) / length
        item["position"] = 1.0 - p if direction == "retour" else p

    if MAX_IMAGES is not None and len(saved) > MAX_IMAGES:
        keep = set(np.linspace(0, len(saved) - 1, MAX_IMAGES).round().astype(int))
        for i, item in enumerate(saved):
            if i not in keep:
                os.remove(os.path.join(staging_dir, item["filename"]))
        saved = [item for i, item in enumerate(saved) if i in keep]

    # Split by block, with a buffer zone around boundaries between different splits.
    split_of_block = assign_blocks_to_splits(N_BLOCKS, SPLIT_RATIOS, SEED)
    buffer = BUFFER * width / length  # in normalised row units
    for item in saved:
        p = item["position"]
        block = min(int(p * N_BLOCKS), N_BLOCKS - 1)
        split = split_of_block[block]
        near_prev = block > 0 and split_of_block[block - 1] != split and p - block / N_BLOCKS < buffer
        near_next = (block < N_BLOCKS - 1 and split_of_block[block + 1] != split
                     and (block + 1) / N_BLOCKS - p < buffer)
        item["block"] = block
        item["split"] = "discarded" if near_prev or near_next else split

    for split in SPLIT_RATIOS:
        os.makedirs(os.path.join(out_dir, split))
    for item in saved:
        src = os.path.join(staging_dir, item["filename"])
        if item["split"] == "discarded":
            os.remove(src)
        else:
            shutil.move(src, os.path.join(out_dir, item["split"], item["filename"]))
    os.rmdir(staging_dir)

    with open(os.path.join(out_dir, "manifest.csv"), "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "video", "frame_idx", "time_s", "position", "block", "split"])
        for item in saved:
            writer.writerow([item["filename"], VIDEO_NAME, item["frame_idx"], item["time_s"],
                             f"{item['position']:.4f}", item["block"], item["split"]])

    print(f"Blocks -> splits: {split_of_block}")
    for split in list(SPLIT_RATIOS) + ["discarded"]:
        print(f"  {split:9s}: {sum(item['split'] == split for item in saved)} images")
    print(f"Done, output in {out_dir}")


if __name__ == "__main__":
    main()
