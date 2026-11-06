"""
Step 03: fine-tune YOLO11-seg on the dataset from the Hugging Face hub, then push
the trained model to the hub. Meant to run on the GPU machine.

What it does:
  1. Downloads the dataset (HF_DATASET_REPO at version DATASET_REVISION).
  2. Fine-tunes MODEL (pretrained COCO weights) with Ultralytics.
  3. If PUSH_TO_HUB is True, uploads the run (best.pt, training curves, settings
     and a model card) to HF_MODEL_REPO, in a folder named after the run.
     Each run gets its own folder, so runs can be compared later.

Output: runs/<RUN_NAME>/ (Ultralytics' usual training folder, local only).

Before the first run on a machine:
    pip install -r requirements.txt
    hf auth login
"""

import os
from datetime import datetime

from huggingface_hub import HfApi, snapshot_download
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))

HF_DATASET_REPO = "Greynar/permabot_onions_weed_seg_first_test"
DATASET_REVISION = "v2"          # tag on the hub (same number as the Roboflow version)

MODEL = "yolo11s-seg.pt"         # n / s / m / l / x: bigger = more accurate but slower on the Jetson
IMGSZ = 1280                     # training image size; large because the weeds are small
EPOCHS = 100
PATIENCE = 30                    # stop if val metrics do not improve for this many epochs
BATCH = 8                        # lower it if the GPU runs out of memory (-1 = automatic)
DEVICE = 0                       # GPU index, or "cpu"
FLIPUD = 0.5                     # vertical flips are valid: the camera looks straight down

RUN_NAME = f"{HF_DATASET_REPO.split('/')[1]}_{DATASET_REVISION}_{os.path.splitext(MODEL)[0]}_{IMGSZ}_" \
           f"{datetime.now():%Y%m%d_%H%M}"

PUSH_TO_HUB = True
HF_MODEL_REPO = "Greynar/permabot_weed_seg_yolo"
PRIVATE = True

# ----------------------------------------------------------------------------

UPLOADED_FILES = ["weights/best.pt", "results.csv", "results.png", "args.yaml", "confusion_matrix.png",
                  "confusion_matrix_normalized.png", "MaskPR_curve.png", "BoxPR_curve.png", "val_batch0_pred.jpg"]


def download_dataset():
    local_dir = os.path.join(HERE, "dataset", "hub", HF_DATASET_REPO.split("/")[1], DATASET_REVISION)
    snapshot_download(HF_DATASET_REPO, repo_type="dataset", revision=DATASET_REVISION, local_dir=local_dir)
    return os.path.join(local_dir, "data.yaml")


def write_model_card(run_dir, metrics):
    lines = "\n".join(f"| {k} | {v:.3f} |" for k, v in metrics.items())
    card = f"""# {RUN_NAME}

YOLO11-seg fine-tuned for crop / weed instance segmentation (PermaBot project).

- Base model: `{MODEL}`
- Dataset: [`{HF_DATASET_REPO}`](https://huggingface.co/datasets/{HF_DATASET_REPO}) at revision `{DATASET_REVISION}`
- Image size: {IMGSZ}, epochs: {EPOCHS} (patience {PATIENCE}), batch: {BATCH}

## Validation metrics (best epoch)

| metric | value |
|---|---|
{lines}

## Use

```python
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
path = hf_hub_download("{HF_MODEL_REPO}", "{RUN_NAME}/weights/best.pt")
model = YOLO(path)
```
"""
    with open(os.path.join(run_dir, "README.md"), "w") as f:
        f.write(card)


def push_to_hub(run_dir):
    api = HfApi()
    api.create_repo(HF_MODEL_REPO, repo_type="model", private=PRIVATE, exist_ok=True)
    api.upload_folder(folder_path=run_dir, path_in_repo=RUN_NAME, repo_id=HF_MODEL_REPO, repo_type="model",
                      allow_patterns=UPLOADED_FILES + ["README.md"], commit_message=f"Run {RUN_NAME}")
    print(f"Pushed to https://huggingface.co/{HF_MODEL_REPO}/tree/main/{RUN_NAME}")


def main():
    data_yaml = download_dataset()
    print(f"Dataset: {data_yaml}")

    model = YOLO(MODEL)
    model.train(data=data_yaml, imgsz=IMGSZ, epochs=EPOCHS, patience=PATIENCE, batch=BATCH, device=DEVICE,
                flipud=FLIPUD, project=os.path.join(HERE, "runs"), name=RUN_NAME, exist_ok=False)
    run_dir = str(model.trainer.save_dir)

    # Ultralytics validates best.pt at the end of training and keeps the results here.
    metrics = {k.replace("metrics/", ""): float(v) for k, v in model.trainer.metrics.items()
               if k.startswith("metrics/")}
    for k, v in metrics.items():
        print(f"  {k:18s} {v:.3f}")
    write_model_card(run_dir, metrics)

    if PUSH_TO_HUB:
        push_to_hub(run_dir)
    else:
        print("PUSH_TO_HUB is False: nothing uploaded.")
    print(f"Run folder: {run_dir}")


if __name__ == "__main__":
    main()
