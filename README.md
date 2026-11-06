# Crop / weed segmentation — fine-tuning YOLO-seg

Goal: detect and segment crops and weeds in top-down videos of market-garden rows,
fully offline (the final robot will have no internet). This replaces the old
YOLOE + Pl@ntNet + optical-flow pipeline in `../Code_brut_yolo/`.

Scripts are numbered in the order they must be run. Each one has its settings as
variables at the top of the file: edit them, then run `python 0X_....py`.

Environment: `conda activate TestYOLO` (packages: `pip install -r requirements.txt`)

Code: GitHub `gregbaltus/PermaBot_weed_detector_finetune_yolo_seg` (clone with SSH:
`git clone git@github.com:gregbaltus/PermaBot_weed_detector_finetune_yolo_seg.git`).
Only the code is on GitHub (see `.gitignore`); images, datasets and model weights
are on the Hugging Face hub (account `Greynar`).

## Current status

| Step | Script | Status |
|---|---|---|
| 1. Extract images + split | `01_create_dataset_from_video.py` | done (OignonRouge80cm) |
| 2. Label in Roboflow | (manual) | done: 30 images, export v2 (labels corrected) |
| 3. Prepare the dataset + push it to the HF hub | `02_prepare_and_push_dataset.py` | done: v2 pushed to the hub |
| 4. Train YOLO-seg (GPU server) | `03_train.py` | written, tested on CPU (2 epochs); first GPU run to do |
| 5. Evaluate on the test set | `04_evaluate.py` | to do |
| 6. Run the model on a video | `05_infer_video.py` | to do |

Plan: first test the whole pipeline on a few images of one video (stage A), then
label all onion videos (stage B), then all crops (stage C). After that, add the
row prior (plants far from the crop row are weeds) and ground-plane voting
(combine the predictions of the same plant seen in several frames).

## Key decisions (and why)

- **Instance segmentation (YOLO11-seg), not boxes.** Onion and leek leaves are
  long and overlap weeds, so boxes are poor. A mask contains its box, so a
  detection-only model can still be trained from the same labels if needed.
- **Classes**: one class per crop + one `weed` class. Fixed IDs, never change them:

  ```
  0 weed   1 onion   2 leek   3 carrot   4 bean   5 pea
  ```

  Dead or dry material is **not labelled**, so the model treats it as background.
  Roboflow numbers classes alphabetically (`onion`=0, `weed`=1), so
  `02_prepare_and_push_dataset.py` remaps them to the IDs above by name.
- **Split by position along the row, never at random.** Neighbouring frames are
  almost identical, and the allé/retour and 50cm/80cm videos film the same plants.
  A random split would put the same plant in train and test and give fake good
  scores. Our split is the one written in `manifest.csv`; the train/valid/test
  split chosen in Roboflow is ignored.

## Labelling rules

- Label every living plant with a polygon: its crop class, or `weed`.
- A dry leaf still attached to a living onion belongs to the onion polygon.
- A weed that is partly dry but still rooted and partly green is a `weed`.
- Dead material (straw, dry leaves, pulled weeds) is not labelled.
- The cart, wheel, cables, drip pipes and strings are not labelled.
- Label **every** weed, even tiny seedlings: an unlabelled weed teaches the
  model that weeds are background.
- Choose images that contain weeds: v2 has about 4 onions for 1 weed.

## Procedure, step by step

### 1. Extract images from a video — `01_create_dataset_from_video.py`

Set `VIDEO_NAME` at the top, then run the script. It:
- removes duplicated frames (the videos really run at about 5 fps, and the
  container repeats each frame about 6 times);
- measures the camera's travel and saves one image every `FRAME_STEP` image
  widths (0.5 by default), so cart stops do not produce copies;
- cuts the row into `N_BLOCKS` blocks, assigns each block to train/val/test, and
  discards images within `BUFFER` image widths of a boundary between splits.
  Positions are reversed for "retour" videos. With the same `N_BLOCKS` and
  `SEED`, all videos of one row put the same spot in the same split.

Output, in `dataset/frames/<video tag>/`:
- `train/`, `val/`, `test/`: the images, named like
  `OignonRouge80cm_retour_f000900.jpg` (f = frame number in the video);
- `manifest.csv`: one row per image, with its time, position along the row,
  block and split.

The script refuses to overwrite an existing folder: delete it to regenerate.

### 2. Label in Roboflow (manual)

1. Create the project as **Instance Segmentation**.
2. Upload `train/`, `val/` and `test/` **separately**, assigning each to its
   split. Do not upload `manifest.csv`.
3. Label with Smart Polygon (SAM-assisted), following the labelling rules above.
4. Generate a version:
   - Preprocessing: **remove Resize** (it stretches the images), keep Auto-Orient.
   - Augmentation: **none** (Ultralytics augments during training).
5. Export as **YOLOv11** (zip). Because the project is a segmentation project,
   this format contains polygons.
6. Put the zip in `dataset/roboflow_exports/`.

### 3. Prepare and push the dataset — `02_prepare_and_push_dataset.py`

Set `EXPORT_ZIP` (the Roboflow zip) at the top. Run it first with
`PUSH_TO_HUB = False`. It:
- checks that the labels are polygons, not boxes;
- remaps class IDs to the fixed list, by name;
- puts each image back in our split, using `manifest.csv` (the original name,
  e.g. `OignonRouge80cm_retour_f001895`, is kept at the start of Roboflow's
  file names);
- draws the polygons in `previews/`: **look at them** to check the labels;
- writes `data.yaml` for Ultralytics and a `README.md` (dataset card).

Output: `dataset/prepared/<DATASET_NAME>/`, regenerated from scratch at each run.

Then set `PUSH_TO_HUB = True` and run it again. It uploads everything except
`previews/` to the HF dataset repo `Greynar/permabot_onions_weed_seg_first_test`
(private), with a version tag (`HF_TAG`, same number as the Roboflow version, e.g. `v2`). You must be logged in once
on the machine: `hf auth login`.

### 4. Train — `03_train.py` (run on the GPU machine)

First time on the GPU machine:

```bash
git clone git@github.com:gregbaltus/PermaBot_weed_detector_finetune_yolo_seg.git
cd PermaBot_weed_detector_finetune_yolo_seg
pip install -r requirements.txt
hf auth login
```

Then set the variables at the top and run `python 03_train.py`. It:
- downloads the dataset from the hub (`HF_DATASET_REPO` at the tag
  `DATASET_REVISION`) into `dataset/hub/`;
- fine-tunes `MODEL` (default `yolo11s-seg.pt`, pretrained on COCO) at `IMGSZ`
  (default 1280, because the weeds are small), with vertical flips added to
  Ultralytics' default augmentations;
- saves the run in `runs/<RUN_NAME>/` (`RUN_NAME` = dataset, version, model,
  image size and date);
- if `PUSH_TO_HUB` is True, uploads `best.pt`, the training curves, the
  confusion matrix and a model card to the private model repo
  `Greynar/permabot_weed_seg_yolo`, in a folder named `<RUN_NAME>/`. Each run
  has its own folder, so runs can be compared.

If the GPU runs out of memory, lower `BATCH` (or set it to -1 for automatic).

### 5. Evaluate — `04_evaluate.py` (to do)

Compute per-class metrics on the test set.

### 6. Run on a video — `05_infer_video.py` (to do)

Write an annotated video with the predicted masks.

## Folder layout

```
Code_finetune_yolo/
├── README.md
├── requirements.txt
├── 01_create_dataset_from_video.py
├── 02_prepare_and_push_dataset.py
├── ...
└── dataset/                        (not on GitHub)
    ├── frames/<video tag>/         images extracted by 01 + manifest.csv
    ├── roboflow_exports/           zips downloaded from Roboflow
    └── prepared/<dataset name>/    YOLO-seg dataset made by 02 (+ previews/)
```

## Notes for the next recording session (spring)

- Check the camera really records at 15–30 fps (MJPEG mode, lower resolution or
  fixed exposure); the current videos are about 5 fps.
- Point the camera straight down at a fixed height; 50 cm gives about 2× more
  pixels per plant than 80 cm, which matters for small weeds.
- Keep the cart, wheel and cables out of the image.
- A shade hood with its own lighting removes the hard shadows.
- Record the same rows several times during the season (growth stages, weather).
