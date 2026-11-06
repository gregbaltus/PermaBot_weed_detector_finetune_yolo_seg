# Crop / weed segmentation — fine-tuning YOLO-seg

Goal: detect and segment crops and weeds in top-down videos of market-garden rows,
fully offline (the final robot will have no internet). This replaces the old
YOLOE + Pl@ntNet + optical-flow pipeline in `../Code_brut_yolo/`.

Scripts are numbered in the order they must be run. Each one has its settings as
variables at the top of the file: edit them, then run `python 0X_....py`.

Environment: `conda activate TestYOLO`

## Current status

| Step | Script | Status |
|---|---|---|
| 1. Extract images + split | `01_create_dataset_from_video.py` | done (OignonRouge80cm) |
| 2. Label in Roboflow | (manual) | done: 30 images, export v1 |
| 3. Prepare the dataset | `02_prepare_dataset.py` | to do |
| 4. Push the dataset to the HF hub | `03_push_dataset_to_hub.py` | to do |
| 5. Train YOLO-seg (GPU server) | `04_train.py` | to do |
| 6. Evaluate on the test set | `05_evaluate.py` | to do |
| 7. Run the model on a video | `06_infer_video.py` | to do |

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
  `02_prepare_dataset.py` remaps them to the IDs above by name.
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
- Choose images that contain weeds: v1 has about 6 onions for 1 weed.

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

### 3. Prepare the dataset — `02_prepare_dataset.py` (to do)

- unzip the Roboflow export;
- check that the labels are polygons, not boxes;
- remap class IDs to the fixed list;
- put each image back in our split, using `manifest.csv` (the original name,
  e.g. `OignonRouge80cm_retour_f001895`, is kept at the start of Roboflow's
  file names);
- draw the polygons on the images, so the result can be checked by eye;
- write `data.yaml` for Ultralytics.

### 4. Push the dataset to the hub — `03_push_dataset_to_hub.py` (to do)

Hugging Face dataset repo: `permabot_onions_weed_seg_first_test`.

### 5. Train — `04_train.py` (to do, run on the GPU server)

Fine-tune YOLO11-seg from pretrained weights and push the trained model to the
HF hub.

### 6. Evaluate — `05_evaluate.py` (to do)

Compute per-class metrics on the test set.

### 7. Run on a video — `06_infer_video.py` (to do)

Write an annotated video with the predicted masks.

## Folder layout

```
Code_finetune_yolo/
├── README.md
├── 01_create_dataset_from_video.py
├── 02_prepare_dataset.py           (to do)
├── ...
└── dataset/
    ├── frames/<video tag>/         images extracted by 01 + manifest.csv
    └── roboflow_exports/           zips downloaded from Roboflow
```

## Notes for the next recording session (spring)

- Check the camera really records at 15–30 fps (MJPEG mode, lower resolution or
  fixed exposure); the current videos are about 5 fps.
- Point the camera straight down at a fixed height; 50 cm gives about 2× more
  pixels per plant than 80 cm, which matters for small weeds.
- Keep the cart, wheel and cables out of the image.
- A shade hood with its own lighting removes the hard shadows.
- Record the same rows several times during the season (growth stages, weather).
