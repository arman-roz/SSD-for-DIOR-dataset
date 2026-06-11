# Project Changes — DIOR Dataset Integration

We are switching from the LVIS/COCO dataset to the **DIOR** dataset for training both SSD300 and YOLOv8.
This file tracks every file we touched, what we changed, and what is still left to do.

---

## What is DIOR?

- **11,725 satellite images** (800×800 px) for training/validation
- **11,738 satellite images** for testing (no labels)
- **20 object classes**: airplane, airport, baseballfield, basketballcourt, bridge, chimney, dam,
  Expressway-Service-area, Expressway-toll-station, golffield, groundtrackfield, harbor, overpass,
  ship, stadium, storagetank, tenniscourt, trainstation, vehicle, windmill
- Labels are stored as **Pascal VOC XML** files (one `.xml` per image)

---

## How We Handle XML Annotations — The Big Picture

DIOR stores its labels as **Pascal VOC XML files** — one `.xml` per image.
Neither SSD300 nor YOLOv8 can read XML directly.
The old code used COCO JSON (`.json`). We replaced all of that.
Here is exactly how each model gets its data:

---

### The Problem

```
DIOR gives us:   00001.jpg  +  00001.xml  (Pascal VOC XML)

SSD300 expects:  image tensor  +  box tensors  +  label tensors   (in Python memory)
YOLOv8 expects:  image file    +  00001.txt  (YOLO format, on disk)
```

Neither model ever receives an XML file.
We convert XML into whatever each model needs — differently for each.

---

### SSD300 — XML is read live, converted to tensors on the fly

Every time the DataLoader asks for a sample during training, `datasets.py` does this:

```
00001.xml  ──►  _parse_xml()  ──►  boxes tensor  (N, 4) absolute pixel coords
                                    labels tensor (N,)   integers 1–20

00001.jpg  ──►  PIL Image

Both go into  transform()  from utils.py
  - augments the image (brightness, crop, flip)
  - resizes to 300×300
  - converts boxes from absolute pixels → fractions (0–1)
  - normalises the image

Model receives:  image (3, 300, 300)  |  boxes (N, 4)  |  labels (N,)
```

**No files are created.** Conversion happens in Python memory at training time.
The old code read a `annotations.json` COCO file instead — we replaced that entirely
with `_parse_xml()` in `datasets.py`.

---

### YOLOv8 — XML is pre-converted to `.txt` files once before training

Ultralytics (the YOLOv8 library) has its own built-in data loader.
It does **not** accept a PyTorch Dataset class — it reads directly from the filesystem.
It expects one `.txt` file per image in YOLO format:

```
# 00001.txt  — one line per object
0 0.356250 0.412500 0.078125 0.092500
   │         │         │         │
class_id   cx/W      cy/H      w/W     h/H   (all normalised 0–1)
```

So we run `convert_to_yolo.py` **once before training** to pre-convert all XML files:

```
00001.xml  ──►  convert_to_yolo.py  ──►  00001.txt  (written to JPEGImages-trainval/)

Also writes:
  data/split_005_train.txt   list of image paths for the 5% training subset
  data/split_005_val.txt     list of image paths for validation
  data/split_005.yaml        config file telling YOLOv8 where images/labels are
  ... (same for split_025, 050, 075, 100)

Then training runs:   python train.py   (reads .txt files, never touches XML again)
```

The old `convert_to_yolo.py` read COCO JSON from a different project — we rewrote it
entirely to read DIOR XML instead.

---

### Side-by-side comparison

| | SSD300 | YOLOv8 |
|---|---|---|
| Raw input | Pascal VOC XML | Pascal VOC XML |
| Who reads XML | `_parse_xml()` in `datasets.py` | `convert_to_yolo.py` (one-time script) |
| When conversion happens | Live, every training batch | Once, before training starts |
| What the model receives | PyTorch tensors | `.txt` files on disk |
| Label format | `boxes (N,4)` + `labels (N,)` as tensors | `class cx cy w h` per line in `.txt` |
| Coordinate format | Fractional 0–1 (after `transform()`) | Fractional 0–1 (computed in converter) |
| Label indexing | 1–20 (0 = background, reserved by SSD) | 0–19 (standard 0-indexed) |

---

## Data Pipeline — `Python_datareader/` — DO NOT TOUCH

These files are already written and working. We use them as-is.

| File | What it does |
|---|---|
| `dior_dataset.py` | Reads DIOR images and XML annotations for PyTorch training |
| `generate_splits.py` | Already ran — created the `.npy` files below |
| `split_005_indices.npy` | Index list for 5% of training data (586 images) |
| `split_025_indices.npy` | Index list for 25% of training data (2,931 images) |
| `split_050_indices.npy` | Index list for 50% of training data (5,862 images) |
| `split_075_indices.npy` | Index list for 75% of training data (8,793 images) |
| `split_100_indices.npy` | Index list for 100% of training data (11,725 images) |

---

## Files We Changed

### ✅ `SSD300/utils.py` — DONE

This file defines the class names and colours used for drawing detection boxes.

| What | Before | After |
|---|---|---|
| Class list | 50 LVIS classes (suitcase, banana, zebra …) | 20 DIOR classes (airplane, ship, vehicle …) |
| Label numbering | background=0, classes 1–50 | background=0, classes 1–20 |
| COCO ID mapping | Had a lookup table converting COCO category numbers | **Removed** — DIOR uses class name strings directly, no number lookup needed |
| Visualization colors | 51 colors (many duplicates at the end) | 21 colors, one per class + background |

Everything else in this file (augmentation, box math, mAP calculation) was left untouched.

---

### ✅ `SSD300/datasets.py` — DONE (updated twice)

This file tells SSD300 how to load training images and their labels.

**First update — replaced LVIS with DIOR:**

| What | Before | After |
|---|---|---|
| Dataset class | `LVISDataset` — read COCO JSON files | `DIORDatasetSSD` — reads Pascal VOC XML files |
| Annotation source | `annotations.json` (COCO format) | One `.xml` file per image |
| Split selection | Fixed folder structure | Reads `train.txt` / `val.txt`, then filters by `.npy` index file |
| Label numbers | Converted COCO category IDs (3, 36, 45 …) | Converts class name strings → numbers 1–20 |
| Box coordinates returned | Fractional (0–1) after SSD transform | Fractional (0–1) after SSD transform ✓ same |
| Augmentation | Applied to image AND boxes together | Applied to image AND boxes together ✓ same |

**Second update — added `split='val'` and `split='train'` support:**

The original `DIORDatasetSSD` only supported `split='trainval'` (both train.txt + val.txt = 11,725 images) or `split='test'`. `train.py` needs a validation loader that uses **only** `val.txt` (5,863 images), so we added two new split modes:

| Split value | Reads from | Image folder |
|---|---|---|
| `'train'` | `Main/train.txt` only | `JPEGImages-trainval/` |
| `'val'` | `Main/val.txt` only | `JPEGImages-trainval/` |
| `'trainval'` | both txt files | `JPEGImages-trainval/` |
| `'test'` | `Main/test.txt` | `JPEGImages-test/` |

> **Why not reuse `DIORDatasetSSD` from `dior_dataset.py`?**
> The pipeline version only transforms the image. SSD's transform must also move the bounding boxes
> (crop, flip, resize them together with the image). So SSD needs its own dataset class.

---

### ✅ `SSD300/detect.py` — DONE

This file runs detection on a single image after training.

| What | Before | After |
|---|---|---|
| Test image path | `/media/ssd/ssd data/VOC2007/JPEGImages/000001.jpg` (Linux path, wrong dataset) | `../JPEGImages-test/00001.jpg` (DIOR test image) |

Nothing else changed.

---

### ✅ `SSD300/model.py` — DONE

This file defines the ResNet50 backbone and the full SSD300 architecture.

| What | Before | After |
|---|---|---|
| Backbone weights | `resnet50(weights=None)` — random, training from scratch | `resnet50(weights='IMAGENET1K_V1')` — ImageNet-1K pretrained |
| Print message | "random weights — training from scratch" | "ImageNet-1K pretrained weights" |

Only the backbone layers get the pretrained weights (`entry`, `layer1`, `layer2`, `layer3`).
The auxiliary convolutions and prediction head always initialise with Kaiming uniform — they are
new layers that did not exist in the original ResNet50 and have no pretrained equivalent.

---

### ✅ `SSD300/train.py` — DONE

This is the main training script for SSD300.

| Line | Before | After |
|---|---|---|
| Import | `from datasets import LVISDataset` | `from datasets import DIORDatasetSSD` |
| Data path | `data_folder = '.../Dataset_1/splits/split_005/train'` (LVIS folder) | `DATA_DIR = os.path.join(os.path.dirname(__file__), '..')` (project root) |
| Split file | not present | `SPLIT_FILE = os.path.join(..., 'Python_datareader', f'{SPLIT}_indices.npy')` |
| n_classes comment | `51 (50 LVIS classes + background)` | `21 (20 DIOR classes + background)` |
| Train dataset | `LVISDataset(data_folder, split='train')` | `DIORDatasetSSD(DATA_DIR, split='trainval', split_file=SPLIT_FILE, split_type='TRAIN')` |
| Val dataset | `LVISDataset(val_folder, split='val')` | `DIORDatasetSSD(DATA_DIR, split='val', split_file=None, split_type='TEST')` |

Training logic, early stopping, LR scheduler, CSV logging, and checkpoint saving are all unchanged.

---

## Files Still To Do

### ✅ `YOLOv8/convert_to_yolo.py` — DONE

Full rewrite. The old version read COCO JSON from a different project folder — nothing was salvageable.

What the new version does:
1. Reads `Main/train.txt` + `Main/val.txt` IDs into one combined list (11,725 total) — same order as `_DIORBase` and `generate_splits.py`.
2. Writes one `.txt` label file per image **directly into `JPEGImages-trainval/`** alongside the `.jpg` files. This is where ultralytics looks when the image path contains no `/images/` component. Done once, shared across all splits.
3. For each split, applies the `.npy` index file to the combined ID list to get the training subset, then writes:
   - `YOLOv8/data/split_NNN_train.txt` — one absolute image path per line (training subset)
   - `YOLOv8/data/split_NNN_val.txt` — one absolute image path per line (full val.txt, fixed across all splits)
   - `YOLOv8/data/split_NNN.yaml` — YAML config pointing to the two list files, with 20 DIOR class names

| Split | Training images | Val images |
|---|---|---|
| split_005 | 586 | 5,863 |
| split_025 | 2,931 | 5,863 |
| split_050 | 5,862 | 5,863 |
| split_075 | 8,793 | 5,863 |
| split_100 | 11,725 | 5,863 |

---

### ✅ `YOLOv8/detect.py` — DONE

One line fix:

| What | Before | After |
|---|---|---|
| Test image path | `'path/to/image.jpg'` (placeholder) | `Path(__file__).resolve().parent.parent / 'JPEGImages-test' / '00001.jpg'` |

Uses `Path(__file__)` so the path is always correct regardless of which directory the script is run from.

---

### ✅ `YOLOv8/train.py` — DONE

| What | Before | After |
|---|---|---|
| Model loading | `YOLO('yolov8n.yaml')` — architecture only, random weights | `YOLO('yolov8n.pt')` — ImageNet-1K pretrained backbone + COCO-trained head |
| Pretrained flag | `pretrained=False` | `pretrained=True` |
| Comment | "no pretrained weights (training from scratch)" | "ImageNet-1K pretrained backbone" |

Everything else (hyperparameters, CSV logging, early stopping, device selection) is unchanged.

---

### ✅ `SSD300/eval.py` — DONE

This file loads a trained checkpoint and computes mAP across the full validation set.

| Line | Before | After |
|---|---|---|
| Import | `from datasets import LVISDataset` | `from datasets import DIORDatasetSSD` |
| Data path | `val_folder = '.../Dataset_1/splits/split_005/val'` (LVIS folder) | `DATA_DIR = os.path.join(os.path.dirname(__file__), '..')` (project root) |
| Val dataset | `LVISDataset(val_folder, split='val')` | `DIORDatasetSSD(DATA_DIR, split='val', split_file=None, split_type='TEST')` |
| Comment | "LVIS has no 'difficult' flag" | "DIOR has no 'difficult' flag" |
| Comment | "Calculate mAP across all 50 LVIS classes" | "Calculate mAP across all 20 DIOR classes" |

`calculate_mAP()` in `utils.py` already uses `len(label_map)` = 21 — no changes needed there.

---

---

## Bug Fixes — Round 2

These fixes were applied after a detailed expert audit of all SSD300 files.
Every change is documented with what was wrong, what it is now, and why.

---

### `SSD300/utils.py` — 5 fixes

#### Fix 1 — `label_color_map` assigned wrong colors to every class

| | Detail |
|---|---|
| **Before** | `{k: distinct_colors[i] for i, k in enumerate(label_map.keys())}` — iterated insertion order; `label_map` inserts the 20 class names first, then appends `'background'` last, so `airplane` received color index 0 (`#FFFFFF` white) and `background` received color index 20 (`#808080` gray) |
| **After** | `{k: distinct_colors[v] for k, v in label_map.items()}` — uses the label's integer ID (0–20) as the color index; `background` (ID=0) now correctly gets `#FFFFFF` and each class gets its own distinct color |
| **Why** | Every bounding box was drawn in the wrong color. Background boxes appeared gray; airplane boxes appeared white instead of red. |

#### Fix 2 — `flip()` mutated the caller's bounding box tensor in-place

| | Detail |
|---|---|
| **Before** | `new_boxes = boxes` — reference assignment, not a copy; writing to `new_boxes[:, 0]` and `new_boxes[:, 2]` silently overwrote the original tensor |
| **After** | `new_boxes = boxes.clone()` — creates an independent copy before any in-place writes |
| **Why** | The original `boxes` tensor (created in `__getitem__`) had columns 0 and 2 permanently corrupted after each horizontal flip call. Did not crash in current code (box is never re-read after the call) but was a landmine for any future use. |

#### Fix 3 — Added `vflip()` and called it in `transform()`

| | Detail |
|---|---|
| **Before** | Only horizontal flip was applied during training augmentation |
| **After** | A new `vflip()` function mirrors the image and boxes vertically; it is applied with 50% probability during TRAIN augmentation, right after the horizontal flip |
| **Why** | DIOR is top-down satellite imagery — objects like ships, vehicles, and airplanes appear at arbitrary orientations. There is no canonical "up". Horizontal flip alone captures only one axis of symmetry. Adding vertical flip means the model sees all four axis-aligned reflections, directly improving mAP on objects oriented toward the top/bottom of the image. |

#### Fix 4 — `torch.exp()` overflow to `inf` in `gcxgcy_to_cxcy`

| | Detail |
|---|---|
| **Before** | `torch.exp(gcxgcy[:, 2:] / 5)` — unclamped; very large predicted offsets (common in early training when the prediction head is random) overflow to `inf`, producing `NaN` decoded boxes |
| **After** | `torch.exp(gcxgcy[:, 2:].clamp(max=10.) / 5)` — caps the exponent argument at 10 before the exp, which limits the maximum decoded size to `e^2 ≈ 7.4` times the prior size, which is already unrealistically large |
| **Why** | `inf` propagates through IoU calculations and eventually into the loss, turning all weights `NaN` and permanently destroying the model in a single batch. |

#### Fix 5 — Stale comment "trained from scratch"

| | Detail |
|---|---|
| **Before** | Comment said "used as standard normalization for ResNet50 trained from scratch" |
| **After** | Comment says "matches the distribution ResNet50 was pretrained on" |
| **Why** | We use ImageNet-1K pretrained weights; the comment was left over from before the pretrained weight change. |

---

### `SSD300/model.py` — 4 fixes

#### Fix 6 — Division by zero in `MultiBoxLoss`

| | Detail |
|---|---|
| **Before** | `conf_loss = (...) / n_positives.sum().float()` — if an entire batch has no positive priors (e.g., all objects cropped away by aggressive augmentation), divides by zero → `NaN` loss → `NaN` gradients → all model weights destroyed |
| **After** | `n_pos = n_positives.sum().float().clamp(min=1.)` then divide by `n_pos` |
| **Why** | Rare but possible with DIOR's aggressive random crop. A single `NaN` loss corrupts the model irreversibly in one step. |

#### Fix 7 — Stale docstring "random weight initialization"

| | Detail |
|---|---|
| **Before** | `ResNet50Base` docstring said "Trained from scratch with random weight initialization" |
| **After** | "ResNet50 backbone with ImageNet-1K pretrained weights" |
| **Why** | We changed to pretrained weights earlier; the docstring was never updated. |

#### Fix 8 — Stale comment "on top of the VGG base"

| | Detail |
|---|---|
| **Before** | `AuxiliaryConvolutions.__init__` had comment "on top of the VGG base" |
| **After** | "on top of the ResNet50 base" |
| **Why** | Copy-pasted from original VGG SSD code; never updated when backbone was swapped. |

#### Fix 9 — Wrong prior-box count in comments ("2116" → "2166")

| | Detail |
|---|---|
| **Before** | Two comments in `PredictionConvolutions.forward` said "there are a total 2116 boxes on this feature map" for the `conv7` branch |
| **After** | "2166" (19 × 19 × 6 = 2166) |
| **Why** | Off-by-50 typo. Total prior box count still summed to correct 8732; only the per-map comment was wrong. |

---

### `SSD300/train.py` — 7 fixes

#### Fix 10 — Data leakage: training set contained validation images

| | Detail |
|---|---|
| **Before** | `DIORDatasetSSD(DATA_DIR, split='trainval', split_file=SPLIT_FILE, ...)` — `generate_splits.py` built indices over the full 11,725 trainval pool; roughly half the selected indices (e.g., ~293 out of 586 for the 5% split) pointed to `val.txt` images; those images also appeared in the validation loader |
| **After** | The raw `.npy` indices are filtered to `< TRAIN_SIZE (5862)` before being passed to the dataset: `TRAIN_INDICES = _raw_indices[_raw_indices < TRAIN_SIZE]`. These are passed directly as a numpy array (no temp file needed — `datasets.py` now accepts both a path and an array) |
| **Why** | Training on val images and then evaluating on them inflates mAP. The model was effectively being tested on data it had seen during training. |

#### Fix 11 — Warmup re-triggered on checkpoint resume, spiking LR

| | Detail |
|---|---|
| **Before** | `if global_iter < WARMUP_ITERS:` — `global_iter` is recomputed from the resumed epoch number, so for small splits (few batches/epoch) it re-entered the warmup window on resume; warmup used the global `lr=1e-3`, overriding the scheduler-reduced LR saved in the checkpoint (e.g., could spike from `1e-5` back to `7e-4`) |
| **After** | `if start_epoch == 0 and global_iter < WARMUP_ITERS:` — warmup only runs on a fresh training start |
| **Why** | Resuming with an inflated LR destabilizes training and discards the scheduler's history of decay. |

#### Fix 12 — Pretrained backbone and random heads shared the same learning rate

| | Detail |
|---|---|
| **Before** | Single `lr=1e-3` for all parameters; only biases vs. weights were distinguished |
| **After** | Four param groups: backbone weights `lr×0.1`, backbone biases `lr×0.2`, head weights `lr`, head biases `lr×2` |
| **Why** | Applying the full LR to a pretrained ResNet50 backbone scrambles the learned features within the first few epochs. The heads (randomly initialized) need the high LR to converge; the backbone needs 10x lower LR to fine-tune without forgetting what ImageNet taught it. |

#### Fix 13 — Backbone BatchNorm updated with batch_size=8, corrupting pretrained features

| | Detail |
|---|---|
| **Before** | `model.train()` put all layers including backbone BatchNorm into train mode; BN used noisy mini-batch statistics (8 samples) and continuously updated its running mean/variance |
| **After** | `model.base.apply(_set_bn_eval)` is called right after `model.train()` in the `train()` function; backbone BN layers are switched back to eval mode, using their stable ImageNet-trained running statistics |
| **Why** | With 8 images per batch, per-batch mean and variance estimates are extremely noisy. Using them to normalize backbone activations destroys the stable features the pretrained weights learned. Freezing BN in the backbone is standard practice when fine-tuning with small batches. |

#### Fix 14 — `torch.load` missing `map_location` (crashes loading GPU checkpoint on CPU)

| | Detail |
|---|---|
| **Before** | `torch.load(checkpoint)` — no `map_location`; PyTorch tries to deserialize tensors onto their original CUDA device; fails with `RuntimeError` on any CPU-only machine |
| **After** | `torch.load(checkpoint, map_location=device)` |
| **Why** | The server may differ from the development machine. Without this, loading a GPU-saved checkpoint on a CPU machine crashes immediately. Same fix applied to `eval.py` and `detect.py`. |

#### Fix 15 — `MIN_DELTA=0.001` was below the noise floor of SSD loss

| | Detail |
|---|---|
| **Before** | `MIN_DELTA = 0.001` — SSD val loss fluctuates ~±0.05–0.1 per epoch from mini-batch sampling noise; any downward fluctuation above 0.001 (which is almost always) reset the early-stopping counter, making `PATIENCE=10` effectively infinite |
| **After** | `MIN_DELTA = 0.01` |
| **Why** | Early stopping never triggered, so training ran until `MAX_EPOCHS` regardless of convergence. |

#### Fix 16 — Gradient clipping was disabled

| | Detail |
|---|---|
| **Before** | `grad_clip = None` |
| **After** | `grad_clip = 4.0` |
| **Why** | The randomly initialized prediction and auxiliary heads attached to a pretrained backbone produce large gradients in early epochs. Clipping at 4.0 (standard SSD value) prevents gradient explosion during warmup. |

---

### `SSD300/eval.py` and `SSD300/detect.py` — 1 fix each

#### Fix 17 — `torch.load` missing `map_location`

Same as Fix 14 above — same one-line fix applied to `eval.py:30` and `detect.py:18`.

---

### `SSD300/datasets.py` — 1 fix

#### Fix 18 — `split_file` now accepts a numpy array in addition to a file path

| | Detail |
|---|---|
| **Before** | `split_file` had to be a file path string; `np.load(split_file)` was called unconditionally |
| **After** | `isinstance(split_file, (str, os.PathLike))` check: if it's a path, load from disk; if it's already an array, use it directly |
| **Why** | Fix 10 (data leakage) filters the raw indices in `train.py` and needs to pass the filtered numpy array directly without writing a temporary file. |

---

## Files That Stay Unchanged

| File | Why |
|---|---|
| `Python_datareader/dior_dataset.py` | Already correct for DIOR |
| `Python_datareader/generate_splits.py` | Already ran, `.npy` files exist |
| `YOLOv8/eval.py` | Ready once a trained checkpoint exists |
| `SSD300/create_data_lists.py` | Irrelevant VOC2007 leftover — ignore |
