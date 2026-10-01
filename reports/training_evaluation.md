# Crater detector — training and evaluation

Executed 2026-10-01. Every number below is from the actual run; nothing is
estimated.

## What was trained, and on what
| | |
|---|---|
| Model | YOLO11n, pretrained `yolo11n.pt` |
| Framework | ultralytics 8.4.170, torch 2.14.1+cu130 |
| Device | **CPU** (4-core Intel Xeon; no GPU in this container) |
| Image size | 512 |
| Epochs | 60 (completed in 0.219 h), patience 15, seed 20261001, deterministic |
| Parameters | 2,582,347 |
| Weights | `artifacts/runs/crater_yolo11n/weights/best.pt`, 5454170 bytes, md5 `67c5cf28c8abbcdc7f57e2e92aa68b5e` |

**Labels: real human annotations.** CraterDANet LRO-NAC dataset
(https://github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data), LROC NAC CDR
at 0.5 m/px near the Chang'E-4 site (45–46 S, 176.4–178.8 E), ~20,000 craters,
licence GPL-3.0. Cite: Yang, Xu, Ma, Xu, Liu, "CraterDANet…", IEEE TGRS.
The dataset authors excluded craters below 8 px (4 m).

**Not** the Robbins catalogue (zero craters in range over our ROI) and
**not** the model-free proposals (rejected as a label source — see below).

## Split: rebuilt to remove a leak in the source dataset
The dataset's own `train/` and `test/` directories **share NAC
observations**: `train/M115143943_mosaic_train_small` and
`test/M115143943RE_cal_echo_2_2` both derive from observation M115143943, and
four of the eight test observations also appear under train. Using that split
would place the same ground in train and test.

The split was rebuilt **by NAC observation id**, with an assertion that no
observation appears in two splits:

| Split | Observations | Tiles | Objects |
|---|---|---|---|
| train | 9 | 65 | 22,682 |
| val | 3 | 12 | 6,227 |
| test | 4 | 25 | 10,477 |

## Results
Validation (best epoch, during training):
| P | R | mAP50 | mAP50-95 |
|---|---|---|---|
| 0.710 | 0.631 | 0.629 | 0.241 |

**Untouched test split** (observation-disjoint from train and val):
| P | R | mAP50 | mAP50-95 |
|---|---|---|---|
| **0.647** | **0.608** | **0.578** | **0.194** |

Inference 229 ms/image on CPU at 512 px.

## Caveats that must travel with these numbers

1. **These are NOT Malapert performance.** The training domain is 45–46 S at
   0.5 m/px with mid-latitude illumination. The Malapert ROI is 85.9 S at
   1.0 m/px with grazing polar illumination, where shadow geometry and
   contrast differ substantially. Performance on Malapert is **UNVALIDATED**
   and cannot be inferred from this table.
2. **max_det capping.** Ultralytics warned that test tiles hold up to 673
   objects against a default `max_det=300`, and auto-raised it to 673 for the
   test run. The training-time validation ran at the default while val tiles
   average ~519 objects each, so **the validation metrics above are likely
   capped and understated**. A production run should use smaller tiles or an
   explicitly raised `max_det`.
3. **mAP50-95 is low (0.194) by construction.** Objects are tiny — the
   dataset floor is 8 px — so box localisation is coarse relative to object
   size and the high-IoU thresholds punish it heavily. mAP50 is the more
   meaningful figure at this object scale.
4. **CPU-only, 60 epochs.** This is not the ~100-epoch budget the brief
   anticipates, and no hyperparameter search was done. The same script runs
   on a GPU with `--device 0`.
5. **Thresholds were not tuned.** No operating point was selected against the
   test set; defaults were used throughout.

## Why the model-free proposals were not used as labels
`src/crater/proposals.py` was built, inspected and rejected for training.
Two defects were found and fixed — a per-scale 99th-percentile threshold
(relative, so it fired on ~1% of pixels regardless of content) and
`fftconvolve` zero-padding (which manufactured dense spurious clusters along
every tile edge). After fixing, a flat noise field yields zero proposals and
synthetic craters are recovered. It still produced offset duplicates around
rims and mis-sized a 200 px synthetic crater as 80 px, so it is retained as an
**annotation aid** (`source_label_type=model_proposal`,
`review_status=unreviewed`) and never as ground truth.

## What would make this a Malapert result
1. A reviewed local annotation set on the Malapert mosaic (the only way to
   measure recall and precision at 86 S).
2. Fine-tuning this model on it, with the 0.5 → 1.0 m/px scale handled
   explicitly (resample Malapert tiles to 0.5 m/px at inference, or retrain
   at the native scale).
3. Then, and only then, full-ROI inference, deduplication, rim measurement
   and an R-plot over the measured 439.918 km² survey area.
