#!/usr/bin/env python3
"""Convert the CraterDANet LRO-NAC crater dataset to YOLO format.

Source: https://github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data
  LROC NAC CDR, 0.5 m/px, ~20000 human-annotated craters, 22 NAC pairs near
  the Chang'E-4 site (45-46 S, 176.4-178.8 E). Craters below 8 px (4 m) were
  excluded by the authors. Licence GPL-3.0.
  Cite: Huan Yang, Xinchao Xu, Youqing Ma, Yaming Xu, Shaochuang Liu,
  "CraterDANet: A Convolutional Neural Network for Small-Scale Crater
  Detection via Synthetic-to-Real Domain Adaptation," IEEE TGRS.

These are REAL HUMAN ANNOTATIONS, unlike the proposal pseudo-labels in
data/yolo. Metrics computed here are therefore meaningful -- for THIS domain.

Splitting is by SOURCE IMAGE, never by tile: tiles cut from one NAC product
share terrain, so a tile-level split would leak.
"""
from __future__ import annotations

import csv
import json
import random
import re
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "data" / "external" / "Lunar_Crater_Detection_Data" / "LRO_DATA"
OUT = ROOT / "data" / "yolo_external"
TILE, STRIDE = 512, 244
SEED = 20261001
PIXEL_SCALE_M = 0.5


def read_boxes(p: Path) -> list[tuple[float, float, float, float]]:
    out = []
    with open(p) as fh:
        for row in csv.DictReader(fh):
            try:
                x, y = float(row["X"]), float(row["Y"])
                w, h = float(row["W"]), float(row["H"])
            except (KeyError, ValueError):
                continue
            if w > 0 and h > 0:
                out.append((x, y, w, h))      # top-left + size
    return out


def observation_id(stem: str) -> str:
    """Base NAC observation id, e.g. M115143943RE_cal_echo_2_2 -> M115143943."""
    m = re.match(r"(M\d+)", stem)
    return m.group(1) if m else stem


def main() -> int:
    rng = random.Random(SEED)
    all_src = sorted((SRC / "train").glob("*.png")) + sorted((SRC / "test").glob("*.png"))

    # The dataset's own train/test directories SHARE NAC observations: e.g.
    # train/M115143943_mosaic_train_small and test/M115143943RE_cal_echo_2_2
    # both derive from observation M115143943, and 4 of the 8 test
    # observations also appear in the train directory. Splitting on those
    # directories would leak the same ground between train and test, so the
    # split is rebuilt by OBSERVATION ID instead.
    groups: dict[str, list[Path]] = {}
    for p in all_src:
        groups.setdefault(observation_id(p.stem), []).append(p)
    obs = sorted(groups)
    rng.shuffle(obs)
    n = len(obs)
    n_test = max(1, round(0.25 * n))
    n_val = max(1, round(0.20 * n))
    test_obs, val_obs, train_obs = obs[:n_test], obs[n_test:n_test + n_val], obs[n_test + n_val:]
    assign = {"train": [q for o in train_obs for q in groups[o]],
              "val": [q for o in val_obs for q in groups[o]],
              "test": [q for o in test_obs for q in groups[o]]}
    print(f"{n} distinct NAC observations -> "
          f"train {len(train_obs)} / val {len(val_obs)} / test {len(test_obs)}")
    for k, v in assign.items():
        print(f"  {k}: observations {sorted({observation_id(q.stem) for q in v})}")
    # hard check: no observation may appear in two splits
    sets = {k: {observation_id(q.stem) for q in v} for k, v in assign.items()}
    for a in sets:
        for b in sets:
            if a < b:
                shared = sets[a] & sets[b]
                assert not shared, f"observation leak between {a} and {b}: {shared}"
    print("  leakage check: no observation shared between splits OK")

    stats = {}
    for split, paths in assign.items():
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)
        n_tiles = n_obj = 0
        for p in paths:
            txt = p.with_suffix(".txt")
            if not txt.exists():
                print(f"  !! no annotation for {p.name}")
                continue
            boxes = read_boxes(txt)
            im = np.array(Image.open(p).convert("L"))
            H, W = im.shape
            for r0 in range(0, max(H - TILE, 0) + 1, STRIDE):
                for c0 in range(0, max(W - TILE, 0) + 1, STRIDE):
                    r1, c1 = min(r0 + TILE, H), min(c0 + TILE, W)
                    sub = im[r0:r1, c0:c1]
                    if sub.shape != (TILE, TILE):
                        continue
                    lines = []
                    for (x, y, w, h) in boxes:
                        # keep a crater only if its CENTRE is in this tile, so
                        # no object is counted twice across overlapping tiles
                        cx, cy = x + w / 2.0, y + h / 2.0
                        if not (c0 <= cx < c1 and r0 <= cy < r1):
                            continue
                        # clip the box to the tile; a clipped box is still a
                        # valid training target but its extent is truncated
                        bx0, by0 = max(x, c0), max(y, r0)
                        bx1, by1 = min(x + w, c1), min(y + h, r1)
                        bw, bh = bx1 - bx0, by1 - by0
                        if bw < 2 or bh < 2:
                            continue
                        lines.append("0 {:.6f} {:.6f} {:.6f} {:.6f}".format(
                            ((bx0 + bx1) / 2 - c0) / TILE,
                            ((by0 + by1) / 2 - r0) / TILE,
                            bw / TILE, bh / TILE))
                    if not lines:
                        continue
                    name = f"{p.stem}_r{r0:04d}_c{c0:04d}"
                    Image.fromarray(sub).save(OUT / "images" / split / f"{name}.png")
                    (OUT / "labels" / split / f"{name}.txt").write_text("\n".join(lines))
                    n_tiles += 1
                    n_obj += len(lines)
        stats[split] = {"tiles": n_tiles, "objects": n_obj,
                        "source_images": [p.stem for p in paths]}
        print(f"{split}: {n_tiles} tiles, {n_obj} objects from {len(paths)} source images")

    (OUT / "dataset.yaml").write_text(
        "# CraterDANet LRO-NAC crater dataset, converted to YOLO\n"
        "# REAL HUMAN ANNOTATIONS. Licence GPL-3.0. Cite Yang et al., IEEE TGRS.\n"
        "# 0.5 m/px, Chang'E-4 region (45-46 S). NOT Malapert: see domain shift note.\n"
        f"path: {OUT}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames: [crater]\n")
    (OUT / "PROVENANCE.json").write_text(json.dumps({
        "source_repo": "https://github.com/yizuifangxiuyh/Lunar_Crater_Detection_Data",
        "citation": ("Huan Yang, Xinchao Xu, Youqing Ma, Yaming Xu, Shaochuang Liu, "
                     "CraterDANet: A Convolutional Neural Network for Small-Scale "
                     "Crater Detection via Synthetic-to-Real Domain Adaptation, "
                     "IEEE Transactions on Geoscience and Remote Sensing."),
        "licence": "GPL-3.0",
        "instrument": "LRO NAC CDR", "pixel_scale_m": PIXEL_SCALE_M,
        "region": "near Chang'E-4 landing site, 45-46 S, 176.4-178.8 E",
        "source_label_type": "human", "review_status": "published",
        "authors_note": "craters below 8 px (4 m) were excluded by the dataset authors",
        "split_rule": ("by NAC OBSERVATION ID, never by tile and never by the dataset\u2019s own train/test directories, which share observations"),
        "tile_px": TILE, "stride_px": STRIDE, "seed": SEED,
        "stats": stats,
        "domain_shift_vs_malapert": {
            "pixel_scale": "0.5 m/px source vs 1.0 m/px Malapert mosaic",
            "latitude": "45-46 S source vs 85.9 S Malapert",
            "illumination": ("mid-latitude incidence vs grazing polar incidence; "
                             "shadow geometry and contrast differ substantially"),
            "consequence": ("metrics from this dataset do NOT transfer as Malapert "
                            "performance; local reviewed labels are still required "
                            "to validate on Malapert"),
        },
    }, indent=2))
    print("\nwrote", OUT / "dataset.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
