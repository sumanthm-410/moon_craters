#!/usr/bin/env python3
"""Build a YOLO dataset from the Malapert mosaic using model-free proposals.

PROVENANCE WARNING, which must travel with every result derived from this
dataset: the labels are UNREVIEWED MODEL PROPOSALS from
``crater.proposals`` (source_label_type = "model_proposal",
review_status = "unreviewed").  They are NOT human ground truth and NOT the
Robbins catalogue, which contains zero craters in the approved 20-1000 m
range over this ROI (reports/robbins_audit.md).

Any metric computed against these labels measures agreement with the proposal
generator, not detection of real craters.
"""
from __future__ import annotations

import csv
import json
import random
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crater import proposals as pr
from crater import splits, tiling

X0, Y1, N, PIX = -11000.0, 132000.0, 21000, 1.0
X1, Y0 = X0 + N * PIX, Y1 - N * PIX
TILE, MARGIN, BUFFER, SIDE = 512, 64, 1000.0, 8600.0
COG = ROOT / "artifacts" / "malapert_LO1_cog.tif"
OUT = ROOT / "data" / "yolo"
PER_SPLIT = {"train": 400, "val": 80, "test": 80}
SEED = 20261001
MIN_D_M, MAX_D_M = 20.0, 1000.0


def main() -> int:
    random.seed(SEED)
    rng = random.Random(SEED)

    val = box(X0, Y0, X0 + SIDE, Y0 + SIDE)
    test = box(X1 - SIDE, Y1 - SIDE, X1, Y1)
    train = box(X0, Y0, X1, Y1).difference(val).difference(test)
    regions = [splits.SplitRegion("train", train), splits.SplitRegion("val", val),
               splits.SplitRegion("test", test)]
    tr = Affine.translation(X0, Y1) @ Affine.scale(PIX, -PIX)
    grid = tiling.TileGrid(tile_px=TILE, margin_px=MARGIN, transform=tr,
                           pixel_scale_m=PIX, level=0)
    tiles = {t.tile_id: t for t in tiling.iter_tiles(grid, width=N, height=N)}
    asg = splits.assign_tiles_to_splits(list(tiles.values()), regions, buffer_m=BUFFER)

    by_split: dict[str, list[str]] = {}
    for a in asg.assignments:
        if a.split:
            by_split.setdefault(a.split, []).append(a.tile_id)
    print({k: len(v) for k, v in by_split.items()})

    for split in ("train", "val", "test"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    prov_rows = []
    az = None
    totals = {}
    with rasterio.open(COG) as ds:
        for split, ids in by_split.items():
            chosen = rng.sample(sorted(ids), min(PER_SPLIT[split], len(ids)))
            n_obj = 0
            for i, tid in enumerate(chosen):
                t = tiles[tid]
                win = rasterio.windows.Window(t.col_off, t.row_off, t.width, t.height)
                img = ds.read(1, window=win)
                if (img > 0).mean() < 0.95:        # skip mostly-fill tiles
                    continue
                if az is None:                      # estimate once, reuse
                    az = pr.estimate_solar_azimuth(img)
                    print(f"estimated solar azimuth: {az:.1f} deg")
                props = pr.propose(img, PIX, azimuth_deg=az)
                props = [p for p in props
                         if MIN_D_M <= p.diameter_px * PIX <= MAX_D_M]
                if not props:
                    continue
                # 8-bit render for the detector (display scaling only)
                v = img[img > 0]
                lo, hi = np.percentile(v, [2, 98])
                vis = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
                Image.fromarray((vis * 255).astype(np.uint8)).save(
                    OUT / "images" / split / f"{tid}.png")
                lines = []
                for p in props:
                    x0, y0, x1b, y1b = p.bbox_xyxy
                    cx = min(max((x0 + x1b) / 2 / t.width, 0.0), 1.0)
                    cy = min(max((y0 + y1b) / 2 / t.height, 0.0), 1.0)
                    w = min(max((x1b - x0) / t.width, 0.0), 1.0)
                    h = min(max((y1b - y0) / t.height, 0.0), 1.0)
                    lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
                    prov_rows.append({
                        "tile_id": tid, "split": split,
                        "x_px": f"{p.x_px:.2f}", "y_px": f"{p.y_px:.2f}",
                        "diameter_px": f"{p.diameter_px:.2f}",
                        "diameter_m": f"{p.diameter_px*PIX:.2f}",
                        "score": f"{p.score:.5f}",
                        "source_label_type": p.source_label_type,
                        "review_status": p.review_status,
                        "generator": "crater.proposals.propose",
                        "solar_azimuth_deg": f"{az:.1f}",
                    })
                (OUT / "labels" / split / f"{tid}.txt").write_text("\n".join(lines))
                n_obj += len(lines)
                if (i + 1) % 100 == 0:
                    print(f"  {split}: {i+1}/{len(chosen)} tiles, {n_obj} objects", flush=True)
            totals[split] = n_obj
            print(f"{split}: {n_obj} proposals over "
                  f"{len(list((OUT/'images'/split).glob('*.png')))} tiles")

    (OUT / "dataset.yaml").write_text(
        "# Malapert crater detection dataset\n"
        "# LABELS ARE UNREVIEWED MODEL PROPOSALS, NOT HUMAN GROUND TRUTH.\n"
        "# See reports/robbins_audit.md and docs/annotation_guide.md.\n"
        f"path: {OUT}\n"
        "train: images/train\nval: images/val\ntest: images/test\n"
        "nc: 1\nnames: [crater]\n")

    with open(OUT / "annotation_provenance.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(prov_rows[0].keys()))
        w.writeheader()
        w.writerows(prov_rows)

    (OUT / "PROVENANCE.json").write_text(json.dumps({
        "label_source": "crater.proposals (model-free matched filter)",
        "source_label_type": "model_proposal",
        "review_status": "unreviewed",
        "human_reviewed": False,
        "robbins_craters_in_roi_within_range": 0,
        "warning": ("Metrics against these labels measure agreement with the "
                    "proposal generator, not detection of real craters."),
        "solar_azimuth_deg": az, "seed": SEED,
        "pixel_scale_m": PIX, "tile_px": TILE,
        "objects_per_split": totals,
    }, indent=2))
    print("\nwrote", OUT / "dataset.yaml")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
