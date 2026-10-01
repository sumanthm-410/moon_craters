#!/usr/bin/env python3
"""Build a download-ready crater annotation pack from the Malapert mosaic.

Pack layout (artifacts/annotation_pack/):
  train_tiles/      30 tiles to annotate, from the TRAIN split
  train_prelabels/  YOLO boxes from the trained detector, to CORRECT (optional)
  eval_tiles/       10 tiles from the TEST split -- annotate BLIND, no prelabels
  labels.txt        the single class name: crater
  tile_manifest.csv maps every tile back to mosaic/projected coordinates
  README.txt        how to annotate and what counts as a crater

Tiles are cut at 1 m/px and upsampled 2x to 1024 px (0.5 m per display pixel):
small craters are easier to box, and it matches the scale the detector was
trained at. The manifest records this so labels can be mapped back exactly.
"""
from __future__ import annotations

import csv
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from PIL import Image
from shapely.geometry import box

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from crater import splits, tiling

X0, Y1, N, PIX = -11000.0, 132000.0, 21000, 1.0
X1, Y0 = X0 + N * PIX, Y1 - N * PIX
TILE, MARGIN, BUFFER, SIDE, UP = 512, 64, 1000.0, 8600.0, 2
COG = ROOT / "artifacts" / "malapert_LO1_cog.tif"
WEIGHTS = ROOT / "artifacts" / "runs" / "crater_yolo11n" / "weights" / "best.pt"
OUT = ROOT / "artifacts" / "annotation_pack"
N_TRAIN, N_EVAL, SEED = 30, 10, 20261001
MIN_D_M = 20.0

README = """MALAPERT CRATER ANNOTATION PACK
================================

WHAT TO DO
1. Open https://www.makesense.ai  (free, runs in your browser, no account,
   images never leave your computer).
2. Click "Get Started", drop in ALL images from ONE folder (train_tiles/ first).
3. Choose "Object Detection". When asked for labels, load labels.txt
   (or type the single label: crater).
4. OPTIONAL, train_tiles only: Actions -> Import Annotations -> YOLO,
   select the .txt files from train_prelabels/ plus labels.txt.
   These are MODEL GUESSES. Delete wrong ones, fix sizes, add missed craters.
5. Draw a box around every crater. Zoom with the mouse wheel.
6. Actions -> Export Annotations -> "A .zip package containing files in
   YOLO format". Save it as train_labels.zip.
7. Repeat for eval_tiles/ as eval_labels.zip -- but DO NOT import any
   prelabels for eval tiles. Annotate them from scratch. They are the
   independent test of whether the model works at Malapert, and they are
   only worth anything if the model's guesses never influenced them.
8. Send both zips back (commit them to the repo, or upload them in chat).

WHAT COUNTS AS A CRATER (short version of docs/annotation_guide.md)
- Roughly circular depression with at least two of: a rim arc visible over
  half the circumference; a bright sunlit wall facing a dark shadowed wall;
  a floor lower than the surroundings.
- BOX THE RIM, NOT THE SHADOW. Sunlight here is very low, so shadows are
  long. Draw the box tight to the rim crest, completing the circle through
  the shadow by eye.
- Minimum size: 20 m = 40 display pixels across (images are 0.5 m/pixel).
  Skip anything smaller.
- Nested crater (small inside large): box BOTH.
- Crater cut by the image edge: box the visible part.
- Not craters: boulders and their shadows, isolated shadows with no rim,
  bright spots with no relief, ridges, slope streaks.
- Unsure? Skip it. A missing box costs less than a wrong one.

TIME: roughly 3-6 minutes per tile once you are used to it. Each tile is
512 x 512 m of ground. If time is short, the 10 eval tiles matter most.
"""


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    for d in ("train_tiles", "train_prelabels", "eval_tiles"):
        (OUT / d).mkdir(parents=True)
    (OUT / "labels.txt").write_text("crater\n")
    (OUT / "README.txt").write_text(README)

    val = box(X0, Y0, X0 + SIDE, Y0 + SIDE)
    test = box(X1 - SIDE, Y1 - SIDE, X1, Y1)
    train = box(X0, Y0, X1, Y1).difference(val).difference(test)
    tr = Affine.translation(X0, Y1) @ Affine.scale(PIX, -PIX)
    grid = tiling.TileGrid(tile_px=TILE, margin_px=MARGIN, transform=tr,
                           pixel_scale_m=PIX, level=0)
    tiles = {t.tile_id: t for t in tiling.iter_tiles(grid, width=N, height=N)}
    asg = splits.assign_tiles_to_splits(
        list(tiles.values()),
        [splits.SplitRegion("train", train), splits.SplitRegion("val", val),
         splits.SplitRegion("test", test)], buffer_m=BUFFER)
    pool = {"train": [], "test": []}
    for a in asg.assignments:
        if a.split in pool and not tiles[a.tile_id].partial:
            pool[a.split].append(a.tile_id)
    rng = random.Random(SEED)
    pick = {"train": rng.sample(sorted(pool["train"]), N_TRAIN),
            "test": rng.sample(sorted(pool["test"]), N_EVAL)}

    model = None
    if WEIGHTS.exists():
        from ultralytics import YOLO
        model = YOLO(str(WEIGHTS))

    rows = []
    with rasterio.open(COG) as ds:
        for split, ids in pick.items():
            folder = "train_tiles" if split == "train" else "eval_tiles"
            for tid in ids:
                t = tiles[tid]
                img = ds.read(1, window=rasterio.windows.Window(
                    t.col_off, t.row_off, t.width, t.height))
                v = img[img > 0]
                lo, hi = (np.percentile(v, [1, 99]) if v.size else (0, 1))
                vis = np.clip((img.astype(np.float32) - lo) / max(hi - lo, 1), 0, 1)
                im = Image.fromarray((vis * 255).astype(np.uint8)).resize(
                    (t.width * UP, t.height * UP), Image.LANCZOS)
                name = f"malapert_{tid}.png"
                im.save(OUT / folder / name)
                n_pre = 0
                if split == "train" and model is not None:
                    r = model.predict(np.stack([np.array(im)] * 3, -1), imgsz=1024,
                                      conf=0.25, max_det=1500, device="cpu",
                                      verbose=False)[0]
                    lines = [f"0 {x:.6f} {y:.6f} {w:.6f} {h:.6f}"
                             for x, y, w, h in r.boxes.xywhn.tolist()
                             # project floor is 20 m = 40 display px at 0.5 m/px;
                             # the detector was trained down to 4 m, so drop
                             # everything below the floor rather than make the
                             # reviewer delete hundreds of boxes
                             if max(w, h) * im.width >= MIN_D_M / (PIX / UP)]
                    (OUT / "train_prelabels" / name.replace(".png", ".txt")).write_text(
                        "\n".join(lines))
                    n_pre = len(lines)
                ulx, uly = t.transform.c, t.transform.f
                rows.append({"file": f"{folder}/{name}", "split": split, "tile_id": tid,
                             "col_off": t.col_off, "row_off": t.row_off,
                             "ulx_m": ulx, "uly_m": uly,
                             "source_m_per_px": PIX, "display_m_per_px": PIX / UP,
                             "upsample": UP, "stretch_lo_dn": float(lo),
                             "stretch_hi_dn": float(hi), "prelabels": n_pre,
                             "source": "bdr.nac_roi.malapertlo1 (LO1), controlled mosaic"})
                print(f"{split:5} {tid}  prelabels={n_pre}")
    with open(OUT / "tile_manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    z = shutil.make_archive(str(ROOT / "artifacts" / "malapert_annotation_pack"), "zip", OUT)
    print("wrote", z, f"{Path(z).stat().st_size/2**20:.1f} MiB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
