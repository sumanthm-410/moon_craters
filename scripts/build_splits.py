#!/usr/bin/env python3
"""Build spatially independent train/val/test regions for the Malapert ROI.

Single-axis bands waste too much of a 21 km ROI to inter-split buffers: at a
defensible 1000 m ejecta buffer the validation band comes out empty at
1024 px tiles and reaches only ~5% at 512 px (DECISIONS.md D-012).

This script uses a 2-D layout instead: validation and test are *corner*
blocks, which are buffered on only two interior sides rather than on both
sides of a full-width band, so far less ground is discarded. Train is the
connected remainder.

Region sizes are then solved so that the **usable** areas (after buffering
and tile assignment) hit the 70/15/15 target, rather than the raw region
areas hitting it while the usable areas miss. Both ratios are reported.

The result is audited with crater.splits.check_leakage, which is an
independent verifier, not the assigner.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from affine import Affine
from shapely.geometry import box

from crater import splits, tiling

# ROI: the controlled Malapert mosaic grid (verified from the GeoTIFF, D-009)
X0, Y1 = -11000.0, 132000.0          # upper-left, projected metres
SIDE_PX = 21000
PIXEL_M = 1.0
X1, Y0 = X0 + SIDE_PX * PIXEL_M, Y1 - SIDE_PX * PIXEL_M
BUFFER_M = 1000.0                     # one crater diameter, ejecta argument (D-008)
TARGET = (0.70, 0.15, 0.15)


def corner_regions(side_m: float) -> list[splits.SplitRegion]:
    """val = lower-left corner block, test = upper-right, train = remainder."""
    val = box(X0, Y0, X0 + side_m, Y0 + side_m)
    test = box(X1 - side_m, Y1 - side_m, X1, Y1)
    train = box(X0, Y0, X1, Y1).difference(val).difference(test)
    return [
        splits.SplitRegion("train", train),
        splits.SplitRegion("val", val),
        splits.SplitRegion("test", test),
    ]


def evaluate(side_m: float, tile_px: int, margin_px: int):
    tr = Affine.translation(X0, Y1) @ Affine.scale(PIXEL_M, -PIXEL_M)
    grid = tiling.TileGrid(tile_px=tile_px, margin_px=margin_px, transform=tr,
                           pixel_scale_m=PIXEL_M, level=0)
    tiles = list(tiling.iter_tiles(grid, width=SIDE_PX, height=SIDE_PX))
    asg = splits.assign_tiles_to_splits(tiles, corner_regions(side_m), buffer_m=BUFFER_M)
    counts: dict[str, int] = {}
    for a in asg.assignments:
        if a.split:
            counts[a.split] = counts.get(a.split, 0) + 1
    total = sum(counts.values())
    frac = {k: v / total for k, v in counts.items()} if total else {}
    return asg, tiles, counts, frac


def main() -> int:
    tile_px, margin_px = 512, 64
    print(f"ROI x[{X0:.0f},{X1:.0f}] y[{Y0:.0f},{Y1:.0f}] = "
          f"{(X1-X0)/1000:.0f} x {(Y1-Y0)/1000:.0f} km, buffer {BUFFER_M:.0f} m ground")
    print(f"fine level: {tile_px} px tiles, {margin_px} px margin, {PIXEL_M} m/px\n")

    print("Solving corner-block side so that USABLE areas hit 15% each:")
    print(f"{'side km':>8} {'train':>7} {'val':>6} {'test':>6} {'usable ratio':>28}")
    best = None
    for side_km in (7.5, 8.0, 8.2, 8.4, 8.6, 8.8, 9.0):
        try:
            _, _, c, f = evaluate(side_km * 1000, tile_px, margin_px)
        except ValueError as exc:
            # The module refuses a layout whose train remainder is no longer one
            # contiguous block -- a real guard, not a failure of this search.
            print(f"{side_km:>8.1f}  rejected: {str(exc)[:80]}")
            continue
        r = (f.get('train', 0), f.get('val', 0), f.get('test', 0))
        print(f"{side_km:>8.1f} {c.get('train',0):>7} {c.get('val',0):>6} {c.get('test',0):>6} "
              f"  {r[0]:.3f}/{r[1]:.3f}/{r[2]:.3f}")
        err = abs(r[1] - 0.15) + abs(r[2] - 0.15)
        if best is None or err < best[0]:
            best = (err, side_km)

    side_km = best[1]
    print(f"\nselected corner side = {side_km:.1f} km")
    asg, tiles, counts, frac = evaluate(side_km * 1000, tile_px, margin_px)
    n_disc = sum(1 for a in asg.assignments if not a.split)
    print(f"tiles total {len(tiles)}, assigned {sum(counts.values())}, discarded {n_disc} "
          f"({100*n_disc/len(tiles):.1f}%)")
    print(f"counts {counts}")
    print(f"usable tile ratio train/val/test = "
          f"{frac.get('train',0):.3f}/{frac.get('val',0):.3f}/{frac.get('test',0):.3f}")

    reasons: dict[str, int] = {}
    for a in asg.assignments:
        if not a.split:
            reasons[a.reason] = reasons.get(a.reason, 0) + 1
    print(f"discard reasons: {reasons}")

    print("\nIndependent leakage audit of the assignment:")
    rep = splits.check_leakage(asg.split_tiles(), buffer_m=BUFFER_M)
    print(f"  tiles audited {rep.n_tiles}, pairs examined {rep.n_pairs_examined}")
    print(f"  split counts {rep.split_tile_counts}")
    print(f"  findings: {len(rep.findings)}")
    for f_ in rep.findings[:5]:
        print("   -", f_)
    print(f"  CLEAN = {len(rep.findings) == 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
