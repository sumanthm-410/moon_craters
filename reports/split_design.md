# Spatial split design — Malapert ROI

Executed 2026-10-01 with `scripts/build_splits.py`. Numbers below are from
that run on the real mosaic grid, not estimates.

## Problem
A defensible inter-split buffer is of order one crater diameter, because the
continuous ejecta blanket extends roughly that far; for the approved
20-1000 m range that is **1000 m** (D-008). On a 21 km ROI, single-axis
bands spend too much of the frame on buffers:

| fine tile | train/val/test | discarded | usable ratio |
|---|---|---|---|
| 1024 px | 504 / **0** / 56 | 29.3% | 0.896 / **0.000** / 0.104 |
| 512 px | 1925 / 110 / 275 | 23.8% | 0.831 / 0.048 / 0.120 |

A 15% band of a 21 km ROI is 3.15 km tall; after 1 km of buffer on each side
only ~1.15 km remains, which barely holds one tile row.

## Solution: 2-D corner blocks
Validation and test are placed as **corner blocks** rather than full-width
bands. A corner block is buffered on only its two interior sides (its other
two sides are the ROI edge), so far less ground is discarded. Train is the
connected remainder.

Block side was solved so the **usable** areas hit the target, rather than the
raw region areas hitting it while the usable areas miss:

| corner side | train | val | test | usable ratio |
|---|---|---|---|---|
| 7.5 km | 1969 | 272 | 272 | 0.784 / 0.108 / 0.108 |
| 8.0 km | 1831 | 306 | 306 | 0.749 / 0.125 / 0.125 |
| 8.4 km | 1729 | 342 | 342 | 0.717 / 0.142 / 0.142 |
| **8.6 km** | **1677** | **380** | **380** | **0.688 / 0.156 / 0.156** |
| 8.8 km | 1625 | 380 | 380 | 0.681 / 0.159 / 0.159 |
| 9.0 km | 1571 | 420 | 420 | 0.652 / 0.174 / 0.174 |

**Selected: 8.6 km corner blocks**, giving 0.688 / 0.156 / 0.156 — effectively
the 70/15/15 target, reached honestly rather than by forcing region sizes.

Final assignment (512 px tiles, 64 px margin, 1 m/px, 1000 m ground buffer):
- 3025 tiles enumerated, **2437 assigned**, 588 discarded (19.4%)
- train 1677, val 380, test 380
- discard reasons: `within_buffer_of_other_split` 452,
  `straddles_region_boundary` 136

Validation went from 110 tiles (bands) to **380** (corner blocks), and the
discarded fraction fell from 23.8% to 19.4%.

## The layout search is constrained, not free
At an 11 km corner side the module **refused** the layout:

> split region 'train' is a MultiPolygon with 2 parts; a split must be ONE
> contiguous geographic block.

Two 11 km blocks in opposite corners of a 21 km square pinch the remainder
into two pieces. The contiguity guard caught this rather than silently
accepting a disconnected (effectively random) split.

## The leakage audit is not vacuous — proven
`check_leakage` reports the final assignment **CLEAN (0 findings)**. That
statement is only worth something if the audit can fail, so it was tested:

| test | findings |
|---|---|
| final assignment at its design buffer (1000 m) | 0 |
| same assignment audited at 3000 m | **2586** (`cross_split_footprint_proximity`) |
| final assignment with ONE train tile flipped to val | **15** |

The audit therefore detects both a tightened criterion and a single
misassigned tile. (It reports 0 pairs examined at 1000-2000 m because
discarding whole tiles quantises the real gap to the 384 m tile stride, so
surviving cross-split tiles sit well beyond 2 km apart — comfortably outside
the 1 km requirement.)

## Still to apply
- The **coarse** pyramid level (4 m/px) must be assigned through the SAME
  `assign_tiles_to_splits` call as the fine level, or the same ground at two
  scales can land in different splits (INTERFACES.md note 10).
- All three illumination variants (LO1, LOA, LOH) of a given ground block
  must share a split, or the same terrain appears in train and test under
  different lighting (D-013).
- Tiles must be filtered by the valid-data mask (DN > 1) before counting:
  these numbers are geometric and do not yet exclude unimaged ground.
