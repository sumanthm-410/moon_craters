# STATUS

Last update: 2026-10-01
Phase: **1 — core implemented and tested; branch-A source verified.
Awaiting decisions on ROI extent, tile design and split ratio.**

Status vocabulary: IMPLEMENTED / TESTED / EXECUTED / VERIFIED / BLOCKED.

## Test suite
**631 tests passing** across 11 modules (run by the lead, not self-reported):

| Module(s) | Owner | Tests |
|---|---|---|
| body, geometry | lead | 61 |
| tiling, splits | A | 105 |
| boxes, dedup | B | 164 |
| area, sfd | C | 98 |
| download, manifest, stages | D | 203 |

## Completed and verified
| Item | Status | Evidence |
|---|---|---|
| Host/toolchain audit | VERIFIED | reports/environment_audit.md |
| Network egress now OPEN | VERIFIED | all PDS/LROC/USGS/NAIF/Kaggle hosts reachable |
| Lunar geometry core, analytic vs PROJ | VERIFIED | 61 tests, agreement ~1e-10 m |
| All 6 candidate product IDs resolve | VERIFIED | ODE, correct `nac.<id>` lowercase form |
| M132795356LC is a **CDR**, not an EDR | VERIFIED | ODE `pt=CDRNAC4` |
| Branch-A source identified and projection verified | VERIFIED | reports/branch_a_product_verification.md |
| PDS4 label x-corner sign defect found | VERIFIED | raster geotransform vs label |
| Pilot patches at 1 m/px, both ROI candidates | VERIFIED | reports/evidence/pilot_nac_patches.png |
| Windowed HTTP reads avoid 10.6 GiB download | VERIFIED | Accept-Ranges honoured by NASA PDS store |

## Key quantitative findings
- **Tile design**: at 1 m/px and 85.9 S a 1000 m crater spans 1001.3 px. With
  25% context it needs a **2504 px** tile (4096 recommended). A 1024 px tile
  only covers the full 20-1000 m range from pyramid level 2 (4 m/px), where a
  20 m crater is 5.0 px — below the project's 8 px floor. **One tile size at
  one scale cannot serve 20 m - 1000 m.** Decision required.
- **Split feasibility** (1024 px tiles, 1 m/px, 1000 m ground buffer, 70/15/15
  along y): 20 km ROI leaves the **validation split empty**; 30 km gives
  80.6/6.5/13.0 with 20.6% discarded; 35 km gives 79.2/7.8/13.0 with 15.4%
  discarded. The planner reports infeasibility and does **not** force the
  target.
- **Storage**: windowed ROI extraction at 1 m/px — 20 km = 381 MiB,
  30 km = 858 MiB, 40 km = 1.49 GiB. Comfortably inside the 30 GiB budget.

## Blocked / awaiting decision
| Item | Needs |
|---|---|
| ROI centre (D-002) | user; empirical evidence now favours candidate A |
| ROI extent | user; >= 30 km needed for a 3-way split |
| Tile design (D-007) | user; single large tile vs two-scale pyramid |
| Split ratio (D-008) | user; 70/15/15 is not achievable, ~79/8/13 is |
| Terrain correction status of the BDR mosaic | verification against ASU/PDS documentation |

## Not started
S4 ground truth audit, S6 pre-training checkpoint, S7 Kaggle training,
S8 inference, S9 R-plot. No download, upload or training has been run.
