# STATUS

Last update: 2026-10-01
Phase: **2 — data products built and verified. Blocked on annotation (human
time) and Kaggle authentication before any training.**

Vocabulary: IMPLEMENTED / TESTED / EXECUTED / VERIFIED / BLOCKED.

## Test suite
**631 tests passing** across 11 modules, run by the lead (not self-reported).
A 12th module (`catalogue`) is in flight with its owning agent.

## EXECUTED and VERIFIED
| Item | Evidence |
|---|---|
| Environment + egress audit | reports/environment_audit.md |
| Lunar geometry core, analytic vs PROJ to ~1e-10 m | 61 tests |
| Source selected: ASU **controlled** Malapert mosaic, 1 m/px, v2.0 (2024) | reports/malapert_roi_source.md |
| Projection verified incl. k0 = 1 from the label | reports/georeferencing_and_calibration.md |
| PDS4 `upperleft_corner_x` sign defect found, systematic across 2 product families | same |
| Browse TIF proven a linear 8-bit rescale (r = 0.9999), not a display stretch | same |
| 3 variants downloaded, MD5-verified against their PDS4 labels | data/manifests/manifest.csv |
| Raw memmap read bit-identical to GDAL PDS4 driver | scripts/build_survey.py |
| Survey area 439.918 km² true (naive +0.2459%, predicted +0.244%) | artifacts/survey_area.json |
| Counting areas A_i, 439.081 km² (D=20 m) to 399.019 km² (D=1 km) | same |
| Spatial splits 0.688/0.156/0.156, leakage audit CLEAN and proven non-vacuous | reports/split_design.md |
| Georeferenced COG + survey mask exported | artifacts/ |

## Deliverables on disk
| Path | What |
|---|---|
| `artifacts/malapert_LO1_cog.tif` | primary mosaic, uint16, tiled, overviews, verified CRS |
| `artifacts/survey_mask.tif` | per-pixel count of valid illumination variants |
| `artifacts/survey_area.json` | true surface area + A_i by diameter + edge rule |
| `data/manifests/manifest.csv` | provenance, 3 accepted, MD5 from PDS4 labels |
| `reports/evidence/survey_qa.png` | mosaic / coverage / splits / native crop |
| `REPRODUCE.md` | commands, pitfalls, verified numbers to check against |

## BLOCKED
| Item | Blocker |
|---|---|
| Annotation campaign | **human labelling time** — the catalogue cannot substitute |
| Kaggle dataset + GPU training | `KAGGLE_API_TOKEN` absent in this container (D-018) |
| Inference, dedup, rim measurement | needs a Malapert-validated model |
| Detector R-plot | needs the above |

## Detector TRAINED (external domain)
YOLO11n trained on the CraterDANet LRO-NAC dataset (real human labels,
GPL-3.0). Observation-disjoint test split: **P 0.647, R 0.608, mAP50 0.578,
mAP50-95 0.194**. These are Chang'E-4-region metrics at 0.5 m/px and are NOT
Malapert performance; Malapert remains UNVALIDATED.
See reports/training_evaluation.md.

`crater.dedup`, `crater.boxes`, `crater.sfd` and `crater.area` are
IMPLEMENTED and TESTED but have only ever run on synthetic fixtures with
known answers — never on real detections. That distinction is deliberate.

## In flight
Robbins catalogue audit (agent). Preliminary extract contains **16 craters**
in the whole 441 km² ROI, the sampled ones at ~1000 m and flagged
`above_range=1`. If that holds, the catalogue provides effectively no usable
ground truth for a 20 m–1 km survey, and a reviewed local annotation campaign
is mandatory rather than optional.
