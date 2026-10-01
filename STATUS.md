# STATUS

Last update: 2026-10-01
Phase: **0 — environment & feasibility audit. BLOCKED awaiting decisions.**

Status vocabulary: IMPLEMENTED / TESTED / EXECUTED / VERIFIED / BLOCKED.

## Completed
| Item | Status | Evidence |
|---|---|---|
| Host resource audit | VERIFIED | reports/environment_audit.md |
| Egress policy mapping | VERIFIED | reports/environment_audit.md |
| Geospatial stack install | VERIFIED | rasterio 1.4.4 / GDAL 3.10.3 imported |
| Reachable lunar data survey | VERIFIED | reports/data_feasibility_malapert.md |
| Windowed /vsicurl read at Malapert | VERIFIED | reports/evidence/probe_vsicurl.py |
| Visual inspection of candidate imagery | VERIFIED | reports/evidence/malapert_chips.png |

## Blocked
| Item | Status | Cause |
|---|---|---|
| LROC NAC acquisition (sec. 4) | BLOCKED | egress 403 on all LROC/PDS/ODE hosts |
| Robbins catalogue (sec. 6) | BLOCKED | egress 403 on PDS/USGS/Zenodo/Figshare/Kaggle |
| Kaggle dataset + GPU training (sec. 9) | BLOCKED | egress 403 on kaggle.com; no local GPU |
| ISIS processing (sec. 5) | UNVERIFIED | registries reachable but ISIS needs mission
  kernel data from naif.jpl.nasa.gov, which is DENIED |

## Not started
Sections 5, 7, 8, 10, 11 — all depend on the blocked inputs above.

## Next action
Awaiting user decision (see DECISIONS.md, D-001..D-004).
No bulk download, upload, or training has been attempted or will be before
the required approvals.
