# DECISIONS

Each entry: status, date, rationale, evidence. OPEN items need the user.

## D-001 — Network egress for planetary archives — **DECIDED 2026-10-01**
User will widen the environment Network access / add the denied hosts.
Until that lands, work proceeds on the data-independent core (PLAN.md Option 3).
No stage that needs the archives will be marked EXECUTED on simulated data.

### original finding
The environment denies every LROC, PDS, ODE, USGS, NAIF and Kaggle host
(VERIFIED, reports/environment_audit.md). Nothing in sections 4-11 can run
until this changes. Options presented to user; awaiting choice.

## D-002 — ROI centre and extent — **DEFERRED by user 2026-10-01**
User chose to resolve the centre authoritatively rather than guess between the
two candidates. Action once egress opens: resolve against the IAU gazetteer
and LROC footprint metadata, then set roi.centre in config/project.yaml.
No tiling or download may start while this is OPEN.

### original finding
The brief states Malapert Massif at 85.9 S, 2.9 E. Literature surfaced in
search places the Malapert massif / Artemis candidate region near
85.8 S, 356.4 E (= -3.6 E). These differ by ~6.5 deg longitude, which at
86 S is roughly 14 km on the ground — a materially different site.
Both were probed; both contain data. NOT RESOLVED — needs user confirmation
or an authoritative gazetteer lookup (IAU gazetteer host currently denied).
Recommendation pending D-001.

## D-003 — Target diameter range — **DECIDED 2026-10-01: 20 m to 1000 m**
Approved by user. Rationale: 20 m is ~15-20 px across at 1.0-1.5 m/px NAC,
enough for a rim fit and above the pixel-noise regime; 1000 m is where the
Robbins catalogue becomes approximately complete, so the catalogue series and
the detector series overlap and can be compared on the R-plot.
Consequence (accepted): Robbins does not adequately label 20 m - 1 km craters,
so a reviewed local annotation campaign is REQUIRED. This is the small-crater
label gap of section 6 and it is not assumed away.

## D-004 — Processing branch (A map-projected / B EDR+ISIS / C CDR) — **OPEN, with a new hard constraint**
Branch B (EDR + ISIS) is the scientifically preferred route. Two independent
blockers were VERIFIED on 2026-10-01:
  1. SPICE kernels from naif.jpl.nasa.gov are denied by egress (403), and
     ISIS cannot initialise camera geometry without them.
  2. There is NO container runtime in this environment: the `docker` client
     binary is present but the daemon is absent
     (`dial unix /var/run/docker.sock: no such file or directory`), and there
     is no conda/mamba/apptainer. ISIS therefore cannot be installed or run
     here by any available mechanism.
Consequence: branch B requires BOTH the egress change AND a container
runtime (or a conda-based ISIS install). If a runtime cannot be provided,
the project is restricted to branch A — reusing already map-projected NAC
products (e.g. controlled NAC mosaics / RDRs) whose provenance, projection,
calibration and terrain correction must each be verified before reuse.
This is a decision the user must make once egress is open; it materially
changes the geometry provenance we can claim.

## D-005 — Kaguya TC rejected as a measurement source — **DECIDED (reject)**
Date 2026-10-01. The 7.4 m/px Kaguya TC ortho mosaic is reachable and does
cover Malapert, but median DN = 254/255 at both candidate centres: the lit
surface is saturated and craters appear only as shadow silhouettes
(VERIFIED by numeric stats and visual inspection).
Rationale: a shadow outline is not a rim. Using it for diameters would
produce a systematically biased, illumination-dependent measurement while
appearing quantitative. Rejected for measurement.
It MAY still serve as a qualitative detection demonstration — see PLAN.md
Option 2, clearly labelled non-metric.

## D-006 — No Earth CRS on lunar data — **DECIDED (standing rule)**
All lunar products use the Moon 2000 sphere, R = 1737400 m. Any south-polar
work uses a lunar south polar stereographic CRS defined on that sphere.
No EPSG Earth code will be assigned at any stage.


## D-007 — Tile scale and pyramid design — **OPEN, blocking S5**
Measured 2026-10-01 with `crater.tiling.recommend_tile_size` at 1 m/px,
85.9 S, for the approved 20-1000 m range:
- a 1000 m crater spans 1001.3 px (k inflates it), so with 25% context the
  tile must be 2504 px; 4096 px recommended;
- a 1024 px tile covers the full range only from pyramid level 2 (4 m/px),
  where a 20 m crater is 5.0 px, below the project's 8 px floor.
One tile size at one scale therefore cannot serve the approved range.
Options: (a) two-scale pyramid, fine level 1 m/px for small craters and a
coarse level for large ones; (b) single scale at 1 m/px with 2560-4096 px
tiles; (c) single scale at 2 m/px with 1024 px tiles, accepting 20 m = 10 px.
Note for (b): a detector resizes tiles to its `imgsz`, so a 4096 px tile fed
at imgsz 1024 is effectively 4 m/px and gains nothing for small craters.

## D-008 — Split buffer and achievable ratio — **OPEN, blocking S5**
A defensible inter-split buffer is of order one crater diameter, since the
continuous ejecta blanket extends roughly that far; for this diameter range
that is **1000 m**. Measured consequence (1024 px tiles, 1 m/px, axis y):
| ROI side | train/val/test tiles | discarded | achieved area ratio |
|---|---|---|---|
| 20 km | 459 / 0 / 54 | 31.0% | 0.888 / 0.000 / 0.112  (val EMPTY) |
| 30 km | 1040 / 80 / 160 | 20.6% | 0.806 / 0.065 / 0.130 |
| 35 km | 1426 / 138 / 230 | 15.4% | 0.792 / 0.078 / 0.130 |
| 40 km | 1908 / 159 / 318 | 15.4% | 0.795 / 0.068 / 0.137 |
70/15/15 is not achievable at any tested extent with a 1 km buffer and
single-axis bands. The planner reports this and does not resize regions.
Its own output passes `check_leakage` clean at every extent.

## D-009 — Georeferencing is taken from the raster, not the PDS4 label — **DECIDED 2026-10-01**
The PDS4 label of NAC_POLE_P860S0337 gives `upperleft_corner_x = -40627.5 m`;
the delivered GeoTIFF geotransform and the project's independent analytic
projection both give +40627.5 m (y agrees to 0.4 m). The label sign is wrong.
All georeferencing is therefore read from the delivered raster. Trusting the
label would have mirrored the entire survey about the x axis.

## D-010 — Branch A source product — **DECIDED 2026-10-01 (pending calibration checks)**
LROC RDR BDRNPL "NAC Polar" south mosaics at 1 m/pixel, polar stereographic
on the Moon 2000 sphere, ASU-produced. Chosen because branch B is impossible
here (no container runtime, D-004). Projection VERIFIED. Calibration state
and terrain-correction status are NOT yet verified and must be before any
measurement is published.
