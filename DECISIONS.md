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
