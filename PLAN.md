# PLAN — Malapert Massif crater survey

## 1. Scientific objective (my understanding)
Produce a reproducible, geometrically valid survey of impact craters over a
defined area of Malapert Massif (~86 S), in which:
- imagery is traceable to verified archive products and radiometrically
  calibrated, never raw DN;
- every crater position and diameter is derived through an explicit lunar
  map projection on the Moon 2000 sphere, with the planimetric vs
  terrain-surface distinction stated;
- detections come from a detector trained and evaluated on *spatially
  independent* ground, with recall/precision measured as a function of
  diameter, not just a global mAP;
- the output is a crater size-frequency distribution presented as an R-plot
  with honest Poisson intervals, documented completeness limits, and a
  clearly stated reliable diameter range and usable survey area.
The deliverable is a defensible measurement with quantified uncertainty over
a documented size range — explicitly NOT "every crater".

## 2. Blockers (verified, not speculative)
See reports/environment_audit.md. In short: the environment's network policy
denies every LROC, PDS, ODE, USGS, NAIF and Kaggle host. That removes the
imagery, the ground truth and the training platform simultaneously.
The reachable USGS service bucket holds only global basemaps; the only
sub-10 m product (Kaguya TC, 7.4 m/px) is saturated at Malapert
(median DN 254/255) and shows craters only as shadow silhouettes
(reports/data_feasibility_malapert.md). Rejected for measurement (D-005).

There is no workaround available to me. Routing around a policy denial is
out of scope, and the mirrors that search surfaced (lroc.im-ldi.com,
trek.nasa.gov) are denied as well — and would in any case need provenance
verification before use.

## 3. Options for the user

### Option 1 — Unblock the archives, then run the project as specified
User widens the environment Network access setting, or adds these hosts:
  pds.lroc.asu.edu, wms.lroc.asu.edu, quickmap.lroc.asu.edu,
  ode.rsl.wustl.edu, oderest.rsl.wustl.edu, pds-geosciences.wustl.edu,
  pds-imaging.jpl.nasa.gov, planetarymaps.usgs.gov, astrogeology.usgs.gov,
  naif.jpl.nasa.gov (SPICE kernels, needed for ISIS geometry),
  www.kaggle.com (dataset + GPU training).
This is the only route that supports the full brief at NAC scale.

Pilot ROI recommendation under Option 1:
  centre as resolved in D-002, half-extent 10 km x 10 km (400 km^2).
  At ~1.0-1.5 m/px NAC that is ~1.0e4 x 1.0e4 px per strip-equivalent.
  Diameter range recommendation: **20 m to 1000 m**.
    - floor 20 m  = ~15-20 px across at 1.0-1.5 m/px, enough for a rim fit
      and above the single-pixel noise regime;
    - ceiling 1 km = where Robbins becomes approximately complete, so the
      catalogue series and the detector series overlap and can be compared.
  Craters below ~20 m are detectable but not reliably *measurable*, and
  Robbins does not label them — that is the small-crater label gap and it
  must be closed by local annotation, not assumed away.

Storage estimate (pilot, per acquisition):
  raw NAC EDR            ~0.3-0.8 GB each
  ISIS .cub calibrated   ~2-4x the EDR
  projected strip        ~1-2 GB
  mosaic (10x10 km, 1 m) ~100 Mpx, ~0.1-0.4 GB with compression
  tiles + labels         ~1-3 GB
  Pilot total for 6 candidates: ~25-45 GB.
  Available disk is 30 GiB -> the full 6-strip pilot does NOT fit without
  staged cleanup. This must be managed, not discovered late.

### Option 2 — Degraded demonstration on reachable data
Build the full pipeline and run it end to end on Kaguya TC + LOLA, with the
mosaic and every derived size labelled NON-METRIC / shadow-derived.
Honest assessment: this demonstrates the *software*, not the science. With
saturated imagery, no crater catalogue, and no GPU, it cannot produce a
defensible R-plot. I do not recommend presenting its numbers as results.

### Option 3 — Build and test the data-independent core now
Independent of which data route is chosen, roughly 60% of this project is
pure computation that I can write and test immediately against synthetic
fixtures with known answers:
  - lunar CRS and polar-stereographic round trips, pole and
    longitude-domain edge cases;
  - geodesic rim sampling from catalogue centre+diameter on the sphere;
  - pixel<->world transforms, multiscale coordinate recovery;
  - YOLO box normalisation and clipping, border-object policy;
  - spatial split construction and automated leakage tests;
  - deduplication that does not merge distinct nested craters;
  - usable survey area with projection-distortion weighting;
  - R-plot binning, units, Poisson intervals, zero-count handling;
  - download validation (HTML-error-page detection, checksum, raster
    readability) and stage resume (--dry-run/--resume/--validate-only).
This is the recommended use of the time while Option 1 is arranged.

## 4. Staged implementation plan
S0  environment + feasibility audit                      DONE
S1  data-independent core + tests (Option 3)             ready to start
S2  acquisition + manifest + provenance                  needs D-001
S3  calibration, geometry, mosaicking, QA                needs D-001, D-004
S4  ground truth audit + annotation campaign             needs D-001
S5  tiling, splits, leakage tests                        needs S3, S4
S6  pre-training checkpoint (mandatory stop)             needs S5
S7  Kaggle dataset + GPU training                        needs D-001 + approval
S8  full-ROI inference, dedup, rim measurement           needs S7
S9  R-plot, uncertainty, limitations                     needs S8

## 5. Decisions required
D-001 network egress (blocking), D-002 ROI centre, D-003 diameter range,
D-004 processing branch. See DECISIONS.md.
