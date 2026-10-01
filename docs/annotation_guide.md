# Crater annotation guide — Malapert Massif survey

Version 1.0, 2026-10-01. Binding for every human-reviewed label in this
project. Approved diameter range: **20 m to 1000 m** (DECISIONS.md D-003).

Every annotation carries `source_label_type` and `review_status`:

| `source_label_type` | meaning |
|---|---|
| `human` | drawn by a human annotator against imagery |
| `catalogue` | derived from Robbins centre+diameter, geometry approximate |
| `model_proposal` | detector output offered as a suggestion only |

| `review_status` | meaning |
|---|---|
| `unreviewed` | not yet seen by a human |
| `accepted` | a human confirmed the feature and its extent |
| `corrected` | a human adjusted the geometry |
| `rejected` | a human judged it not a crater, or not measurable |

**A `model_proposal` never becomes `human` by being accepted.** It becomes
`source_label_type=model_proposal, review_status=accepted`. Relabelling
pseudo-labels as human ground truth is forbidden and is the single most
damaging error available in this project, because it makes the evaluation
set agree with the model by construction.

The independent evaluation subset must contain **only** labels that were
`human` from the outset, drawn without the detector's output visible.

## 1. What counts as a crater
Annotate an approximately circular depression showing at least TWO of:
1. a raised or clearly defined rim arc over >= 50% of the circumference;
2. an illumination pair — a bright rim/wall facing the Sun and a shadowed
   interior wall opposite, consistent with the scene's solar azimuth;
3. a distinct floor at lower elevation than surrounding terrain.

Do not annotate: albedo spots with no relief, boulders and their shadows,
isolated shadows with no visible rim, lineaments, or scarps.

## 2. The rim is the measurement, not the shadow
The diameter is the **rim crest to rim crest** distance.
A filled dark region is NOT the rim. At the high solar incidence typical of
86 S, the shadow fills only part of the crater and its extent changes with
illumination, so a shadow-derived diameter is biased and
illumination-dependent. Annotators place the rim where the slope breaks,
using the lit rim arc, and complete the circle through the shadow by
inference, flagging the crater `shadow_flag=1`.

If the rim cannot be located on at least half the circumference, mark the
crater `review_status=rejected` with reason `rim_not_locatable`, or retain it
as a detection with `quality_flag=unmeasured`. Do not guess a diameter.

## 3. Geometry recorded
Primary geometry is an **ellipse**: centre, major axis, minor axis, azimuth
of the major axis. For a well-resolved circular crater major ~ minor.
Reported sizes:
- `major_diameter_m`, `minor_diameter_m`;
- `equivalent_diameter_m = sqrt(major * minor)` — stated explicitly as the
  geometric mean of the fitted axes;
- all are **planimetric** great-circle distances on the Moon 2000 sphere, not
  terrain-surface distances (see `geometry.DIAMETER_DISTANCE_KIND`). On a
  steep massif flank the true surface distance is longer; this project does
  not correct for it and must not claim to.

## 4. Ambiguous and special cases
- **Partial / clipped craters** (crossing a tile or survey edge): annotate the
  visible arc, set `edge_flag=1` and record the visible fraction. A clipped
  crater may be used for detection training but is **excluded from the
  size-frequency count** unless its full rim is recoverable, because its
  diameter is not measurable. The counting edge rule is in `area.py`.
- **Nested craters**: a smaller crater inside a larger one is a SEPARATE
  entry. Both are annotated. The deduplicator is explicitly required not to
  merge them.
- **Overlapping / doublet craters**: annotate each separately where two rims
  are distinguishable; if the rims are indistinguishable, annotate one
  feature and flag `ambiguous_multiplicity`.
- **Degraded craters**: annotate if a rim is still traceable; flag
  `degraded=1`. Heavily subdued circular depressions with no rim break are
  rejected as `rim_not_locatable`.
- **Secondary craters and chains**: annotate individually, flag
  `likely_secondary=1` where they form an obvious cluster or chain. They are
  counted separately in the size-frequency analysis because including
  secondaries changes the slope.
- **Permanently shadowed regions**: no annotation is possible. The area is
  excluded from the survey mask, not annotated as empty. **Unannotated
  terrain is not verified background** — an unlabelled region must be
  excluded from the counting area, never treated as crater-free.

## 5. Size limits
- Below 20 m: do not annotate. Not reliably measurable at the planned scale.
- Above 1000 m: annotate, but these are outside the primary analysis range
  and are flagged `above_range=1`.
- A crater must span at least 8 pixels for a rim fit to be attempted; below
  that record `quality_flag=unmeasured`.

## 6. Reviewer agreement
At least 10% of tiles are annotated independently by two reviewers. Report
inter-annotator agreement as: detection agreement (IoU-matched) and the
distribution of diameter differences. **The spread of that diameter
difference is the annotation component of the measurement uncertainty budget**
and is reported in the uncertainty analysis. It is not optional.
