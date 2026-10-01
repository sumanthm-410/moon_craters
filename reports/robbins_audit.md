# Robbins catalogue audit — Malapert ROI

Written by the lead on 2026-10-01 from the data produced by the catalogue
agent. Every number below comes from `data/manifests/robbins_roi.csv` or from
the figures in `reports/evidence/`.

## Source
Robbins (2018) lunar crater database, obtained from the **USGS Astrogeology
CKAN repository** (an authoritative host):

```
https://astrogeology.usgs.gov/ckan/dataset/f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/
  resource/c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/
  lunar_crater_database_robbins_2018
```
Accessed 2026-10-01T07:52:45Z. Archive 96,227,201 bytes, stored under
`data/raw/robbins/` (git-ignored). No Kaggle mirror was used or trusted.

## THE HEADLINE RESULT
Inside the 441 km² ROI the catalogue contains **16 craters**.
Their diameters, in metres:

```
1008, 1025, 1067, 1104, 1122, 1124, 1271, 1314,
1342, 1361, 1387, 1501, 1696, 2522, 3288, 4748
```

| Band | Count |
|---|---|
| within the approved 20–1000 m range | **0** |
| above 1000 m | 16 |

**Not one catalogue crater falls inside the project's approved diameter
range.** The smallest entry anywhere in the ROI is 1008 m, i.e. just outside
the top of the range.

## What this settles
1. **Robbins cannot supply training labels for this project.** Not "sparse",
   not "incomplete" — zero, in the target range.
2. **Robbins cannot supply a comparison series either.** Sixteen craters over
   441 km², all above the range, cannot populate an informative R-plot. Any
   catalogue series here would be a handful of bins with counts of order one
   and Poisson intervals spanning decades.
3. **Narrowing the diameter range is no longer a viable escape.** Section 6
   option B (a narrower, defensible range) would mean counting only craters
   above ~1 km, where this ROI holds 16 objects. That is not enough to train
   a detector or to make a measurement. Option B is withdrawn as a
   recommendation.
4. **Unlabelled ground is emphatically not verified background.** Training a
   detector on this ROI with catalogue labels and treating everything else as
   negative would teach it that essentially every visible crater is
   background.

## The visual evidence
`reports/evidence/robbins_gap_zoom150.png` shows three 150 × 150 m patches at
1 m/px with a 20 m scale bar and a 25 m grid, at (−2200, 113000),
(1200, 123200) and (8000, 130000) m, each with 100% valid data.

Inspected directly: every patch contains **dozens** of unambiguous craters
whose diameters are at or above the 20 m bar, with clear rim-and-shadow
pairs and, in several cases, raised rims. The Robbins catalogue contains
**0** craters in these patches.

This is the small-crater label gap, and at this ROI it is total rather than
partial.

## Consequence for the project
Of the three options in section 6 of the brief:
- **A. a reviewed small-crater annotation campaign** — viable, and now the
  only route to a detector covering 20–1000 m.
- **B. a narrower, defensible diameter range** — **withdrawn**, see above.
- **C. external-data pretraining, then local annotation and fine-tuning** —
  viable, and reduces (but does not remove) the local annotation burden.

Both surviving options require human-reviewed local labels. No model proposal
may be promoted to `source_label_type = human`
(`docs/annotation_guide.md`), and the independent evaluation subset must
consist only of labels that were human from the outset.

## Status of the catalogue module
`src/crater/catalogue.py` with **42 passing tests** (part of the 673-test
suite). It loads the catalogue with explicit dtypes, normalises longitude to
the project convention, converts diameters to metres, and selects by
**projected** bounding box rather than a planar degree box. Catalogue-derived
rim geometry is emitted with `source_label_type = "catalogue"` and
`review_status = "unreviewed"`, and is labelled an approximation.

## Not completed
The owning agent had not written its own narrative audit or its
rim-vs-imagery positional-offset estimate when this was compiled. The
`rim_arc_fraction` column in the ROI extract (0.77–0.81 on the sampled rows)
suggests a partial rim-trace quality measure, but its definition has not been
verified by the lead and should not be quoted until it is.
