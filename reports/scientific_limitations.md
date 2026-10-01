# Scientific limitations

Living document. Version 0.1, 2026-10-01 — written at project start so that
limitations are stated before results exist, not retrofitted to them.
Items marked [OPEN] cannot be resolved until the pipeline runs on real data.

## 1. Limitations already established (verified, not anticipated)

**L-1. No archive access in the current environment.** Every LROC, PDS, ODE,
USGS and NAIF host is denied by network policy. No imagery, no SPICE kernels,
no crater catalogue. Until this changes, nothing in sections 4-11 of the brief
can be executed, and no result from this project may be described as measured.

**L-2. No ISIS, and no way to install it here.** No container daemon, no
conda. If branch B (EDR -> ISIS -> terrain-aware projection) is required for
defensible geometry, it must run in a different environment. Branch A (reusing
map-projected products) avoids this but inherits whatever geometry provenance
those products carry, which must then be verified rather than assumed.

**L-3. The only reachable sub-10 m product is unusable at this site.** The
Kaguya TC ortho mosaic (7.4 m/px) covers Malapert but is saturated there
(median DN 254/255 at both candidate centres; verified numerically and by
visual inspection, `reports/data_feasibility_malapert.md`). Craters appear
only as shadow silhouettes. A shadow outline is not a rim, so this product
cannot support diameter measurement, only qualitative detection.

**L-4. The ROI centre is not yet established.** The brief gives
85.9 S, 2.9 E; literature places the Malapert massif near 85.8 S, 356.4 E,
about 14 km away. This is unresolved (D-002) and no tiling or download may
proceed until it is.

## 2. Limitations intrinsic to the science (will persist even after L-1)

**L-5. The small-crater label gap.** The Robbins catalogue is approximately
complete only above ~1-2 km. The approved target range is 20 m - 1000 m, so
across almost the whole range the catalogue is NOT ground truth. Consequences
that must be respected throughout:
- unlabelled small craters are **not** verified background and must not be
  used as negative training evidence;
- agreement between the detector and the catalogue below ~1 km is not
  validation, it is a measure of shared incompleteness;
- an independent, human-annotated evaluation subset is mandatory, and it must
  never contain accepted model proposals (see `docs/annotation_guide.md`).

**L-6. Illumination at 86 S.** Near-grazing solar incidence means deep
shadows, saturated sunward slopes, and a detection probability that depends on
crater orientation relative to the Sun as well as on size. Recall must
therefore be reported per diameter bin AND per illumination/terrain category;
a single global recall number would conceal this.

**L-7. Planimetric, not terrain-surface, distances.** Every diameter this
project reports is a great-circle distance on the Moon 2000 sphere
(`geometry.DIAMETER_DISTANCE_KIND = "planimetric_great_circle"`). On the steep
flanks of a massif the true surface distance is longer. No slope correction is
applied and none may be claimed.

**L-8. Projection distortion is small but not zero.** The south polar
stereographic scale factor is ~1.0012 at 85.9 S, so an uncorrected length is
wrong by ~0.12% and an uncorrected area by ~0.24%. The code divides by k and
k^2 respectively. This is below other error sources here but is corrected
because it is free to correct and grows away from the pole.

**L-9. Bounding-box diameters are a proxy.** A detector box is not a rim fit.
Box-derived diameters are labelled as proxies and their bias against known
circular geometry is measured, not assumed. Rim/ellipse refinement is the
preferred measurement; where it fails, the crater is retained as a flagged
proxy or marked unmeasured — never given a fabricated size.

**L-10. Detector confidence is not measurement certainty.** A high-confidence
detection may still have a poorly localised rim. The two are reported
separately and never conflated.

**L-11. Completeness is empirical, not assumed.** Detection completeness is
estimated from reviewed holdouts. Injection/recovery, if used, is a documented
supplementary experiment. Raw and any corrected counts are shown separately,
and no completeness correction is invented.

**L-12. Counting statistics vs systematics.** Poisson counting uncertainty and
model/measurement systematic uncertainty are propagated and reported
separately, never summed into one opaque error bar.

**L-13. Terrain correction of the source mosaic is unverified.** The label
says "controlled mosaic ... in PolarStereographic projection" and cites
[HENRIKSENETAL2023], but does not state whether it is orthorectified against
a DEM. Map projection is not terrain correction; on steep massif flanks an
uncorrected projection displaces features by about h*tan(emission). Until
resolved, only planimetric diameters are reported and no terrain correction
is claimed.

**L-14. The source is DN, not calibrated reflectance.** processing_level is
"Derived" with no radiometric scaling in the label. Detection and geometry
are supported; no photometric or albedo quantity may be derived.

**L-15. The survey is 441 km^2, so the large-diameter bins are thin.** At
equilibrium density the >1 km bins hold only tens of craters. Those bins will
carry wide Poisson intervals and must be reported as underpowered rather than
interpreted.

## 3. [OPEN] — resolvable only by running the pipeline
- Achievable recall and precision as a function of diameter.
- The true minimum reliably *measurable* diameter, as opposed to the 20 m
  target, which is a design intent and not yet a demonstrated capability.
- The usable survey area after nodata, shadow and seam exclusions.
- Registration residuals between acquisitions, and whether they meet the
  tolerance set by the minimum crater size.
- Whether the crater count and usable area are sufficient for an informative
  R-plot, or whether the comparison is underpowered and must be reported as a
  count table instead.

## 4. Statements this project will NOT make
- That it detected every crater.
- That agreement with Robbins below ~1 km validates the detector.
- That Robbins is complete at small diameters.
- A surface age, without a separately justified chronology analysis.
- A diameter derived from shadow extent presented as a rim measurement.
