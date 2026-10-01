# Robbins lunar crater catalogue — source resolution, schema, and ROI audit

Owner E, 2026-10-01. Everything below was measured on the delivered product and
the real imagery. Nothing is quoted from memory, and no number here is an
estimate unless it says so.

**Headline.** Inside this project's ROI the Robbins catalogue contains
**16 craters, the smallest of which is 1008 m across.** The approved target
range is 20–1000 m (D-003). **The catalogue's contribution to that range is
exactly zero craters.** The archived release is hard-truncated at D = 1.000 km
— verified over all 1 296 796 records, the global minimum of `DIAM_CIRC_IMG`
is exactly 1.0 and there is not one record below it anywhere on the Moon.
A single 150 × 150 m NAC patch inside the ROI shows of order five craters in
the 20–200 m range; the ROI is 441 km², so the unlabelled small-crater
population is of order **10⁵ craters**, against 16 in the catalogue.

---

## 1. Source resolution

### 1.1 What was resolved

| | |
|---|---|
| Product | **Moon Crater Database v1 Robbins**, Astropedia `lunar_crater_database_robbins_2018` |
| PDS4 bundle LID | `urn:nasa:pds:robbins_lunar_crater_database_2018`, version_id 1.0 |
| Publisher | **USGS Astrogeology Science Center** (host of the PDS Cartography and Imaging Sciences Node) |
| Publication date | 2018-08-15 (README inside the bundle is dated 9 April 2019) |
| Citation / DOI | Robbins, S. J. (2019), *JGR Planets* **124**(4), 871–892, [doi:10.1029/2018JE005592](https://doi.org/10.1029/2018JE005592) — the DOI is carried in the PDS4 label's `<Citation_Information>` |
| Landing page | `https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/lunar_crater_database_robbins_2018` |
| **Download URL used** | `https://astrogeology.usgs.gov/ckan/dataset/f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/lunar_crater_database_robbins_2018` |
| File name on disk | `lunar_crater_database_robbins_2018.zip` |
| Format | ZIP containing a complete **PDS4 bundle** (data + document + xml_schema collections) |
| Size | **96 227 201 bytes** (91.77 MiB), exactly the `content-length` the server advertised and the "91.77 MB" the landing page states |
| Server headers | HTTP/2 200, `content-type: application/zip`, `accept-ranges: bytes`, `last-modified: Wed, 16 Aug 2023 17:53:37 GMT` |
| **Access date** | **2026-10-01, 07:36:53 UTC** |
| sha256 (measured) | `694137e141138892555d0f976446875331d3317ea506ddf055d91b917ca307ab` |
| md5 (measured) | `0d4a8667e0a92d4c1962e4112b2002b0` |
| Licence | US Government work, USGS Astrogeology; no access restriction on the page |

Provenance sidecar: `data/raw/robbins/provenance.json` (written by the fetch
script, contains the full member listing).

### 1.2 Why this URL and not another

* The USGS Astropedia page is the archiving institution's own landing page and
  is the one the journal-linked record points to. It is not a mirror.
* The page's historical "Source Online Linkage" to the PDS Imaging Node annex,
  `https://pdsimage2.wr.usgs.gov/Individual_Investigations/moon_lro.kaguya_multi_craterdatabase_robbins_2018/`,
  is **dead**: that host now serves a single-page-app index and returns
  **404** for the path (verified 2026-10-01).
* The migrated PDS cloud IMG bucket `https://pds.mcp.nasa.gov/data/store/img/`
  was walked with S3 prefix listings; it contains only `THEMIS/` and
  `lunar_reconnaissance_orbiter/pds4/lroc/`. **No annex investigation
  directory exists there**, so the catalogue is not currently served from the
  PDS cloud.
* `https://planetarydata.jpl.nasa.gov/img/data/individual_investigation/`
  contains only two unrelated investigations.
* **No Kaggle or Hugging Face copy was used or trusted.** Several exist; none
  was consulted, and no Kaggle handle was assumed valid.

### 1.3 Integrity — verified against the archive's own manifest

The bundle ships `bundle_checksums.txt`. The delivered CSV's MD5 matches it
byte for byte:

```
published in bundle_checksums.txt : 13f2f9df2e71d7049e966ee405c7f014  data/lunar_crater_database_robbins_2018.csv
computed here                     : 13f2f9df2e71d7049e966ee405c7f014
```

This is independent verification of content, not just of transfer size.

> **DEFECT FOUND IN THE ARCHIVE — the PDS4 label's checksum is wrong.**
> `data/lunar_crater_database_robbins_2018.xml` declares
> `<md5_checksum>6bb5e354a5495dc09732b497dd807ddb</md5_checksum>` for the CSV.
> That value matches neither the delivered file nor the bundle's own
> `bundle_checksums.txt`. The bundle manifest and the delivered bytes agree
> with each other; the label does not agree with either. A validator that
> trusts the label would reject a correct download. This is the same class of
> defect as DECISIONS.md D-009 (a wrong PDS4 label value in a different LRO
> product) and is reported, not reconciled.

Download was performed by `crater.download.fetch_to_path` (owner D) with
`expect_raster=False`, `expected_size_bytes=96227201`,
`RetryPolicy(max_attempts=5, backoff 2→60 s, jitter 0.1)`. The validator ran
`exists`, `not_error_document` and `size`; `checksum` and `raster_opens` were
correctly reported as **skipped** (no checksum is published by the archive at
the download URL, and the product is not a raster). Disk cost: 92 MiB for the
zip plus 228 MiB extracted — immaterial against the 30 GiB budget. No
subsetting was needed.

---

## 2. The real schema

Printed from the delivered file, not from memory. Header (first 361 bytes of
the CSV, CRLF-terminated), 21 comma-separated fields:

```
CRATER_ID,LAT_CIRC_IMG,LON_CIRC_IMG,LAT_ELLI_IMG,LON_ELLI_IMG,DIAM_CIRC_IMG,
DIAM_CIRC_SD_IMG,DIAM_ELLI_MAJOR_IMG,DIAM_ELLI_MINOR_IMG,DIAM_ELLI_ECCEN_IMG,
DIAM_ELLI_ELLIP_IMG,DIAM_ELLI_ANGLE_IMG,LAT_ELLI_SD_IMG,LON_ELLI_SD_IMG,
DIAM_ELLI_MAJOR_SD_IMG,DIAM_ELLI_MINOR_SD_IMG,DIAM_ELLI_ANGLE_SD_IMG,
DIAM_ELLI_ECCEN_SD_IMG,DIAM_ELLI_ELLIP_SD_IMG,ARC_IMG,PTS_RIM_IMG
```

First record, verbatim:

```
00-1-000000,-19.8304,264.757,-19.8905,264.665,940.96,21.3179,975.874,905.968,
0.371666,1.07716,35.9919,0.00788792,0.00842373,0.63675,0.560417,0.373749,
0.00208495,0.000968482,0.568712,8088
```

Record count **1 296 796**, exactly as the label's `<records>` declares.
Dtypes as parsed by `crater.catalogue` (explicit, not inferred):
`CRATER_ID` → `string`; fields 2–20 → `float64`; `PTS_RIM_IMG` → `int64`
(min 6, max 8088).

### 2.1 Coordinate convention — from the label, verbatim

`<cart:Geodetic_Model>` of `data/lunar_crater_database_robbins_2018.xml`:

```xml
<cart:latitude_type>Planetocentric</cart:latitude_type>
<cart:spheroid_name>Moon_2000</cart:spheroid_name>
<cart:semi_major_radius unit="m">1737400</cart:semi_major_radius>
<cart:semi_minor_radius unit="m">1737400</cart:semi_minor_radius>
<cart:polar_radius    unit="m">1737400</cart:polar_radius>
<cart:longitude_direction>Positive East</cart:longitude_direction>
```

and `<cart:Bounding_Coordinates>` west `0.000517267`, east `360`,
north `89.9735`, south `-89.8479`.

| Question | Answer | Where it comes from |
|---|---|---|
| Which column is centre longitude? | **`LON_CIRC_IMG`** for the circle fit, `LON_ELLI_IMG` for the ellipse fit | label field order; the two differ by ~0.1–0.5° for real records |
| Which column is centre latitude? | **`LAT_CIRC_IMG`** / `LAT_ELLI_IMG` | same |
| Planetocentric or planetographic? | **Planetocentric** | label `<cart:latitude_type>` — and on a sphere (`a = b = c = 1737400`) the two coincide exactly, so no conversion exists to get wrong |
| East- or west-positive? | **East-positive** | label `<cart:longitude_direction>` |
| 0–360 or −180–180? | **0–360** | label bounding coordinates; confirmed on the data: measured range of `LON_CIRC_IMG` is 0.000517 … 360.000 |
| Datum | **Moon 2000 sphere, R = 1 737 400 m** | label; **identical to `config/project.yaml`**, so no datum shift applies |

The project convention is east-positive in `[-180, 180)` on the same sphere, so
the only transformation needed is a longitude domain change, done with
`crater.geometry.wrap_longitude` and nothing else.

> Note: the Astropedia web page's *FGDC* block states `Radius A 1737151.3`
> (the LOLA mean radius) while the delivered PDS4 label states `1737400`.
> **The label governs**, and 1 737 400 m is also the project's sphere. The
> 249 m difference would be a 0.014 % scale error if the web page were
> believed instead — small, but it is a disagreement inside one archive and is
> recorded here rather than averaged away.

### 2.2 Diameter columns — which fit each one holds

The label declares **no `unit` attribute on any field**. That is a real gap in
the archive, and the bundle's two documents (`lunar_crater_database_README.pdf`,
`lunar_crater_database_archive_description.pdf`) do not define columns either.
The units below are therefore stated *with the evidence that establishes them*.

| Column | Quantity | Unit |
|---|---|---|
| `DIAM_CIRC_IMG` | **diameter of the best-fit circle** — the project default | km |
| `DIAM_CIRC_SD_IMG` | standard deviation of that circle diameter (a fit uncertainty, **not** a second diameter) | km |
| `DIAM_ELLI_MAJOR_IMG` | **major axis length** of the best-fit ellipse (a full axis, not a semi-axis) | km |
| `DIAM_ELLI_MINOR_IMG` | **minor axis length** of the best-fit ellipse | km |
| `DIAM_ELLI_ECCEN_IMG` | eccentricity — **dimensionless, not a length** despite the `DIAM_` prefix | 1 |
| `DIAM_ELLI_ELLIP_IMG` | ellipticity = major/minor — **dimensionless** | 1 |
| `DIAM_ELLI_ANGLE_IMG` | azimuth of the major axis — **an angle** | deg |
| `*_SD_IMG` of each | standard deviation of the corresponding quantity | as above |
| `ARC_IMG` | fraction of the circumference actually traced, in (0, 1] | 1 |
| `PTS_RIM_IMG` | number of traced rim vertices | count |

The `DIAM_` prefix is **not** a reliable indicator of a length: three of the
columns carrying it are dimensionless or angular. `crater.catalogue` therefore
refuses to unit-convert anything outside its explicit `DIAMETER_COLUMNS` map.

**Verified on the delivered data (200 000 records):**

* `max | DIAM_ELLI_ELLIP_IMG − major/minor | = 1.8 × 10⁻⁵` → `ELLIP` is
  confirmed to be `major/minor`.
* `max | DIAM_ELLI_ECCEN_IMG − sqrt(1 − (minor/major)²) | = 1.5 × 10⁻⁴` →
  `ECCEN` is confirmed to be the standard eccentricity, and `MAJOR ≥ MINOR`
  holds for 100 % of records. This settles that `MAJOR`/`MINOR` are full axes
  consistently ordered, not semi-axes or an arbitrary pair.

**Evidence that the length unit is kilometres** (three independent checks,
because the archive declares none):

1. `min(DIAM_CIRC_IMG) = 1.0` **exactly**, over all 1 296 796 records. This
   matches the archive's own "approximately complete … larger than about
   1–2 km" and the release's companion file name
   `Catalog_Moon_Release_20180815_1kmPlus.vrt`. A metre-unit interpretation
   would mean a 1 m cut-off, which is absurd for WAC/TC-based mapping.
2. `max(DIAM_CIRC_IMG) = 2491.87` at lat −52.698, lon 177.587 E — that is the
   **South Pole–Aitken basin** (~2500 km across). Only kilometres fit.
3. The record at lat −89.6587, lon 129.883 E has `DIAM_CIRC_IMG = 20.8243`,
   i.e. **Shackleton** (IAU: ~21 km at ~89.6 S, ~129.8 E). The bundle's own
   archive-description PDF names Shackleton as the validation case for the
   fitting procedure, so this is the archive's own cross-check reproduced.

Conversion applied: `diameter_m = DIAM_* × 1000`, exactly.

### 2.3 Distance kind and missing values

The archive-description PDF states the fits used "Great Circle distances and
bearings". That is the same planimetric-on-the-sphere convention as
`crater.geometry.DIAMETER_DISTANCE_KIND = "planimetric_great_circle"`, so
catalogue diameters and project diameters are the same *kind* of measurement.
Neither is a terrain-surface distance; on the Malapert massif flanks the true
surface distance is longer and this project does not correct for it.

**38 records** have blank `DIAM_ELLI_MAJOR_IMG` / `DIAM_ELLI_MINOR_IMG` (no
ellipse fit). They keep a circle fit. These stay `NaN` throughout — they are
never back-filled from the circle fit, and `equivalent_ellipse_diameter_m`
returns `NaN` for them.

> **Second archive defect.** The label declares `PTS_RIM_IMG` as
> `ASCII_String`; the delivered values are integers (6 … 8088). The module
> parses it as `int64` and reports the disagreement rather than silently
> reconciling it.

---

## 3. The ROI audit

ROI (fixed, taken as a parameter — never hard-coded in `src/`): south polar
stereographic on Moon 2000, `lon_0 = 0`, `lat_0 = lat_ts = −90`, `k0 = 1`,
1 m/px; `x ∈ [−11000, 10000] m`, `y ∈ [111000, 132000] m`; 21 × 21 km.
This is exactly the grid of the controlled Malapert mosaic — verified by
opening the raster: `bounds = (−11000, 111000, 10000, 132000)`,
`transform = (1.0, 0, −11000 / 0, −1.0, 132000)`, CRS
`Polar_Stereographic, latitude_of_origin −90, central_meridian 0, sphere 1737400`.
Catalogue and imagery are in the *same* frame, so the comparison in §4 is a
like-for-like one.

Selection is by **projecting crater centres with `geometry.forward`** and
testing the projected box, never by a degree box.

Manifest written: **`data/manifests/robbins_roi.csv`** (16 rows).
Reproduce with:

```
python3 scripts/fetch_robbins.py --audit-only --audit -11000 10000 111000 132000
```

### 3.1 How many, and how big

**16 craters.** Every one of them is larger than the approved maximum.

| | |
|---|---|
| count | **16** |
| minimum diameter | **1008.2 m** |
| 25th / 50th / 75th percentile | 1117.3 / 1327.9 / 1550.0 m |
| maximum | 4748.0 m |
| unknown diameters | 0 |

Binned (`catalogue.diameter_histogram`, which also reports what falls outside
the bins so no row can be lost):

| bin | count |
|---|---|
| below 20 m | 0 |
| 20 – 100 m | **0** |
| 100 – 200 m | **0** |
| 200 – 500 m | **0** |
| 500 – 1000 m | **0** |
| 1000 – 2000 m | 13 |
| 2000 – 5000 m | 3 |
| ≥ 5000 m | 0 |
| unknown | 0 |

* **In the approved 20–1000 m range: 0 craters (0 %).**
* **Below 20 m: 0.**
* **Above 1000 m: 16 (100 %).**

Catalogue density in the ROI: 16 / 439.9 km² true surface area
(441.0 km² projected ÷ k² at −85.98°, k = 1.0012317) = **0.036 craters km⁻²**
at D ≥ 1 km.

Figure: `reports/evidence/robbins_roi_map.png` — all 16 circles drawn to scale
inside the ROI box.

### 3.2 Completeness

The archive's own Completeness Report, verbatim:

> "this database is estimated to be a complete census of all lunar craters
> larger than 1–2 km in diameter. The exact completeness point varies based on
> location (it is complete to smaller diameters in lunar maria where
> identifying impacts is more objective and terrain is flatter)"

The project therefore holds the completeness limit as a **range, 1000–2000 m**,
not a single number. Malapert is rugged highland terrain at 86 S, i.e. the
*unfavourable* end of the archive's own caveat, so the pessimistic 2000 m end
is the one to plan against.

| | count | share |
|---|---|---|
| at or above the optimistic limit (1000 m) | 16 | 100 % |
| at or above the pessimistic limit (2000 m) | **3** | 19 % |
| in the uncertain 1000–2000 m band | **13** | 81 % |
| below 1000 m | 0 | — |

**What this means for a 20–1000 m project.** The approved range's *upper*
bound, 1000 m, is the *optimistic lower* bound of the catalogue's completeness.
The two ranges do not overlap at all: `APPROVED_DIAMETER_RANGE_M[1] ≤
COMPLETENESS_LIMIT_M[0]`, which is asserted in `tests/test_catalogue.py`.
Across **the whole of the approved range the catalogue is not ground truth** —
not "degraded", not "incomplete at the small end", but literally empty. Even
the 16 craters it does supply are mostly (13 of 16) in the band the archive
itself declines to call complete.

Consequences, stated plainly:

* The catalogue **cannot** be used as a training or evaluation label set for
  this survey. It supplies no positives in range.
* It **cannot** be used to measure detector recall: a detector finding 400
  genuine 50 m craters would score 400 false positives against it.
* Unlabelled ground here is **not** verified crater-free
  (`docs/annotation_guide.md` §4). Treating catalogue absence as a negative
  label is the single most damaging error available with this dataset.
* The D-003 rationale — "1000 m is where the Robbins catalogue becomes
  approximately complete, so the catalogue series and the detector series
  overlap and can be compared on the R-plot" — is **weaker than it reads**.
  At 1000 m exactly there are 0 catalogue craters in the ROI; the overlap is a
  single point, not an interval, and the comparison has 13 craters of support
  in a band the archive calls uncertain. An R-plot comparison over this ROI
  will be dominated by small-number statistics (√16 = 4, i.e. ±25 %).
* The reviewed local annotation campaign that D-003 already calls REQUIRED is
  confirmed as the only route to labels in range.

### 3.3 Projected box vs naive degree box

The ROI's projected corners span lon −5.659 … 5.148, lat −86.326 … −85.634.
Selecting on that degree rectangle instead of the projected square returns
**17 craters instead of 16** — a **6.3 % overcount** on a sample of 16.

The extra crater is `10-1-083322` at lon 5.126, lat −86.111, whose projected
position is **x = 10539 m**, i.e. 539 m *east of the ROI's eastern edge*. It
is inside the degree box and outside the survey. The error is not one-sided
either: a degree box built from the ROI's *edge midpoints* rather than its
corners drops craters that genuinely lie inside the projected square, because
a projected square's corners reach further north than the middle of its
northern edge. Both directions are covered by tests
(`test_projected_box_and_degree_box_disagree_near_the_pole`,
`test_degree_box_also_misses_ground_inside_the_square`).

At this latitude one degree of longitude is ~2.1 km of ground and one degree of
latitude ~30.3 km, and the ROI spans 10.8° of longitude — the degree box is
simply the wrong shape. `catalogue.select_in_degree_box` exists only to
demonstrate this and labels its own output
`selection = "degree_box_INCORRECT_NEAR_POLE"`.

---

## 4. Visual check against the real NAC imagery — honest assessment

Imagery: `NAC_ROI_MALAPERTLO1_P860S0003.PYR.TIF`, read over `/vsicurl` with
`GDAL_HTTP_CAINFO` / `CURL_CA_BUNDLE` set to the proxy bundle, **windowed reads
only** (no whole-raster or full decimated read was attempted).
Valid data taken as **DN > 1** throughout (D-011); fill is rendered black and
excluded from every stretch and statistic.

Coverage at the catalogue craters is not the limiting factor: 14 of the 16 were
measured and all have **96–100 % valid (DN > 1)** pixels in a 2.2 D window
(two of the largest were not measured because one windowed read failed
mid-transfer and was not retried — stated rather than glossed).

**Figure: `reports/evidence/robbins_rim_overlay.png`** — the six smallest ROI
craters, catalogue circle in red, at 1 m/px.

I looked at every panel. Panel by panel:

| crater | D (m) | `ARC_IMG` | rim vertices | what the imagery actually shows |
|---|---|---|---|---|
| `10-1-079239` | 1008 | 0.77 | 9 | A very subdued broad low. **No rim crest anywhere on the red circle.** A crisp ~120 m crater sits just *outside* the circle. Agreement poor. |
| `10-2-003040` | 1025 | 0.81 | 15 | A real shadowed hollow. The circle's eastern limb sits near the lit east wall, but the shadowed depression extends ~100–150 m west of the circle. Circle looks **offset east and slightly small**. |
| `10-1-079269` | 1067 | 1.00 | 10 | A shallow dusky depression; the lower-left arc follows a tonal/slope break reasonably. **Best of the six**, agreement within roughly 100 m. |
| `10-2-003047` | 1104 | 0.85 | 21 | Smooth, gently convex bright ground with a faint arcuate scarp on the SW only. The named feature is at best a heavily degraded depression. Agreement weak. |
| `10-1-079241` | 1122 | 0.67 | 8 | The circle **centre lands on a distinct ~130 m crater**, which is plainly not the 1122 m feature. A broader subdued depression surrounds it and partly follows the circle's S and W. Marginal. |
| `10-1-079339` | 1124 | 1.00 | 12 | A bright rim arc on the east follows the circle fairly well; the SW limb runs through deep shadow. Moderate agreement, offset perhaps ~100 m. |

**Verdict.** The catalogue centres land on *real topographic depressions* in
roughly four to five of the six cases — the craters are genuinely there. But
at 1 m/px the circles frequently do **not** coincide with any traceable rim
crest, because these are heavily degraded highland craters at the catalogue's
own detection limit, mapped on 70–100 m/px WAC, 30 m/px TC and 5–60 m/px
hillshades, with as few as **8–10 rim vertices** and arc fractions as low as
**0.67**. One of the six is not usable as a rim at all.

**Estimated positional offset: of order 100–200 m, i.e. roughly 10–20 % of the
diameter.** This is a *visual* estimate from one illumination, not a measured
registration. I did not fit rims and I am not claiming a measured offset. Two
things limit it further:

* The interiors are substantially shadowed at this low Sun, and a shadow
  boundary is not a rim (`docs/annotation_guide.md` §2, D-005). An apparent
  offset can be an illumination artefact.
* Only variant LO1 was inspected. The mosaic has nine illuminations and
  ~434.6 km² of the ROI has ≥ 3 of them; a proper offset measurement should
  use several.

A systematic frame error is unlikely to explain the residuals: the catalogue
label and the raster both declare the Moon 2000 sphere at R = 1 737 400 m with
`lat_0 = −90, lon_0 = 0`, both products are LOLA-controlled, and the raster's
geotransform matches the ROI bounds exactly. The residuals look like
crater-fitting error at the catalogue's resolution limit, not georeferencing.

**Implication for sizes.** Diameters are right in order of magnitude but I
cannot verify them better than a few tens of percent from this imagery, and
with `ARC_IMG` down to 0.67 on an 8-vertex fit the catalogue's own uncertainty
columns (`DIAM_CIRC_SD_IMG`) should be carried into any use, not dropped.
`data/manifests/robbins_roi.csv` carries `rim_arc_fraction` and
`rim_vertices_traced` per crater for exactly this reason.

---

## 5. The small-crater label gap — the number that matters

**Figure: `reports/evidence/robbins_small_crater_gap.png`** — six 1 × 1 km NAC
patches at 1 m/px spread across the ROI, with *every* catalogue circle that
crosses each patch drawn in red.

**Result: 0 red circles in all six patches.** Six square kilometres of
1 m/px imagery, densely cratered, with not one catalogue label crossing them.

**Figures: `reports/evidence/robbins_gap_count_grid.png`** (three 300 × 300 m
patches, 50 m grid) and **`reports/evidence/robbins_gap_zoom150.png`** (three
150 × 150 m patches, 25 m grid, 20 m reference bar) — rendered for counting.

### 5.1 The eyeball count

Counted by eye on the 150 × 150 m panels (0.0225 km² each), where a 20 m crater
spans 13 % of the panel width and is unambiguous. I counted features showing a
rim-and-shadow pair consistent with the scene illumination, in the
**20–200 m** range, and deliberately excluded anything I judged smaller or
ambiguous under this single illumination:

| patch (projected m) | craters counted, 20–200 m | area | implied density |
|---|---|---|---|
| (−2200, 113000) | ~6 | 0.0225 km² | ~270 km⁻² |
| (1200, 123200) | ~5 | 0.0225 km² | ~220 km⁻² |
| (8000, 130000) | ~5 | 0.0225 km² | ~220 km⁻² |
| **combined** | **~16** | **0.0675 km²** | **~240 km⁻²** |

**A disagreement to record, not to average away.** The lead's own inspection of
these same three panels (commit `14697e2`) described "**dozens** of unambiguous
craters whose diameters are at or above the 20 m bar" per patch, which would be
~10³ km⁻². My conservative count is ~5–6 per patch. The difference is the
threshold, not the imagery: each panel does contain several dozen crater-like
depressions, but on my reading most of them are **below** 20 m (the dense
8–15 m population) or are too subdued to call at one illumination. Neither
count was made by a second reviewer, and neither is a rigorous crater count.
The two readings bracket the answer rather than contradict it.

Treating this as the crude estimate it is — single illumination, one observer,
no second-reviewer check, Poisson scatter of ±25 % on 16 counts, and a
selection that becomes unreliable as the diameter approaches 20 m — the
defensible statement is:

> **Of order 10²–10³ craters per km² in the 20–200 m range are visible in the
> imagery and absent from the catalogue. Over the 441 km² ROI that is of order
> 10⁵ craters** (my conservative count gives ~1.1 × 10⁵; the lead's reading
> gives several times more).

Of which the catalogue contains **zero**. The order of magnitude is the point,
and it is not sensitive to which of the two counts is right.

### 5.2 The gap, stated as a ratio

| | |
|---|---|
| craters in the 20–1000 m approved range, **in the catalogue** | **0** |
| craters in the 20–200 m range, **visible in the imagery** (estimated) | **~10⁵** |
| ratio | **undefined — the denominator is zero** |

This is not a catalogue that is 95 % or 99 % incomplete over the project's
range. It is a catalogue with **no entries at all** in that range, by
construction: the release is truncated at exactly 1.000 km.

This is DECISIONS.md D-003's "small-crater label gap", now measured rather than
assumed. It confirms the decision's accepted consequence: **a reviewed local
annotation campaign is the only source of labels for this project**, and it
sets the scale of that campaign — a complete 20 m-and-up census of the 441 km²
ROI is a ~10⁵-object task and is not feasible by hand. A realistic programme
annotates a sampled subset of tiles to completeness and reports the sampled
area as the counting area (`crater.area`), never the whole ROI.

### 5.3 Project-level consequences

These four points were written by the lead in commit `14697e2` from the audit
numbers above. I have re-derived each from the data and I agree with all four;
they are kept here so they are not lost.

1. **Robbins cannot supply training labels for this project.** Not "sparse",
   not "incomplete" — zero, in the target range.
2. **Robbins cannot supply a comparison series either.** Sixteen craters over
   441 km², all above the range, cannot populate an informative R-plot. Any
   catalogue series here would be a handful of bins with counts of order one.
3. **Narrowing the diameter range is not an escape.** Counting only above
   ~1 km leaves 16 objects in this ROI — not enough to train a detector or to
   make a measurement. The option is withdrawn as a recommendation.
   (My §3.2 adds the caveat that 13 of those 16 are in the 1–2 km band the
   archive itself declines to call complete, which makes it worse still.)
4. **Unlabelled ground is emphatically not verified background.** Training on
   this ROI with catalogue labels and treating everything else as negative
   would teach a detector that essentially every visible crater is background.

Of the three routes available, a reviewed small-crater annotation campaign is
now the only one that reaches 20–1000 m; external-data pretraining followed by
local annotation and fine-tuning reduces, but does not remove, the annotation
burden. Both require human-reviewed local labels, no model proposal may be
promoted to `source_label_type = human` (`docs/annotation_guide.md`), and the
independent evaluation subset must consist only of labels that were `human`
from the outset.

---

## 6. What this module does and does not provide

`src/crater/catalogue.py` (new, owner E):

* `load_robbins_csv` — reads with `dtype=str, na_filter=False` first so pandas'
  NaN vocabulary ("NA", "null", …) is never applied to a measurement column,
  then converts per column with explicit dtypes. A **blank** field becomes an
  explicit `NaN`/`pd.NA`; a **present-but-unparseable** field raises
  `MalformedCatalogueRow` with the row numbers and the offending values. The
  header is checked against the 21-field label schema and the record count can
  be asserted against `<records>`.
* `normalise_catalogue` — `wrap_longitude` for the 0–360 → `[-180, 180)` domain
  change, latitude untouched (same planetocentric sphere), diameter × 1000 to
  metres. Refuses to treat `DIAM_ELLI_ECCEN_IMG` / `ELLIP` / `ANGLE` as
  lengths. Carries `diameter_definition` in a column and in `.attrs`.
* `ProjectedBox` + `select_in_projected_box` — half-open `[min, max)` on both
  axes, selection by `geometry.forward`. `select_in_degree_box` is provided
  only to demonstrate the error and says so.
* `catalogue_rim_annotations` / `rim_annotation_frame` — rim circles via
  `geometry.crater_rim_samples`, every record carrying
  `source_label_type="catalogue"`, `review_status="unreviewed"`,
  `geometry_kind="catalogue_circle_approximation"` and
  `diameter_distance_kind="planimetric_great_circle"`. A crater with no
  diameter yields `boxes.UNMEASURED` and **no rim samples** —
  `quality_flag="unmeasured"` — rather than a guessed circle.
* `diameter_histogram` — counts below the first edge, above the last and
  unknown as their own rows, so bin counts plus unknowns always equal the
  input length. A histogram that quietly drops rows is how a label gap gets
  hidden.

It does **not** provide: a quality judgement on individual craters, any
reconciliation of the two archive defects in §1.3 and §2.3, or any claim that
catalogue geometry is a measurement.

Tests: `tests/test_catalogue.py`, **42 passed**, no network, synthetic fixture
built on the real 21-field header.

---

## 7. Open issues and recommendations

1. **D-003's R-plot overlap argument should be revisited.** Its premise — that
   the catalogue and the detector series overlap at 1000 m — is true only in
   the limit. In this ROI the catalogue has 0 craters at 1000 m and 13 in the
   uncertain 1–2 km band. Either widen the ROI used for the *catalogue* arm of
   the comparison (the catalogue is global; a 100 km box around the ROI holds
   553 craters, a 200 km box 2190) or state that the R-plot comparison is
   anchored on a neighbourhood, not on the survey ROI. **Needs the lead.**
2. **Two defects in the delivered archive** (§1.3 label MD5 wrong; §2.3
   `PTS_RIM_IMG` declared `ASCII_String` but integer) — reported, not worked
   around. Worth a note to the USGS contact (`astroweb@usgs.gov` on the
   landing page) if the project cites this product.
3. **The Astropedia FGDC radius (1737151.3 m) disagrees with the PDS4 label
   (1737400 m).** The label is used. Flagged in case another owner reads the
   web page instead.
4. **The paper is paywalled.** `doi:10.1029/2018JE005592` is closed access
   (OpenAlex `oa_status: closed`, no repository copy; Wiley returns 403). Its
   column-definition table could not be retrieved, so §2.2's units rest on the
   three data-side checks and the bundle's own documents instead. If anyone
   obtains the paper, §2.2 should be checked against its table.
5. **`src/crater/__init__.py` does not list `catalogue`** in `__all__`. That
   file is the lead's, so it was not edited; `from crater import catalogue`
   works regardless. **For the lead to add if wanted.**
6. **Two windowed reads of the two largest ROI craters failed mid-transfer**
   and were not retried. Nothing in the audit depends on them, but a
   production reader should wrap `rasterio` windowed reads in the same bounded
   retry `crater.download` uses.
7. **The visual offset in §4 is an estimate, not a measurement.** If the
   project needs a real catalogue-to-imagery registration figure, it should be
   measured by fitting rims across several illumination variants — which is a
   piece of work, not a by-product of this audit.
8. **`rim_arc_fraction` is now defined — the lead's flag on it can be
   cleared.** Commit `14697e2` recorded that the column's definition "has not
   been verified … and should not be quoted until it is". It is the delivered
   `ARC_IMG` column, carried through unchanged; the manifest names its source
   in `arc_fraction_source_column`. It is the **fraction of the crater's
   circumference over which rim points were actually traced**, in (0, 1], which
   follows from the archive-description PDF ("crater rims were manually traced,
   **where visible**, with approximately 2.5 pixels per vertex point") and is
   supported by the data. Measured on 300 000 records: `ARC_IMG` spans
   0.0276 … **exactly 1.0**, with 1.0 a hard ceiling reached by 40 % of
   records — the signature of a *fraction*, not of a count or an angle. The
   implied traced rim length per vertex, `π·D·ARC_IMG / PTS_RIM_IMG`, has a
   median of 299 m for 1.0–1.2 km craters rising to 380 m above 5 km: roughly
   **constant in absolute ground distance rather than scaling with diameter**,
   which is what a fixed-pixel-scale tracing rule produces, and the right order
   for 2.5 px on 70–100 m/px WAC. (It is not exactly constant — it drifts ~27 %
   across the size range — so this is corroboration, not proof.) In the ROI
   `ARC_IMG` ranges 0.67–1.00. It is a **fit-support
   quality measure and is safe to quote as such**; a value near 0.67 on an
   8-vertex fit means the centre and diameter rest on two-thirds of the rim and
   should be treated as weakly constrained — which is exactly what §4 found by
   eye for `10-1-079241`.

### Process note

`src/crater/catalogue.py` and `tests/test_catalogue.py` were swept into commit
`628de63` by an outside `git add -A` while this work was still in progress, and
commit `14697e2` then committed the rest of these files together with a version
of this report compiled by the lead. **I did not create either commit**, and I
was instructed not to commit. This report has since been rewritten in full from
the source resolution, the schema inspection and the imagery inspection that
were outstanding when `14697e2` was made; the lead's conclusions have been
preserved in §5.3 and its open flag on `rim_arc_fraction` is cleared above.

## Evidence files

| file | what it shows |
|---|---|
| `reports/evidence/robbins_roi_map.png` | all 16 ROI craters to scale in the projected ROI box |
| `reports/evidence/robbins_rim_overlay.png` | catalogue circles over 1 m/px NAC, six smallest ROI craters |
| `reports/evidence/robbins_small_crater_gap.png` | six 1 × 1 km patches, 0 catalogue circles crossing any of them |
| `reports/evidence/robbins_gap_count_grid.png` | three 300 × 300 m patches, 50 m grid |
| `reports/evidence/robbins_gap_zoom150.png` | three 150 × 150 m patches, 25 m grid, 20 m bar — the counting basis |
| `reports/evidence/robbins_zoom_counting.png` | two 400 × 400 m patches, 50 m bar |
| `data/manifests/robbins_roi.csv` | the 16 ROI craters with labels, flags and provenance |
| `data/raw/robbins/provenance.json` | URL, access date, sizes, checksums, bundle member listing (git-ignored) |
