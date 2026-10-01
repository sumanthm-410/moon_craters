# Malapert ROI source — controlled NAC mosaic (selected product)

Verified 2026-10-01. Supersedes the generic NAC_POLE tiles as the primary
survey source (see "Why this and not NAC_POLE" below).

## Product family
`bdr.nac_roi.malapert**.nac_roi_malapert**_p860s0003`, LROC RDR **BDRROI**,
dataset `lro-l-lroc-5-rdr`, producer Arizona State University,
**product version 2.0, created 2024-01-30**.

PDS4 description, verbatim:
> "Malapert Massif Low Sun controlled mosaic in PolarStereographic
> projection centered at 86.0S, 0.3E. For more info, see [HENRIKSENETAL2023]."

The word **controlled** is the reason this product was chosen: it is a
geodetically controlled mosaic with a published control reference, which is
what branch A requires. Observation span 2009-11-26 to 2017-07-17.

## Variants and resolutions
Nine low-Sun illumination variants: `LO1, LOA, LOB, LOC, LOD, LOE, LOF, LOG,
LOH`. Each is published at **1 m/px**, plus `_5m` and `_20m` reductions.

All nine 1 m/px variants share an **identical grid** (verified by opening
each): 21000 x 21000 px, 1.0 m/px, upper-left (-11000.0, 132000.0) m,
x [-11000, 10000], y [111000, 132000]. They are therefore pixel-aligned to
each other and need no co-registration between illuminations.

CRS verified on the rasters: Polar_Stereographic, latitude_of_origin -90,
central_meridian 0, sphere 1737400 m — matching `config/project.yaml`.

Extent: **21.0 x 21.0 km = 441 km^2 projected** (~439.9 km^2 true surface
area after dividing by k^2 at this latitude).
lon -5.659 .. 5.148, lat -86.326 .. -85.634.

## CRITICAL: the fill value is DN = 1 and it is NOT declared
The delivered GeoTIFFs report `nodata = None`, but unimaged ground is filled
with **DN = 1**, not 0. Exactly 0.0% of pixels are 0.
Treating DN=1 as valid data would train the detector on blank fill and would
inflate the survey area. **Valid data is DN > 1**, and the `.MASK.TIF`
sidecar must be used as the authoritative valid-data layer.
This is recorded as a standing rule in DECISIONS.md D-011.

## CORRECTION (2026-10-01, after downloading the 1 m/px products)
The per-variant percentages in the table below were measured on the 20 m/px
browse products, whose frame is **padded** to 24.96 x 21.56 km. They are
fractions of that padded frame, NOT of the 21 x 21 km ROI. Dividing by 0.819
converts them. Measured directly on the 1 m/px `.IMG` inside the true ROI:

| Variant | valid (DN>0) in ROI | area |
|---|---|---|
| LO1 | **100.0%** | 441.0 km^2 projected |
| LOA | 51.5% | 226.9 km^2 |
| LOH | 59.9% | 264.1 km^2 |

Stacked over the three approved variants, inside the ROI:
| at least N | fraction | true surface area |
|---|---|---|
| >= 1 | 100.0% | **439.918 km^2** |
| >= 2 | 72.4% | 318.903 km^2 |
| >= 3 | 38.8% | 170.831 km^2 |

So **LO1 alone covers the entire ROI**, 318.9 km^2 carries two independent
illuminations and 170.8 km^2 carries all three. The naive projected area is
+0.2459% high, matching the predicted 0.244% at this latitude.

## Per-variant coverage (measured on the 20 m/px products — see correction above)
| Variant | valid (DN>1) | mean DN of valid | note |
|---|---|---|---|
| LO1 | 81.7% | 118.9 | best single coverage |
| LOF | 79.6% | 49.1 | dark; visible vertical striping artifact — inspect before use |
| LOH | 48.7% | 109.9 | |
| LOA | 41.2% | 75.6 | |
| LOE | 27.9% | 104.2 | |
| LOD | 26.8% | 139.3 | |
| LOC | 19.9% | 130.9 | |
| LOB | 16.3% | 88.6 | |
| LOG | 10.2% | 71.8 | |

Stacked coverage over the ROI:
| at least N variants | fraction of frame | projected area |
|---|---|---|
| >= 1 | 81.9% | 441.0 km^2 |
| >= 3 | 80.8% | 434.6 km^2 |
| >= 4 | 64.7% | 347.9 km^2 |
| >= 5 | 31.2% | 167.7 km^2 |

(The 20 m product is padded to 24.96 x 21.56 km; the 81.9% is of that padded
frame and corresponds to the full 441 km^2 ROI.)

**~434.6 km^2 of the ROI is imaged under at least three independent
illuminations.** This directly supports measuring recall as a function of
illumination, which matters at 86 S where detection depends on shadow
geometry as well as crater size.

Evidence: `reports/evidence/coverage_by_variant.png` — the left panel shows
the individual NAC strip footprints composing each variant, i.e. how the
strips make the mosaic.

## Visual inspection at 2.9 E, 85.9 S (0.9 km patch, 1 m/px)
LO1, LOA and LOH show genuinely different illumination of the same ground —
the shadow geometry moves between them, and a large crater that is fully
shadowed in LOA is partly lit in LO1. LOB, LOC, LOD, LOE and LOG are blank
(DN=1) at this location. LOF is near-dark with striping.
Evidence: `reports/evidence/illumination_variants.png`.

## Why this and not the NAC_POLE tiles
| | NAC_ROI Malapert | NAC_POLE P860S |
|---|---|---|
| control | **controlled**, published reference | generic polar mosaic |
| version | 2.0 (2024) | 1.0 (2014) |
| resolution | 1 m/px | 1 m/px |
| illumination | **9 variants** | single composite |
| extent | 441 km^2, centred on the target | 52 x 38 km per tile |
| file size | 0.86 GiB | 10.6 GiB |
| ROI fit | purpose-built for Malapert Massif | ROI crosses 2 tiles and overruns the northern edge of the row |

The NAC_POLE row P860S is the northernmost of the south-polar set (rows exist
at 86.0, 87.0, 88.0 and 89.2 S only), so a 35 km square centred at 85.9 S
cannot be covered by it without extending north beyond the product.

## Consequence for the approved ROI extent
The approved extent was 35 km (1225 km^2). The controlled Malapert product
provides **21 km (441 km^2)**. This is a reduction that must be decided by
the user, not absorbed silently. See DECISIONS.md D-012.
