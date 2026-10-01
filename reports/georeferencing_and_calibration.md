# Georeferencing and calibration state of the selected source

Verified 2026-10-01 against the authoritative PDS4 label and the delivered
rasters. This document governs how every stage reads these files.

## 1. Projection — VERIFIED from the label
From `NAC_ROI_MALAPERTLO1_P860S0003.xml`:

| Parameter | Value |
|---|---|
| map_projection_name | Polar Stereographic |
| latitude_of_projection_origin | -90.0 deg |
| longitude_of_central_meridian | 0.0 deg |
| **scale_factor_at_projection_origin** | **1.0** |
| pixel_scale_x / _y | 1.0 m/pixel |
| latitude_type | Planetocentric |
| longitude_direction | Positive East |
| a/b/c_axis_radius | 1737.4 km (sphere) |
| processing_level | **Derived** |

`scale_factor_at_projection_origin = 1.0` confirms k0 = 1 at the pole. This
was previously an assumption in `config/project.yaml`; it is now verified
from the product's own label, so `crater.geometry.point_scale_factor`'s
convention (k = 2k0/(1 - sin phi), k = 1 at the south pole) is correct for
this product.

## 2. The PDS4 `upperleft_corner_x` sign defect — SYSTEMATIC
| Source | upper-left x | upper-left y |
|---|---|---|
| Delivered GeoTIFF geotransform | **-11000.0** | +132000.0 |
| PDS4 label `cart:upperleft_corner_x/y` | **+10999.499997616** | +131999.5000025 |

The label is wrong, and it is internally inconsistent with itself:
its own `west_bounding_coordinate = -5.6594818 deg` lies west of the central
meridian, where `x = rho*sin(dlon)` is necessarily **negative**.

Projecting the label's own corner coordinates with `crater.geometry.forward`
puts the south-east corner at exactly **(+10000.000, +111000.000) m**, which
is precisely the GeoTIFF's lower-right corner.

Half-pixel check (a GeoTIFF corner is a pixel *edge*, a PDS4 corner is a
pixel *centre*, so a -0.5 m offset is expected):

    |label_x| - |geotiff_ulx| = -0.500002 m
     label_y  -  geotiff_uly  = -0.499998 m

Both are exactly -0.5 m. The magnitudes agree perfectly under the half-pixel
convention and **only the x sign is wrong**.

The same defect was independently found in a different product family
(`NAC_POLE_P860S0337`), so it is systematic in this PDS3 -> PDS4 migration,
not a one-off. Both labels carry a "Correction to PDS4 labels" modification
entry.

**RULE (D-009, extended): georeferencing is taken from the delivered GeoTIFF
geotransform. `cart:upperleft_corner_x` is never used.** Trusting it would
mirror the entire survey about the x axis.

## 3. GDAL's PDS4 driver must not be used for georeferencing here
Opening the `.xml` label with GDAL yields a geotransform whose pixel size is
`3.2977886e-05` — the label's **deg/pixel** value, not the 1 m/pixel map
scale — and it inherits the wrong x sign. A projected-metre window computed
against it lands off-target (verified: returned all-zero).

**RULE: read `.IMG` pixel data by array index, and apply the geotransform
from the GeoTIFF.** Never use the PDS4 driver's geotransform.

## 4. Pixel data: authoritative `.IMG` vs browse `.TIF`
| | `.IMG` (PDS4 product) | `.PYR.TIF` / `.TIF` (browse) |
|---|---|---|
| dtype | uint16 (`UnsignedLSB2`) | uint8 |
| fill / missing | `missing_constant = 0` | DN = 1 (undeclared, `nodata = None`) |
| saturation markers | low 2, high 65534 | clipped at 255 |
| distinct values in a 512x512 test window | 7878 | 254 |

The two browse TIFs are pixel-identical to each other. Against the `.IMG`
the browse is a **linear** rescale, not a non-linear display stretch:

    TIF = 0.00534809 * IMG + 0.3696
    Pearson r = 0.999904, Spearman rho = 0.999953, residual rms = 0.479 DN

so ordering and relative structure are preserved. But it collapses ~7900
levels to 254 and **clips the bright tail** (IMG values above ~47700 saturate
at 255; max residual 76.6 DN occurs there).

**RULE: measurement and training use the `.IMG` (uint16). The browse TIFs are
acceptable for display, coverage auditing and quick-look only.**
Geometry is unaffected by the rescale; radiometry and bright-slope rim
contrast are.

## 5. Calibration state — stated precisely, not overstated
`processing_level` is **"Derived"**. The label carries no `scaling_factor` or
`value_offset`, so the stored values are **DN of a controlled mosaic**, not
calibrated I/F or radiance.

What this does and does not license:
- These are NOT raw EDR pixels. They are a geodetically controlled,
  map-projected mosaic produced by ASU, so the project's rule against
  training on uncalibrated raw EDR is satisfied.
- They are NOT photometrically calibrated reflectance. No photometric
  quantity, albedo or I/F may be reported from them.
- Crater detection and **geometric** measurement are supported. Any
  radiometric interpretation is not.

## 6. Terrain correction — STILL UNVERIFIED
The label says "controlled mosaic ... in PolarStereographic projection" and
cites [HENRIKSENETAL2023] for the control. It does **not** state whether the
mosaic is orthorectified against a DEM or simply projected onto a sphere.
**Map projection is not terrain correction.** On the steep flanks of a massif
an uncorrected projection displaces features by roughly h * tan(emission
angle). Until this is resolved from the cited reference or ASU documentation,
the project reports planimetric diameters only and makes no claim of
terrain correction. Recorded in scientific_limitations.md.
