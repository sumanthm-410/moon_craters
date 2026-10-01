# Branch A source verification — LROC NAC polar mosaics

Date 2026-10-01. All values read from the archive in this session.

## Product selected
LROC RDR **BDRNPL** — "NAC Polar" mosaics, south pole set
`bdr.nac_pole.nac_pole_south.nac_pole_p860s****`, dataset `lro-l-lroc-5-rdr`,
producer Arizona State University, product version 1.0,
creation 2014-12-04. Resolution **1.0 m/pixel**.

Discovered via the ODE REST API (`oderest.rsl.wustl.edu`), which is an
authoritative PDS node interface. ODE itself publishes the file location as
`pds.lroc.im-ldi.com`, which 302-redirects to the NASA PDS store at
`pds.mcp.nasa.gov`. The im-ldi host is therefore not a guessed mirror; it is
the location the archive index points to, and the bytes are served by NASA PDS.

## Why this product, under branch A
Branch A requires already map-projected products whose provenance,
projection, scale, calibration and terrain treatment are verified before
reuse. The alternative (EDR -> ISIS) is impossible here: there is no
container runtime and no conda (D-004). These BDR mosaics are ASU-produced,
controlled NAC mosaics delivered already in a lunar polar projection.

## Projection — VERIFIED, and a label defect found
PDS4 label `NAC_POLE_P860S0337.xml` declares:

| Parameter | Label value |
|---|---|
| map_projection_name | Polar Stereographic |
| longitude_of_central_meridian | 0.0 deg |
| latitude_of_projection_origin | -90.0 deg |
| pixel_scale_x / _y | 1.0 m/pixel |
| latitude_type | Planetocentric |
| a/b/c_axis_radius | 1737.4 km (sphere) |
| longitude_direction | Positive East |

This matches `config/project.yaml` `target_crs` exactly:
`+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +a=1737400 +b=1737400`.

**Label defect.** The label's `cart:upperleft_corner_x` is **-40627.5 m**,
but the delivered GeoTIFF's own geotransform gives **+40627.0 m** (and the
independent analytic projection in `crater.geometry` gives +40627.5 m from
the label's own lat/lon bounds). The y value agrees to 0.4 m. The sign of
the label's x corner is therefore wrong; the label carries a modification
history entry reading "Correction to PDS4 labels", consistent with a
PDS3->PDS4 migration defect.

**Decision: georeferencing is taken from the delivered raster, never from
`cart:upperleft_corner_x`.** Had the label been trusted, the survey would
have been mirrored about the x axis.

Independent confirmation that the standard parallel is the pole (k0 = 1):
the raster's y bounds reproduce `rho(lat)` from `crater.geometry.forward`
to within half a pixel. A different `lat_ts` would scale all radii by a
constant and this would not match.

## Tiles covering the ROI candidates (read from the rasters)
| Tile | size (px) | projected x | projected y | lon range | lat range |
|---|---|---|---|---|---|
| P860S0112 | 52246 x 38443 | 0 .. 52246 | 98083 .. 136526 | 0.00 .. 28.04 | -86.77 .. -85.18 |
| P860S3487 | 52246 x 38443 | -52246 .. 0 | 98083 .. 136526 | -28.04 .. 0.00 | -86.77 .. -85.18 |

ROI candidate A (2.9 E, 85.9 S) lies in **P860S0112**.
ROI candidate B (356.4 E, 85.8 S) lies in **P860S3487**.

## Files per tile (sizes measured by HTTP HEAD after redirect)
| File | Size | Purpose |
|---|---|---|
| `.IMG` | 10.636 GiB | PDS4 product, detached label |
| `.PYR.TIF` | 3.571 GiB | tiled GeoTIFF, 256x256 blocks |
| `.MASK.TIF` | 2 MiB | valid-data mask |
| `.BROWSE.PNG` | 283 KiB | browse |
| `NAC_POLE_SOUTH_IMAGES.TXT` | 16 KiB | **source NAC image list — provenance layer** |

The NASA PDS store advertises `Accept-Ranges: bytes` and honours it, so
**windowed reads work and the 10.6 GiB product does not have to be
downloaded**. A 30 km ROI extracted at 1 m/px is ~858 MiB.

## Pixel quality at the two ROI candidates (1536 x 1536 px native windows)
| Centre | stretch (p1-p99) | nodata | mask valid | assessment |
|---|---|---|---|---|
| A 2.9 E, 85.9 S | 34 - 199 | 0.0% | 100% | well exposed, not saturated; abundant small craters with clear rim/shadow pairs across a wide size range |
| B 356.4 E, 85.8 S | 129 - 255 | 0.0% | 100% | much brighter, approaching saturation; strong illumination gradient; downslope lineation texture; markedly fewer crisp small craters |

Evidence: `reports/evidence/pilot_nac_patches.png` (display stretch only).

## Still to verify before this product is used for measurement
- Calibration state and photometric normalisation of the BDR mosaic.
- Terrain correction: whether these are orthorectified against a DEM or
  simply projected. **Map projection is not terrain correction**, and this
  is not yet established for this product.
- Registration residuals between adjacent tiles and against control.
- The source NAC image list, for per-pixel acquisition provenance.
