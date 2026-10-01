# Reachable lunar data over Malapert — measured feasibility

All numbers below were read directly from the rasters in this session via
GDAL `/vsicurl` windowed HTTP range reads (no bulk download).
Evidence: `reports/evidence/probe_vsicurl.py`, `probe_chips.py`,
`malapert_chips.png`.

Host: `asc-pds-services.s3.us-west-2.amazonaws.com` (USGS Astrogeology
web-service bucket). This is a *service* bucket, not the PDS archive of
record. Provenance must be re-verified against PDS before any result is
published. Status: REACHABLE, provenance UNVERIFIED.

## Products found (global simple-cylindrical, Moon sphere R=1737400 m)

| Product | Grid | Nominal scale | nodata | Tiled |
|---|---|---|---|---|
| Kaguya_TCortho_Mosaic_Global_4096ppd.tif | 1474593 x 737297 uint8 | 4096 ppd ~ 7.4 m/px | 0 | 256x256 |
| LRO_WAC_Mosaic_Global_303ppd_v3.tif | 109164 x 54582 uint8 | 303 ppd ~ 100 m/px | none | 256x256 |
| LRO_LOLA_DEM_Global_256ppd_v06_16bit.tif | 92160 x 46080 int16 | 256 ppd ~ 118 m/px | -32768 | 256x256 |

All are tiled with no overviews; windowed reads succeeded in 0.2-2.7 s.
CRS is a lunar geographic CRS (GCS_Moon / ESRI:104903) — correct body, but
unprojected, so it is unusable for measurement at 86 deg S without
reprojection to a south-polar projection.

## Pixel statistics over the two candidate centres (0.1 deg lat half-window)

| Product | centre 2.9E 85.9S | centre 356.4E 85.8S |
|---|---|---|
| Kaguya TC | p2=15 **p50=254** p98=254 | p2=197 **p50=254** p98=254 |
| WAC | p2=3 p50=121 p98=228 | p2=163 p50=223 p98=236 |
| LOLA DEM | p2=3995 p50=4660 p98=5152 | p2=2030 p50=3293 p98=4582 |

## Visual inspection (performed — see malapert_chips.png)
- **Kaguya TC**: the illuminated surface is clipped at DN 254 (median = 254,
  i.e. >50% of pixels saturated). Craters are visible ONLY as black shadow
  silhouettes against a blown-out background. At centre B there is also a
  visible rectangular block/seam artifact and directional streaking.
  => Photometry is destroyed. A shadow outline is NOT a rim; deriving
  diameters from it would violate the project's own measurement rule.
- **WAC**: real but soft, low-contrast, ~100 m/px. Far too coarse for the
  small-crater objective.
- **LOLA DEM**: smooth, real topography at ~118 m/px; resolves the massif
  slope only, not individual small craters.

## Verdict
No reachable product supports the stated objective (NAC-scale small-crater
detection and rim measurement). Kaguya TC is the only sub-10 m option and it
is saturated at this latitude. There is no usable crater catalogue reachable
at all, so there is no ground truth and no evaluation set regardless of
imagery.
