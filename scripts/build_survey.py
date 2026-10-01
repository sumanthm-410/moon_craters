#!/usr/bin/env python3
"""Build the survey products: valid masks, survey area, and a georeferenced COG.

Reads the authoritative uint16 .IMG by array index and applies the GeoTIFF
geotransform, per DECISIONS.md D-014/D-015 (the PDS4 label's x corner sign is
wrong and GDAL's PDS4 geotransform is in deg/pixel).

Outputs (artifacts/, git-ignored; tracked via manifests):
    survey_mask.tif        uint8 coverage count 0..3 over the three variants
    malapert_LO1_cog.tif   the primary mosaic as a georeferenced COG
    survey_area.json       true surface area, with the k^2 correction applied
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.enums import Resampling

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crater import area as ca
from crater import geometry as g

HDR, N, PIX = 42000, 21000, 1.0
X0, Y1 = -11000.0, 132000.0          # GeoTIFF geotransform (authoritative, D-009)
VARIANTS = ["MALAPERTLO1", "MALAPERTLOA", "MALAPERTLOH"]
RAW = ROOT / "data" / "raw" / "malapert"
OUT = ROOT / "artifacts"
TRANSFORM = Affine.translation(X0, Y1) @ Affine.scale(PIX, -PIX)
CRS = rasterio.crs.CRS.from_proj4(
    "+proj=stere +lat_0=-90 +lat_ts=-90 +lon_0=0 +k=1 +x_0=0 +y_0=0 "
    "+a=1737400 +b=1737400 +units=m +no_defs"
)


def img(v: str) -> np.memmap:
    return np.memmap(RAW / f"NAC_ROI_{v}_P860S0003.IMG", dtype="<u2", mode="r",
                     offset=HDR, shape=(N, N))


def main() -> int:
    OUT.mkdir(exist_ok=True)
    grid = ca.StereographicGrid(x_origin=X0, y_origin=Y1, pixel_size_m=PIX,
                                height=N, width=N)

    # --- coverage masks, built in row blocks to bound memory -----------------
    print("building coverage masks (DN > 0 is valid; label missing_constant = 0)")
    count = np.zeros((N, N), dtype=np.uint8)
    for v in VARIANTS:
        a = img(v)
        tot = 0
        for r0 in range(0, N, 2100):
            r1 = min(r0 + 2100, N)
            blk = np.asarray(a[r0:r1]) > 0
            count[r0:r1] += blk.astype(np.uint8)
            tot += int(blk.sum())
        print(f"  {v:12} valid {100*tot/N**2:6.2f}%  ({tot*PIX**2/1e6:7.1f} km^2 projected)")
        del a

    usable = count >= 1
    np.save(OUT / "coverage_count.npy", count[::10, ::10])  # small QA copy

    # --- areas, with and without the projection correction ------------------
    # Computed in row blocks: a full-resolution StereographicGrid caches several
    # 21000x21000 float64 arrays (lat, k, k**2, per-pixel area), which is ~3.5 GB
    # each and OOM-kills the process. Each block gets its own grid with a shifted
    # y_origin, so the tested area maths in crater.area is used unchanged.
    print("\nsurvey area (block-wise, full resolution)")
    BLK = 1500
    res = {}
    for name, m in [("any_variant", count >= 1), ("two_or_more", count >= 2),
                    ("all_three", count >= 3)]:
        true_km2 = 0.0
        naive_km2 = 0.0
        for r0 in range(0, N, BLK):
            r1 = min(r0 + BLK, N)
            sub = m[r0:r1]
            if not sub.any():
                continue
            gb = ca.StereographicGrid(x_origin=X0, y_origin=Y1 - r0 * PIX,
                                      pixel_size_m=PIX, height=r1 - r0, width=N)
            true_km2 += ca.true_area_km2(sub, gb)
            naive_km2 += ca.naive_projected_area_m2(sub, gb) / 1e6
        err = naive_km2 / true_km2 - 1.0 if true_km2 else float("nan")
        res[name] = {"true_surface_km2": true_km2,
                     "naive_projected_km2": naive_km2,
                     "naive_relative_error": err,
                     "pixels": int(m.sum())}
        print(f"  {name:13} true {true_km2:8.3f} km^2   naive {naive_km2:8.3f} km^2  "
              f"(naive is {100*err:+.4f}% high)")

    # --- diameter-dependent counting area (edge rule) -----------------------
    # Computed on a 5x decimated mask (5 m/px): the full-resolution distance
    # transform needs a 21000x21000 float64 array. At 5 m/px the smallest
    # buffer in play (D/2 = 10 m for a 20 m crater) is 2 px, which is adequate;
    # the approximation is recorded in the output.
    DEC = 5
    dmask = usable[::DEC, ::DEC]
    gdec = ca.StereographicGrid(x_origin=X0, y_origin=Y1, pixel_size_m=PIX * DEC,
                                height=dmask.shape[0], width=dmask.shape[1])
    diam = np.array([20.0, 50.0, 100.0, 250.0, 500.0, 1000.0])
    A_i = ca.usable_area_km2_by_diameter(gdec, dmask, diam)
    print("\ncounting area A_i by diameter (edge rule: centre >= D/2 inside)")
    for d, a_ in zip(diam, A_i):
        print(f"  D = {d:7.1f} m -> A_i = {a_:8.3f} km^2  "
              f"({100*a_/res['any_variant']['true_surface_km2']:5.1f}% of survey)")
    res["counting_area_km2_by_diameter"] = {str(d): float(a_) for d, a_ in zip(diam, A_i)}
    res["counting_area_decimation_m_per_px"] = PIX * DEC
    res["edge_rule"] = ca.EDGE_RULE
    res["scale_factor_k_at_roi_centre"] = g.point_scale_factor(-85.9)

    (OUT / "survey_area.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT/'survey_area.json'}")

    # --- survey mask raster --------------------------------------------------
    prof = dict(driver="GTiff", height=N, width=N, count=1, dtype="uint8",
                crs=CRS, transform=TRANSFORM, nodata=0, tiled=True,
                blockxsize=512, blockysize=512, compress="deflate", predictor=2,
                BIGTIFF="IF_SAFER")
    with rasterio.open(OUT / "survey_mask.tif", "w", **prof) as ds:
        ds.write(count, 1)
        ds.build_overviews([2, 4, 8, 16, 32], Resampling.nearest)
        ds.update_tags(
            description="Count of approved illumination variants with valid data",
            variants=",".join(VARIANTS), valid_rule="source DN > 0",
            note="See DECISIONS.md D-011/D-014/D-015")
    print(f"wrote {OUT/'survey_mask.tif'}")

    # --- primary mosaic as a georeferenced COG ------------------------------
    print("writing primary mosaic COG (uint16, deflate, tiled, overviews)")
    a = img("MALAPERTLO1")
    prof16 = dict(prof, dtype="uint16", nodata=0, predictor=2)
    with rasterio.open(OUT / "malapert_LO1_cog.tif", "w", **prof16) as ds:
        for r0 in range(0, N, 2100):
            r1 = min(r0 + 2100, N)
            ds.write(np.asarray(a[r0:r1]), 1,
                     window=rasterio.windows.Window(0, r0, N, r1 - r0))
        ds.build_overviews([2, 4, 8, 16, 32], Resampling.average)
        ds.update_tags(
            source="bdr.nac_roi.malapertlo1.nac_roi_malapertlo1_p860s0003",
            processing_level="Derived", units="DN (not calibrated I/F)",
            georeferencing="GeoTIFF geotransform; PDS4 upperleft_corner_x sign is wrong (D-009)")
    print(f"wrote {OUT/'malapert_LO1_cog.tif'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
