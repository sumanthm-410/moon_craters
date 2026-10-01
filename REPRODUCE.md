# Reproducing and continuing this project

Everything below was executed and verified on 2026-10-01 unless marked
NOT RUN. Nothing here describes a result that was not produced.

## 1. Environment
```bash
python3 -m pip install -r requirements.txt     # pinned; Python 3.11.15
python3 -m pytest tests/ -q                    # expect: all tests pass
```
Geospatial stack ships GDAL 3.10.3 / PROJ 9.5.1 inside the rasterio/pyproj
wheels. No system GDAL, no ISIS, no conda and **no container runtime** are
required for any stage that exists today (D-004).

Outbound access needed: `pds.mcp.nasa.gov`, `oderest.rsl.wustl.edu`,
`astrogeology.usgs.gov`, `pypi.org`. (`www.kaggle.com` only for the training
stage, which has not been run.)

## 2. Stages, in order
```bash
# a. acquire the three approved illumination variants (~2.5 GiB)
python3 scripts/acquire_malapert.py            # --dry-run / --validate-only supported
#    -> data/raw/malapert/*.IMG  (MD5-verified against each PDS4 label)
#    -> data/manifests/manifest.csv

# b. survey masks, true surface area, counting areas, COG
python3 scripts/build_survey.py
#    -> artifacts/survey_mask.tif, malapert_LO1_cog.tif, survey_area.json

# c. spatial splits + leakage audit
python3 scripts/build_splits.py

# d. QA figure
python3 scripts/make_qa_figure.py
#    -> reports/evidence/survey_qa.png
```

## 3. Non-obvious facts you must not lose
These were each found the hard way; ignoring any of them silently corrupts
the result.

1. **The PDS4 `cart:upperleft_corner_x` sign is wrong** in these products
   (both NAC_ROI and NAC_POLE). Georeferencing comes from the delivered
   GeoTIFF. Trusting the label mirrors the survey about the x axis. (D-009)
2. **GDAL's PDS4 driver geotransform is in deg/pixel** and inherits that sign.
   Read `.IMG` pixels by array index (`np.memmap`, offset 42000, `<u2`,
   21000x21000 — verified bit-identical to the driver) and apply the GeoTIFF
   transform. (D-015)
3. **Fill values differ between products**: the `.IMG` uses DN = 0
   (`missing_constant`), the browse `.TIF` uses DN = 1 and declares
   `nodata = None`. Valid data is DN > 0 in the IMG. (D-011)
4. **`<object_length>` is not the file size** — it is the 42000-byte PDS3
   attached header. Use the `<File>` block's `<file_size>`. Getting this wrong
   caused three good downloads to be (correctly) rejected.
5. **A full-resolution `StereographicGrid` OOM-kills a 15 GiB machine**
   (caches several 21000² float64 arrays). Compute areas block-wise.
6. **Counting area and crater counts must use the same edge rule**, or R is
   biased high (INTERFACES.md note 9).
7. **All illumination variants of a ground block share a split**, and all
   pyramid levels go through one `assign_tiles_to_splits` call. (D-013)

## 4. Verified numbers to check against
| Quantity | Value |
|---|---|
| ROI | 21 x 21 km, x[-11000,10000] y[111000,132000] m |
| True survey area (any variant) | 439.918 km² |
| ... >= 2 variants / all 3 | 318.903 / 170.831 km² |
| Naive projected area error | +0.2459% (predicted +0.244%) |
| Counting area at D = 20 m / 1000 m | 439.081 / 399.019 km² |
| Splits (512 px tiles, 1 km buffer) | train 1677, val 380, test 380 |
| Usable split ratio | 0.688 / 0.156 / 0.156 |
| Leakage audit | CLEAN (0 findings); proven non-vacuous |

## 5. NOT RUN — what remains, and what each needs
| Stage | Blocker |
|---|---|
| Annotation campaign | **human labelling time.** The catalogue cannot substitute. |
| Kaggle dataset + GPU training | `KAGGLE_API_TOKEN` in the environment (D-018) |
| Full-ROI inference + dedup + rim measurement | a trained model |
| Detector R-plot | the above |

The analysis code for the last two is written and tested (`crater.dedup`,
`crater.boxes`, `crater.sfd`, `crater.area`) but has only ever been exercised
on synthetic fixtures with known answers — never on real detections.

## 6. To enable Kaggle
The CLI reads `KAGGLE_API_TOKEN` from the environment or
`~/.kaggle/kaggle.json`. A credential added at the egress/proxy layer does
**not** satisfy it. Add the token as an environment secret, restart the
session, then verify read-only:
```bash
kaggle datasets list --mine --page-size 1
```
Never paste the token into chat. See `docs/kaggle_setup.md`.
