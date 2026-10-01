# Stable interfaces and file ownership

Agents MUST NOT create, edit or delete files outside their own ownership block.
If you believe another owner's file is wrong, report it; do not edit it.

## Owned by the lead (do not modify)
- `src/crater/body.py`, `src/crater/geometry.py`
- `tests/test_geometry.py`, `tests/conftest.py`
- `PLAN.md`, `STATUS.md`, `DECISIONS.md`, `INTERFACES.md`, `config/project.yaml`

## Ownership blocks
| Owner | Files |
|---|---|
| A: tiling & splits | `src/crater/tiling.py`, `src/crater/splits.py`, `tests/test_tiling.py`, `tests/test_splits.py` |
| B: boxes & dedup | `src/crater/boxes.py`, `src/crater/dedup.py`, `tests/test_boxes.py`, `tests/test_dedup.py` |
| C: area & size-frequency | `src/crater/area.py`, `src/crater/sfd.py`, `tests/test_area.py`, `tests/test_sfd.py` |
| D: acquisition & stages | `src/crater/download.py`, `src/crater/manifest.py`, `src/crater/stages.py`, `tests/test_download.py`, `tests/test_stages.py` |

## Frozen API of `crater.geometry` (verified, 61 tests passing)
All functions accept scalars or numpy arrays and return the same shape.

```python
wrap_longitude(lon_deg)                  -> lon in [-180, 180)
longitude_difference(lon_a, lon_b)       -> signed smallest difference, degrees
forward(lon_deg, lat_deg, body=MOON, lon_0=0.0, k0=1.0)   -> (x_m, y_m)
inverse(x_m, y_m, body=MOON, lon_0=0.0, k0=1.0)           -> (lon_deg, lat_deg)
point_scale_factor(lat_deg, k0=1.0)      -> k  (linear, conformal)
area_scale_factor(lat_deg, k0=1.0)       -> k**2
great_circle_distance(lon1, lat1, lon2, lat2, body=MOON)  -> metres
geodesic_destination(lon, lat, bearing_deg, distance_m, body=MOON) -> (lon, lat)
crater_rim_samples(lon, lat, diameter_m, n_samples=72, body=MOON)  -> (lon[], lat[])
DIAMETER_DISTANCE_KIND = "planimetric_great_circle"
```

`crater.body` exposes `MOON` (a frozen `Body`), `MOON_RADIUS_M = 1737400.0`,
`Body.geographic_proj4`, `Body.polar_stereographic_proj4(south=True, lon_0=0.0)`.

## Non-negotiable project rules
1. The projection is **south polar stereographic on the Moon 2000 sphere**.
   Never assign an Earth EPSG code. Never convert degrees to metres with a
   planar factor; use `great_circle_distance` / `geodesic_destination`.
2. `point_scale_factor` is ~1.0012 at 86 S. A length measured in **projected**
   metres must be divided by `k` to become a ground distance; a projected
   **area** must be divided by `k**2`. Areas and diameters that ignore this are
   wrong by ~0.12% and ~0.24% respectively at the ROI latitude.
3. All distances are **planimetric** (great-circle on the sphere), never
   terrain-surface. Any reported measure must say which it is.
4. Approved target diameter range: **20 m to 1000 m** (DECISIONS.md D-003).
5. The ROI centre is **deliberately unset** (D-002). Never hard-code one;
   take it as a parameter.
6. Never fabricate data, URLs, metrics or download results. Functions that
   cannot produce a valid answer must raise or return an explicit
   unknown/flagged value -- never a plausible-looking number.
7. Pure functions, no network access in unit tests, fixed seeds, numpy-based.
8. Python 3.11, numpy 2.4, pandas 3.0, pyproj 3.7, rasterio 1.4, pytest.
   `tests/conftest.py` already puts `src/` on `sys.path`; import as
   `from crater import geometry as g`.
