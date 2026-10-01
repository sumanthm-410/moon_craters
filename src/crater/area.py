"""Usable survey area on the lunar sphere, from a raster validity mask.

Why a pixel count is not an area
--------------------------------
The project grid is south polar stereographic on the Moon 2000 sphere
(``config/project.yaml``, ``crater.body.Body.polar_stereographic_proj4``).  That
projection is conformal, so a pixel of side ``p`` projected metres always covers
``p**2`` *projected* square metres, but the patch of sphere underneath it is
smaller by the areal scale factor ``k**2 =
crater.geometry.area_scale_factor(lat)``::

    true_pixel_area = p**2 / area_scale_factor(lat_of_pixel_centre)

A naive ``n_valid_pixels * p**2`` therefore **overestimates** true surface area
by exactly ``k**2 - 1`` (weighted over the mask):

====================  ==============  ======================
latitude              ``k``           naive area is high by
====================  ==============  ======================
-90 (pole)            1.000000        0 %
-89                   1.000076        0.015 %
-86 (ROI)             1.001219        **0.244 %**
-85.9                 1.001281        0.256 %
-80                   1.007654        1.54 %
-70                   1.031091        6.31 %
-45                   1.171573        37.3 %
0 (equator)           2.0             300 %
====================  ==============  ======================

At the ROI latitude 0.244 % of a counting area is small but not negligible: it
biases every crater density and every R value by the same 0.244 %, in one
direction, and it grows without bound away from the pole.  INTERFACES.md rule 2
makes dividing by ``k**2`` mandatory, so every area in this module is a true
surface area.  :func:`naive_projected_area_m2` exists only so that the error can
be measured and reported, never to be used as a counting area.

Numerical method
----------------
Each pixel's true area is evaluated with the midpoint rule: ``k**2`` is taken at
the pixel centre and held constant across the pixel.  The per-pixel error is
``O(p**2 * d2(k^-2)/dy2)`` and the whole-mask error falls off as ``p**2``; the
residual error of a mask-derived area is dominated instead by the staircase
approximation of the region boundary (``O(p)`` in the worst case).
:func:`spherical_cap_area_m2` is the exact analytic reference that a mask-based
area must converge to as ``p -> 0``; ``tests/test_area.py`` exercises that
convergence.

All areas are **true spherical surface areas** in square metres unless the name
says ``km2``.  All distances are planimetric great-circle distances on the
sphere (INTERFACES.md rule 3); no terrain-surface lengthening is applied.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from scipy import ndimage

from . import geometry as g
from .body import MOON, Body

#: Square metres per square kilometre.
M2_PER_KM2: float = 1.0e6

#: Every area here is a true area on the sphere, not a projected-plane area.
AREA_KIND = "true_spherical_surface"

#: Every distance used by the edge rule is planimetric great-circle.
EDGE_DISTANCE_KIND = g.DIAMETER_DISTANCE_KIND


# --------------------------------------------------------------------------- #
# Analytic reference
# --------------------------------------------------------------------------- #
def spherical_cap_area_m2(colatitude_deg, body: Body = MOON) -> float:
    """Exact surface area of a spherical cap, ``2 pi R**2 (1 - cos theta)``.

    ``colatitude_deg`` is the angular radius of the cap measured from its pole
    (for a south-polar cap, ``90 + latitude_of_the_rim`` in degrees).  This is
    the closed-form reference that any mask-based area computation must converge
    to; it involves no projection and no pixels.

    Raises
    ------
    ValueError
        If the colatitude is outside [0, 180].
    """
    theta = np.asarray(colatitude_deg, dtype=float)
    if np.any(theta < 0.0) or np.any(theta > 180.0):
        raise ValueError("colatitude_deg must lie in [0, 180]")
    area = 2.0 * np.pi * body.radius_m**2 * (1.0 - np.cos(np.radians(theta)))
    return area if np.ndim(colatitude_deg) else float(area)


def colatitude_of_latitude_deg(lat_deg):
    """South-polar colatitude of a latitude: ``90 + lat`` degrees.

    The south pole has colatitude 0; the equator 90.  Returned as the same
    shape as the input.
    """
    return 90.0 + np.asarray(lat_deg, dtype=float)


# --------------------------------------------------------------------------- #
# The grid
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StereographicGrid:
    """A north-up, square-pixel raster grid in south polar stereographic metres.

    Parameters
    ----------
    x_origin, y_origin
        Projected coordinates of the **outer corner** of pixel (row 0, col 0),
        i.e. the upper-left corner of the grid, in metres.  ``y`` decreases with
        increasing row, matching the usual GIS/rasterio north-up convention.
    pixel_size_m
        Pixel side length in **projected** metres.  Pixels are square in the
        projected plane; their ground footprint shrinks by ``1/k``.
    height, width
        Number of rows and columns.
    lon_0, k0
        Projection parameters, passed through to :mod:`crater.geometry`.
    body
        Target body; the Moon 2000 sphere by default.

    Notes
    -----
    The ROI centre is deliberately not defaulted anywhere (DECISIONS.md D-002);
    a grid must be constructed explicitly from caller-supplied coordinates.
    """

    x_origin: float
    y_origin: float
    pixel_size_m: float
    height: int
    width: int
    lon_0: float = 0.0
    k0: float = 1.0
    body: Body = MOON
    _cache: dict = field(default_factory=dict, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.pixel_size_m <= 0:
            raise ValueError("pixel_size_m must be positive")
        if self.height < 1 or self.width < 1:
            raise ValueError("height and width must be at least 1")

    # -- construction ------------------------------------------------------ #
    @classmethod
    def centred_on_projected_point(
        cls,
        x_centre_m: float,
        y_centre_m: float,
        half_extent_m: float,
        pixel_size_m: float,
        *,
        lon_0: float = 0.0,
        k0: float = 1.0,
        body: Body = MOON,
    ) -> "StereographicGrid":
        """Grid of ``2*half_extent_m`` projected metres a side, centred on a point.

        The pixel count is rounded up so that the requested extent is fully
        covered, and the grid is re-centred on the requested point.
        """
        if half_extent_m <= 0:
            raise ValueError("half_extent_m must be positive")
        n = int(np.ceil(2.0 * half_extent_m / pixel_size_m))
        span = n * pixel_size_m
        return cls(
            x_origin=x_centre_m - span / 2.0,
            y_origin=y_centre_m + span / 2.0,
            pixel_size_m=pixel_size_m,
            height=n,
            width=n,
            lon_0=lon_0,
            k0=k0,
            body=body,
        )

    @classmethod
    def centred_on_lonlat(
        cls,
        lon_deg: float,
        lat_deg: float,
        half_extent_m: float,
        pixel_size_m: float,
        *,
        lon_0: float = 0.0,
        k0: float = 1.0,
        body: Body = MOON,
    ) -> "StereographicGrid":
        """Grid centred on a lon/lat point.  ``half_extent_m`` is projected metres."""
        x, y = g.forward(lon_deg, lat_deg, body=body, lon_0=lon_0, k0=k0)
        return cls.centred_on_projected_point(
            x, y, half_extent_m, pixel_size_m, lon_0=lon_0, k0=k0, body=body
        )

    # -- derived geometry -------------------------------------------------- #
    @property
    def shape(self) -> tuple[int, int]:
        return (self.height, self.width)

    @property
    def projected_pixel_area_m2(self) -> float:
        """Area of one pixel in the **projected** plane (not a surface area)."""
        return float(self.pixel_size_m) ** 2

    def pixel_centres_xy(self) -> tuple[np.ndarray, np.ndarray]:
        """2-D arrays of pixel-centre x and y in projected metres."""
        cached = self._cache.get("xy")
        if cached is None:
            p = float(self.pixel_size_m)
            xs = self.x_origin + (np.arange(self.width) + 0.5) * p
            ys = self.y_origin - (np.arange(self.height) + 0.5) * p
            cached = np.meshgrid(xs, ys)
            self._cache["xy"] = cached
        return cached

    def pixel_centres_lonlat(self) -> tuple[np.ndarray, np.ndarray]:
        """2-D arrays of pixel-centre longitude and latitude in degrees."""
        cached = self._cache.get("lonlat")
        if cached is None:
            x, y = self.pixel_centres_xy()
            cached = g.inverse(x, y, body=self.body, lon_0=self.lon_0, k0=self.k0)
            self._cache["lonlat"] = cached
        return cached

    def pixel_latitudes(self) -> np.ndarray:
        """2-D array of pixel-centre latitudes in degrees."""
        return self.pixel_centres_lonlat()[1]

    def pixel_colatitudes(self) -> np.ndarray:
        """2-D array of pixel-centre south-polar colatitudes in degrees."""
        return colatitude_of_latitude_deg(self.pixel_latitudes())

    def area_scale_factor(self) -> np.ndarray:
        """2-D array of ``k**2`` at each pixel centre."""
        cached = self._cache.get("k2")
        if cached is None:
            cached = g.area_scale_factor(self.pixel_latitudes(), k0=self.k0)
            self._cache["k2"] = cached
        return cached

    def point_scale_factor(self) -> np.ndarray:
        """2-D array of the linear scale factor ``k`` at each pixel centre."""
        cached = self._cache.get("k")
        if cached is None:
            cached = g.point_scale_factor(self.pixel_latitudes(), k0=self.k0)
            self._cache["k"] = cached
        return cached

    def pixel_true_areas_m2(self) -> np.ndarray:
        """2-D array of the **true surface area** of every pixel, square metres.

        ``projected_pixel_area / area_scale_factor(lat)``, midpoint rule.
        """
        cached = self._cache.get("true_area")
        if cached is None:
            cached = self.projected_pixel_area_m2 / self.area_scale_factor()
            self._cache["true_area"] = cached
        return cached


# --------------------------------------------------------------------------- #
# Mask assembly
# --------------------------------------------------------------------------- #
def _as_bool_mask(mask, shape: tuple[int, int], name: str) -> np.ndarray:
    arr = np.asarray(mask)
    if arr.shape != shape:
        raise ValueError(f"{name} has shape {arr.shape}, expected {shape}")
    if arr.dtype != bool:
        raise TypeError(
            f"{name} must be a boolean array; got dtype {arr.dtype}. "
            "Pass an explicit boolean mask so that nodata sentinels cannot be "
            "silently truthy."
        )
    return arr


def union_coverage(masks: Sequence[np.ndarray], shape: tuple[int, int] | None = None) -> np.ndarray:
    """Logical OR of several source-coverage masks: **overlap is counted once**.

    Mosaicked or overlapping source images (two NAC strips over the same ground,
    a repeat pass, two tiles sharing a seam) must contribute their shared ground
    exactly once.  Summing per-source areas double-counts the overlap; this
    reduces the sources to a single footprint first.

    Raises
    ------
    ValueError
        If ``masks`` is empty and no ``shape`` is given, or shapes disagree.
    """
    masks = list(masks)
    if not masks:
        if shape is None:
            raise ValueError("union_coverage needs at least one mask, or a shape")
        return np.zeros(shape, dtype=bool)
    if shape is None:
        shape = np.asarray(masks[0]).shape
    out = np.zeros(shape, dtype=bool)
    for i, m in enumerate(masks):
        out |= _as_bool_mask(m, shape, f"masks[{i}]")
    return out


def usable_mask(
    coverage_masks: Sequence[np.ndarray],
    *,
    nodata_masks: Sequence[np.ndarray] = (),
    excluded_masks: Sequence[np.ndarray] = (),
    shape: tuple[int, int] | None = None,
) -> np.ndarray:
    """Build the survey validity mask.

    ``usable = union(coverage) & ~union(nodata) & ~union(excluded)``

    Parameters
    ----------
    coverage_masks
        One boolean mask per source image, True where that source supplies
        pixels.  Overlapping sources are unioned, so shared ground is counted
        once.
    nodata_masks
        True where a pixel carries a nodata / fill value, is outside the
        source's valid footprint, or failed a quality gate.  Removed.
    excluded_masks
        True where terrain is deliberately excluded from the survey -- permanent
        shadow, slopes beyond the mappable limit, saturated illumination
        (DECISIONS.md D-005), unmeasurable terrain.  Removed.

        These pixels are excluded from the counting area **and** craters whose
        centres fall in them must be excluded from the counts, or the density is
        biased.  This module only computes the area; keeping the two consistent
        is the caller's job.
    shape
        Required only when ``coverage_masks`` is empty.
    """
    cov = union_coverage(coverage_masks, shape=shape)
    out = cov.copy()
    if len(nodata_masks):
        out &= ~union_coverage(nodata_masks, shape=cov.shape)
    if len(excluded_masks):
        out &= ~union_coverage(excluded_masks, shape=cov.shape)
    return out


# --------------------------------------------------------------------------- #
# Areas
# --------------------------------------------------------------------------- #
def true_area_m2(mask: np.ndarray, grid: StereographicGrid) -> float:
    """True spherical surface area of the True pixels of ``mask``, square metres.

    Each pixel contributes ``pixel_size**2 / area_scale_factor(lat)``.  This is
    the only area that may be used as a crater-counting area.
    """
    m = _as_bool_mask(mask, grid.shape, "mask")
    return float(np.sum(grid.pixel_true_areas_m2()[m]))


def true_area_km2(mask: np.ndarray, grid: StereographicGrid) -> float:
    """:func:`true_area_m2` expressed in square kilometres."""
    return true_area_m2(mask, grid) / M2_PER_KM2


def naive_projected_area_m2(mask: np.ndarray, grid: StereographicGrid) -> float:
    """``count * pixel_size**2`` -- the **wrong** answer, for error reporting only.

    This is a projected-plane area, not a surface area.  It exceeds
    :func:`true_area_m2` by the mask-weighted mean of ``k**2 - 1``: about
    0.244 % at 86 S, 1.54 % at 80 S, 6.3 % at 70 S, 300 % at the equator.  Never
    use it as a counting area; use it only to quantify the bias, as
    :func:`naive_area_relative_error` does.
    """
    m = _as_bool_mask(mask, grid.shape, "mask")
    return float(np.count_nonzero(m)) * grid.projected_pixel_area_m2


def naive_area_relative_error(mask: np.ndarray, grid: StereographicGrid) -> float:
    """Fractional overestimate of a naive pixel-count area, ``naive/true - 1``.

    Equals the ``k**2``-weighted mean distortion over the mask, so it is
    ~0.00244 for a mask at 86 S.  Returns NaN for an empty mask -- there is no
    error to report and no plausible number may be invented (INTERFACES.md
    rule 6).
    """
    true = true_area_m2(mask, grid)
    if true <= 0.0:
        return float("nan")
    return naive_projected_area_m2(mask, grid) / true - 1.0


def mask_from_colatitude(
    grid: StereographicGrid, max_colatitude_deg: float, min_colatitude_deg: float = 0.0
) -> np.ndarray:
    """Pixels whose **centre** lies in a south-polar colatitude annulus.

    With ``min_colatitude_deg = 0`` this is a spherical cap, whose exact area is
    :func:`spherical_cap_area_m2`.  Used to validate mask-based areas against
    the closed form, and as a crude ROI cut.
    """
    if not 0.0 <= min_colatitude_deg <= max_colatitude_deg <= 180.0:
        raise ValueError("require 0 <= min_colatitude <= max_colatitude <= 180")
    colat = grid.pixel_colatitudes()
    return (colat >= min_colatitude_deg) & (colat <= max_colatitude_deg)


# --------------------------------------------------------------------------- #
# Diameter-dependent usable area (the edge rule)
# --------------------------------------------------------------------------- #
#: Human-readable statement of the edge rule implemented below.
EDGE_RULE = (
    "A crater of diameter D is counted only if its centre lies at least D/2 of "
    "planimetric ground distance inside the survey boundary, so that the whole "
    "rim is inside the surveyed and non-excluded area. The counting area for "
    "diameter D is therefore the usable mask eroded by a ground buffer of D/2, "
    "and it shrinks monotonically with D."
)


def boundary_distance_m(
    grid: StereographicGrid,
    mask: np.ndarray,
    *,
    half_pixel_correction: bool = True,
) -> np.ndarray:
    """Planimetric ground distance from each usable pixel centre to the boundary.

    The survey boundary is the edge of the True region of ``mask``, which
    includes the outer edge of the grid: the grid is padded with one ring of
    unusable pixels before the distance transform, so a pixel next to the array
    edge is correctly treated as being next to the boundary rather than as
    interior.

    The Euclidean distance transform works in the **projected** plane, giving
    projected metres; each pixel's value is then divided by the local linear
    scale factor ``k`` to become a ground distance (INTERFACES.md rule 2).  That
    is exact in the limit of a short buffer and in error by at most the
    variation of ``k`` along the path -- below 0.03 % for the 1 km buffers of
    this project's largest craters at the ROI latitude, where ``k`` changes by
    ~1.5e-5 per km.

    With ``half_pixel_correction`` (the default) half a pixel is subtracted,
    because the transform measures centre-to-centre distance to the nearest
    unusable pixel whereas the boundary itself lies about half a pixel nearer.
    Unusable pixels get distance 0.

    Returns
    -------
    ndarray
        Same shape as ``mask``; ground metres, 0 outside the usable mask.
    """
    m = _as_bool_mask(mask, grid.shape, "mask")
    padded = np.pad(m, 1, mode="constant", constant_values=False)
    dist_px = ndimage.distance_transform_edt(padded)[1:-1, 1:-1]
    projected_m = dist_px * float(grid.pixel_size_m)
    if half_pixel_correction:
        projected_m = projected_m - 0.5 * float(grid.pixel_size_m)
    projected_m = np.where(m, np.maximum(projected_m, 0.0), 0.0)
    return projected_m / grid.point_scale_factor()


def usable_area_m2_by_diameter(
    grid: StereographicGrid,
    mask: np.ndarray,
    diameters_m,
    *,
    half_pixel_correction: bool = True,
) -> np.ndarray:
    """Diameter-dependent counting area ``A_i``, square metres.

    Implements :data:`EDGE_RULE`: for each requested diameter ``D``, the
    counting area is the true surface area of the pixels whose centre is at
    least ``D/2`` of ground distance inside the survey boundary.

    ``A_i`` is non-increasing in ``D`` by construction.  It reaches exactly 0.0
    once the buffer exceeds the inradius of the usable region; a zero area is
    returned as 0.0 and *not* clipped to something positive, so downstream code
    must treat such a bin as having no counting area at all rather than dividing
    by a fabricated number (INTERFACES.md rule 6).

    Parameters
    ----------
    diameters_m
        Scalar or array of crater diameters in metres (project range 20-1000 m,
        DECISIONS.md D-003).  Must be non-negative.

    Returns
    -------
    ndarray
        Same shape as ``diameters_m`` (0-d for a scalar), in square metres.
    """
    d = np.asarray(diameters_m, dtype=float)
    if np.any(~np.isfinite(d)) or np.any(d < 0.0):
        raise ValueError("diameters_m must be finite and non-negative")
    dist = boundary_distance_m(grid, mask, half_pixel_correction=half_pixel_correction)
    m = _as_bool_mask(mask, grid.shape, "mask")
    per_pixel = grid.pixel_true_areas_m2()
    flat_dist = dist[m]
    flat_area = per_pixel[m]
    out = np.empty(d.shape, dtype=float)
    for idx in np.ndindex(*d.shape) if d.ndim else [()]:
        keep = flat_dist >= (float(d[idx]) / 2.0)
        out[idx] = float(np.sum(flat_area[keep]))
    return out


def usable_area_km2_by_diameter(
    grid: StereographicGrid,
    mask: np.ndarray,
    diameters_m,
    *,
    half_pixel_correction: bool = True,
) -> np.ndarray:
    """:func:`usable_area_m2_by_diameter` in square kilometres."""
    return (
        usable_area_m2_by_diameter(
            grid, mask, diameters_m, half_pixel_correction=half_pixel_correction
        )
        / M2_PER_KM2
    )


def bin_representative_diameters_m(bin_low_m, bin_high_m, kind: str = "high") -> np.ndarray:
    """Diameter at which to evaluate ``A_i`` for a size bin.

    ``kind``:

    ``"high"``
        the bin's upper edge -- the **conservative** choice, and the default.
        Every crater in the bin then satisfies the edge rule over the whole
        returned area, so no crater is counted on ground that the area excludes.
    ``"geometric"``
        ``sqrt(low*high)``, the bin's plotting diameter.  Slightly larger area,
        but craters in the upper half of the bin can then be counted outside the
        area they are normalised by, which biases R upward.
    """
    low = np.asarray(bin_low_m, dtype=float)
    high = np.asarray(bin_high_m, dtype=float)
    if kind == "high":
        return high
    if kind == "geometric":
        return np.sqrt(low * high)
    raise ValueError(f"unknown kind {kind!r}; use 'high' or 'geometric'")


def crater_passes_edge_rule(
    grid: StereographicGrid,
    mask: np.ndarray,
    crater_x_m,
    crater_y_m,
    diameter_m,
    *,
    half_pixel_correction: bool = True,
) -> np.ndarray:
    """Whether each crater centre satisfies :data:`EDGE_RULE`.

    Centres are given in projected metres; a centre outside the grid, or on an
    unusable pixel, fails.  Use this to select the craters counted in the
    numerator so that they match the ``A_i`` used in the denominator.
    """
    x = np.atleast_1d(np.asarray(crater_x_m, dtype=float))
    y = np.atleast_1d(np.asarray(crater_y_m, dtype=float))
    d = np.broadcast_to(np.atleast_1d(np.asarray(diameter_m, dtype=float)), x.shape)
    dist = boundary_distance_m(grid, mask, half_pixel_correction=half_pixel_correction)
    p = float(grid.pixel_size_m)
    col = np.floor((x - grid.x_origin) / p).astype(int)
    row = np.floor((grid.y_origin - y) / p).astype(int)
    inside = (row >= 0) & (row < grid.height) & (col >= 0) & (col < grid.width)
    out = np.zeros(x.shape, dtype=bool)
    r = np.where(inside, row, 0)
    c = np.where(inside, col, 0)
    out[inside] = dist[r[inside], c[inside]] >= (d[inside] / 2.0)
    if np.ndim(crater_x_m) == 0 and np.ndim(crater_y_m) == 0:
        return bool(out[0])
    return out


__all__ = [
    "AREA_KIND",
    "EDGE_DISTANCE_KIND",
    "EDGE_RULE",
    "M2_PER_KM2",
    "StereographicGrid",
    "bin_representative_diameters_m",
    "boundary_distance_m",
    "colatitude_of_latitude_deg",
    "crater_passes_edge_rule",
    "mask_from_colatitude",
    "naive_area_relative_error",
    "naive_projected_area_m2",
    "spherical_cap_area_m2",
    "true_area_km2",
    "true_area_m2",
    "union_coverage",
    "usable_area_km2_by_diameter",
    "usable_area_m2_by_diameter",
    "usable_mask",
]
