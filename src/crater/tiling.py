"""Tiling of a south-polar-stereographic mosaic into detector tiles.

Why this module exists
----------------------
A crater detector sees a few hundred to a few thousand pixels at a time, while
the ROI mosaic is tens of thousands of pixels across.  The mosaic must therefore
be cut into overlapping tiles, and every detection made in a tile must be
mappable back to an unambiguous place on the Moon.  Two things make that
non-trivial on a polar product, and both are handled explicitly here:

1. **The projection is not the ground.**  The mosaic lives in south polar
   stereographic metres on the Moon 2000 sphere (``crater.body.MOON``).  A pixel
   is a fixed number of *projected* metres wide, but the ground length it covers
   is smaller, because the stereographic plane is stretched away from the pole.
   Per the standing project rule (INTERFACES.md point 2), a length in projected
   metres becomes a ground length by **dividing by the point scale factor k**
   (``crater.geometry.point_scale_factor``), and a projected area by ``k**2``.
   At the ROI latitude (~85.9 S) k ~ 1.0012, so a nominally "1 m/px" mosaic
   samples ~0.9988 ground metres per pixel, and a 1000 m crater spans ~1001.2
   pixels rather than 1000.  That 0.12% is small but it is not zero, and it is
   never silently dropped here: see :func:`ground_pixel_scale_m` and
   :func:`recommend_tile_size`.

2. **Tiles overlap on purpose.**  A crater bisected by a tile edge cannot be
   measured, so every tile carries a context margin on each side.  The margin
   means neighbouring tiles *share image data*.  That sharing is the single
   biggest leakage hazard for a geographic train/val/test split, which is why
   this module exposes both the full tile window and the tile's exclusive
   "core" window (see :class:`Tile`), and why :mod:`crater.splits` refuses to
   place two overlapping tiles in different splits.

Pixel coordinate convention
---------------------------
Pixel coordinates are *continuous*: an integer column/row value is a pixel
**edge**, and the centre of pixel ``i`` is at ``i + 0.5``.  This is the
``affine`` / GDAL convention and it is the only one under which the multiscale
mapping in :func:`level_pixel_to_full_pixel` is exact.  Helpers that take a
``centre`` flag add or remove the half-pixel for you; see
:data:`PIXEL_COORDINATE_CONVENTION`.

Distance kind
-------------
Every *ground* length produced or consumed here is planimetric great-circle on
the sphere, matching ``crater.geometry.DIAMETER_DISTANCE_KIND``.  Nothing in
this module applies a terrain-surface (slope-lengthened) correction.

Nothing in this module hard-codes an ROI centre or extent; the ROI is
deliberately unset (DECISIONS.md D-002) and arrives as a transform plus a
raster size.
"""
from __future__ import annotations

import math
import warnings as _warnings
from dataclasses import dataclass
from typing import Iterator, Sequence

import numpy as np
from affine import Affine

from . import geometry as _g
from .body import MOON, Body

__all__ = [
    "PIXEL_COORDINATE_CONVENTION",
    "EDGE_POLICIES",
    "DEFAULT_CONTEXT_FRACTION",
    "TileSizeWarning",
    "TileGrid",
    "Tile",
    "TileSizeRecommendation",
    "iter_tiles",
    "tile_list",
    "pixel_to_world",
    "world_to_pixel",
    "pixel_to_lonlat",
    "lonlat_to_pixel",
    "projected_to_ground_length",
    "ground_to_projected_length",
    "ground_pixel_scale_m",
    "level_scale_factor",
    "downsample_transform",
    "level_raster_size",
    "level_pixel_to_full_pixel",
    "full_pixel_to_level_pixel",
    "recover_mosaic_pixel",
    "recover_lonlat",
    "mosaic_pixel_to_tile_pixel",
    "lonlat_to_tile_pixel",
    "tile_corners_projected",
    "tile_bounds_projected",
    "tile_corners_lonlat",
    "tile_footprint_lonlat",
    "recommend_tile_size",
]

#: Human-readable statement of the pixel indexing convention used everywhere here.
PIXEL_COORDINATE_CONVENTION = (
    "continuous pixel coordinates; integer values are pixel EDGES and the "
    "centre of pixel i is i + 0.5"
)

#: How the right/bottom edge of the raster is handled.  See :class:`TileGrid`.
EDGE_POLICIES = ("partial", "shift")

#: Default context band, as a fraction of the largest target diameter, that must
#: remain visible on every side of a crater for a rim fit to be defensible.
DEFAULT_CONTEXT_FRACTION = 0.25


class TileSizeWarning(UserWarning):
    """Raised as a warning when a tile geometry cannot hold the target craters."""


# --------------------------------------------------------------------------- #
# Projected <-> ground lengths (INTERFACES.md point 2)
# --------------------------------------------------------------------------- #
def projected_to_ground_length(length_m, lat_deg, k0: float = 1.0):
    """Convert a length in **projected** metres to a ground (planimetric) length.

    The stereographic plane is stretched by the point scale factor ``k``, so a
    projected length must be *divided* by ``k``.  Ignoring this overstates
    lengths by ~0.12% at 85.9 S -- about 1.2 m on a 1000 m crater, which is
    larger than the measurement precision this survey claims.
    """
    k = _g.point_scale_factor(lat_deg, k0=k0)
    return np.asarray(length_m, float) / k if np.ndim(length_m) or np.ndim(lat_deg) \
        else float(length_m) / float(k)


def ground_to_projected_length(length_m, lat_deg, k0: float = 1.0):
    """Convert a ground (planimetric great-circle) length to projected metres.

    The inverse of :func:`projected_to_ground_length`: multiply by ``k``.  This
    is the direction that matters when sizing tiles, because a crater diameter
    is a *ground* quantity while a tile is measured in projected pixels.
    """
    k = _g.point_scale_factor(lat_deg, k0=k0)
    return np.asarray(length_m, float) * k if np.ndim(length_m) or np.ndim(lat_deg) \
        else float(length_m) * float(k)


def ground_pixel_scale_m(pixel_scale_m: float, lat_deg: float, k0: float = 1.0) -> float:
    """Ground metres covered by one pixel of a mosaic with ``pixel_scale_m``.

    ``pixel_scale_m`` is the *projected* pixel size (what a GeoTIFF's affine
    transform reports).  The ground sampling is ``pixel_scale_m / k`` and is
    therefore slightly finer than the nominal number away from the pole.
    """
    return float(projected_to_ground_length(pixel_scale_m, lat_deg, k0=k0))


# --------------------------------------------------------------------------- #
# Pixel <-> world helpers
# --------------------------------------------------------------------------- #
def _affine_coeffs(transform: Affine):
    if not isinstance(transform, Affine):
        raise TypeError(
            "transform must be an affine.Affine (rasterio's convention); "
            f"got {type(transform).__name__}"
        )
    if transform.determinant == 0.0:
        raise ValueError("transform is singular (determinant 0) and cannot be inverted")
    return (transform.a, transform.b, transform.c,
            transform.d, transform.e, transform.f)


def pixel_to_world(transform: Affine, col, row, *, centre: bool = False):
    """Map pixel coordinates to **projected metres** (x, y).

    Parameters
    ----------
    transform :
        The affine transform of the raster the ``col``/``row`` refer to.  For a
        pyramid level use :func:`downsample_transform`.
    col, row :
        Scalars or numpy arrays, in continuous pixel coordinates
        (:data:`PIXEL_COORDINATE_CONVENTION`).
    centre :
        If True, ``col``/``row`` are integer pixel *indices* and the centre of
        that pixel is returned (a half pixel is added before projecting).

    Notes
    -----
    This is deliberately implemented from the affine coefficients rather than
    ``transform * (col, row)`` so that whole arrays of detections can be
    transformed in one call, as the rest of the project is numpy-based.
    No lon/lat is involved: the output is in the projected CRS.
    """
    a, b, c, d, e, f = _affine_coeffs(transform)
    scalar = not (np.ndim(col) or np.ndim(row))
    cc = np.asarray(col, dtype=float)
    rr = np.asarray(row, dtype=float)
    if centre:
        cc = cc + 0.5
        rr = rr + 0.5
    x = a * cc + b * rr + c
    y = d * cc + e * rr + f
    return (float(x), float(y)) if scalar else (x, y)


def world_to_pixel(transform: Affine, x, y, *, centre: bool = False):
    """Map **projected metres** (x, y) to pixel coordinates; inverse of
    :func:`pixel_to_world`.

    With ``centre=True`` the returned values are the pixel indices whose
    centres sit at ``(x, y)``; i.e. ``world_to_pixel(t, *pixel_to_world(
    t, c, r, centre=True), centre=True)`` returns ``(c, r)`` exactly (to
    float64 round-off).  With ``centre=False`` both directions use pixel edges.
    The result is **not** rounded to integers -- sub-pixel detection centres are
    meaningful and rounding is the caller's decision.
    """
    inv = ~transform
    a, b, c, d, e, f = _affine_coeffs(inv)
    scalar = not (np.ndim(x) or np.ndim(y))
    xx = np.asarray(x, dtype=float)
    yy = np.asarray(y, dtype=float)
    col = a * xx + b * yy + c
    row = d * xx + e * yy + f
    if centre:
        col = col - 0.5
        row = row - 0.5
    return (float(col), float(row)) if scalar else (col, row)


def pixel_to_lonlat(transform: Affine, col, row, *, centre: bool = False,
                    body: Body = MOON, lon_0: float = 0.0, k0: float = 1.0):
    """Pixel coordinates -> (lon, lat) in degrees.

    Composes :func:`pixel_to_world` with ``crater.geometry.inverse``; the
    projection parameters must match those the mosaic was built with.
    """
    x, y = pixel_to_world(transform, col, row, centre=centre)
    return _g.inverse(x, y, body=body, lon_0=lon_0, k0=k0)


def lonlat_to_pixel(transform: Affine, lon, lat, *, centre: bool = False,
                    body: Body = MOON, lon_0: float = 0.0, k0: float = 1.0):
    """(lon, lat) in degrees -> pixel coordinates.

    Composes ``crater.geometry.forward`` with :func:`world_to_pixel`.  This is
    the only sanctioned way to place a catalogue crater on the mosaic; never
    scale degrees to metres with a planar factor (INTERFACES.md point 1).
    """
    x, y = _g.forward(lon, lat, body=body, lon_0=lon_0, k0=k0)
    return world_to_pixel(transform, x, y, centre=centre)


# --------------------------------------------------------------------------- #
# Multiscale pyramid arithmetic
# --------------------------------------------------------------------------- #
def level_scale_factor(level: int) -> int:
    """Downsample factor of pyramid level ``level``, i.e. ``2 ** level``.

    Level 0 is the full-resolution mosaic.  Powers of two are used so the
    mapping between levels is exact in binary floating point -- a non-dyadic
    pyramid would make the coordinate recovery below inexact.
    """
    if not isinstance(level, (int, np.integer)) or isinstance(level, bool):
        raise TypeError(f"level must be an int, got {type(level).__name__}")
    if level < 0:
        raise ValueError(f"pyramid level must be >= 0, got {level}")
    return 1 << int(level)


def downsample_transform(transform: Affine, level: int) -> Affine:
    """Affine transform of pyramid level ``level`` of a raster.

    A level-L pixel covers ``2**L`` full-resolution pixels per side, so the
    level-L transform is the base transform composed with a scale of ``2**L``.
    The origin is unchanged, which is what keeps :func:`level_pixel_to_full_pixel`
    consistent with it.
    """
    f = level_scale_factor(level)
    return transform @ Affine.scale(float(f))


def level_raster_size(width: int, height: int, level: int) -> tuple[int, int]:
    """Size in pixels of pyramid level ``level`` of a ``width`` x ``height`` raster.

    Uses ceiling division, the GDAL/COG overview convention: a partially filled
    coarse pixel still exists and still contains data, so truncating would drop
    a strip of the mosaic.
    """
    f = level_scale_factor(level)
    if width <= 0 or height <= 0:
        raise ValueError("raster width and height must be positive")
    return (-(-int(width) // f), -(-int(height) // f))


def level_pixel_to_full_pixel(col, row, level: int):
    """Continuous level-L pixel coordinates -> continuous level-0 coordinates.

    Exact, because the mapping is multiplication by ``2**L``::

        col_0 = col_L * 2**L

    This is why the continuous (edge-based) convention matters: the *edge* of a
    level-L pixel coincides exactly with an edge of a level-0 pixel, whereas the
    *centre* of a level-L pixel sits at the centre of a 2**L x 2**L block of
    level-0 pixels, i.e. on a half-integer level-0 coordinate.  Converting a
    level-L detection index ``i`` by hand as ``i * 2**L`` silently reports the
    block's top-left corner instead of its centre; pass ``i + 0.5`` (the
    detection's continuous coordinate) to get the physically correct point.
    """
    f = float(level_scale_factor(level))
    scalar = not (np.ndim(col) or np.ndim(row))
    cc = np.asarray(col, float) * f
    rr = np.asarray(row, float) * f
    return (float(cc), float(rr)) if scalar else (cc, rr)


def full_pixel_to_level_pixel(col, row, level: int):
    """Continuous level-0 pixel coordinates -> continuous level-L coordinates.

    Exact inverse of :func:`level_pixel_to_full_pixel` (division by a power of
    two is exact in binary floating point for all finite inputs).
    """
    f = float(level_scale_factor(level))
    scalar = not (np.ndim(col) or np.ndim(row))
    cc = np.asarray(col, float) / f
    rr = np.asarray(row, float) / f
    return (float(cc), float(rr)) if scalar else (cc, rr)


def recover_mosaic_pixel(col_in_tile, row_in_tile, tile_col_off: int,
                         tile_row_off: int, level: int):
    """Recover full-resolution mosaic pixel coordinates from a tile detection.

    Parameters
    ----------
    col_in_tile, row_in_tile :
        Continuous pixel coordinates of the detection *within its tile*, in
        level-L pixels (the units the detector actually works in).  Pass
        ``index + 0.5`` for a pixel centre.
    tile_col_off, tile_row_off :
        The tile's offset **in level-L pixels** -- exactly what
        :class:`Tile` carries for a grid built at that level.
    level :
        Pyramid level the tile was cut from; downsample factor ``2 ** level``.

    Returns
    -------
    (col_full, row_full)
        Continuous level-0 mosaic pixel coordinates.

    Notes
    -----
    The composition is ``(off + p) * 2**L``, applied in that order.  Adding the
    offset *after* scaling (a common bug) is wrong by ``off * (2**L - 1)``
    pixels, which at level 3 and a 512 px tile is thousands of pixels -- tens of
    kilometres on the ground, enough to attribute a crater to the wrong massif.
    """
    f = float(level_scale_factor(level))
    scalar = not (np.ndim(col_in_tile) or np.ndim(row_in_tile))
    cc = (np.asarray(tile_col_off, float) + np.asarray(col_in_tile, float)) * f
    rr = (np.asarray(tile_row_off, float) + np.asarray(row_in_tile, float)) * f
    return (float(cc), float(rr)) if scalar else (cc, rr)


def mosaic_pixel_to_tile_pixel(col_full, row_full, tile_col_off: int,
                               tile_row_off: int, level: int):
    """Exact inverse of :func:`recover_mosaic_pixel` (for round-trip checks)."""
    f = float(level_scale_factor(level))
    scalar = not (np.ndim(col_full) or np.ndim(row_full))
    cc = np.asarray(col_full, float) / f - np.asarray(tile_col_off, float)
    rr = np.asarray(row_full, float) / f - np.asarray(tile_row_off, float)
    return (float(cc), float(rr)) if scalar else (cc, rr)


def recover_lonlat(col_in_tile, row_in_tile, tile_col_off: int, tile_row_off: int,
                   level: int, base_transform: Affine, *, body: Body = MOON,
                   lon_0: float = 0.0, k0: float = 1.0):
    """Recover (lon, lat) for a detection in a level-L tile.

    ``base_transform`` is the **level-0** mosaic transform; the level scaling is
    applied to the pixel coordinates rather than to the transform so that one
    authoritative transform describes the whole pyramid.  Equivalent to using
    ``downsample_transform(base_transform, level)`` on the level-L coordinates,
    and that equivalence is covered by the tests.
    """
    col_full, row_full = recover_mosaic_pixel(
        col_in_tile, row_in_tile, tile_col_off, tile_row_off, level)
    return pixel_to_lonlat(base_transform, col_full, row_full,
                           body=body, lon_0=lon_0, k0=k0)


def lonlat_to_tile_pixel(lon, lat, tile_col_off: int, tile_row_off: int, level: int,
                         base_transform: Affine, *, body: Body = MOON,
                         lon_0: float = 0.0, k0: float = 1.0):
    """Inverse of :func:`recover_lonlat`: (lon, lat) -> level-L in-tile pixels."""
    col_full, row_full = lonlat_to_pixel(base_transform, lon, lat,
                                         body=body, lon_0=lon_0, k0=k0)
    return mosaic_pixel_to_tile_pixel(col_full, row_full, tile_col_off,
                                      tile_row_off, level)


# --------------------------------------------------------------------------- #
# Tiles
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Tile:
    """One detector window cut from a mosaic (or from one of its pyramid levels).

    Attributes
    ----------
    tile_id :
        Deterministic, sortable identifier ``"<scale_label>_r<row>_c<col>"``.
        It encodes the scale label and the offsets, so two tiles from different
        pyramid levels can never collide, and sorting ids sorts tiles in
        raster order.
    level, scale_label :
        Pyramid level (downsample ``2**level``) and its human label.
    col_off, row_off :
        Offset of the tile window in **this level's** pixels.
    width, height :
        Actual size of the window read from the raster.  For an edge tile under
        the ``"partial"`` policy this is smaller than ``TileGrid.tile_px``.
    transform :
        Affine transform of the tile itself, mapping in-tile level-L pixel
        coordinates to projected metres.
    partial :
        True when the window was clipped by the raster edge.  Partial tiles are
        **emitted, never dropped** -- see :class:`TileGrid`.
    pad_cols, pad_rows :
        How many columns/rows of padding would be needed to bring the window up
        to the nominal tile size.  Zero for interior tiles.  Reported rather
        than applied, because what to pad *with* (zeros, reflection, nodata)
        changes detector behaviour and is not this module's decision.
    core_col_off, core_row_off, core_width, core_height :
        The tile's **exclusive** window, in this level's pixels.  Core windows
        of all tiles in a grid tile the raster exactly: they are disjoint and
        their union is the whole raster.  A detection should be attributed to
        the tile whose core contains its centre; that makes cross-tile
        duplicate suppression a lookup rather than a geometric merge.
    """

    tile_id: str
    level: int
    scale_label: str
    col_off: int
    row_off: int
    width: int
    height: int
    transform: Affine
    partial: bool
    pad_cols: int
    pad_rows: int
    core_col_off: int
    core_row_off: int
    core_width: int
    core_height: int

    @property
    def col_stop(self) -> int:
        """Exclusive right edge of the window, in this level's pixels."""
        return self.col_off + self.width

    @property
    def row_stop(self) -> int:
        """Exclusive bottom edge of the window, in this level's pixels."""
        return self.row_off + self.height

    @property
    def core_col_stop(self) -> int:
        return self.core_col_off + self.core_width

    @property
    def core_row_stop(self) -> int:
        return self.core_row_off + self.core_height

    @property
    def window(self) -> tuple[int, int, int, int]:
        """``(col_off, row_off, width, height)`` -- rasterio ``Window`` order."""
        return (self.col_off, self.row_off, self.width, self.height)

    def contains_core(self, col_in_tile: float, row_in_tile: float) -> bool:
        """Is an in-tile continuous coordinate inside this tile's exclusive core?"""
        c = self.col_off + float(col_in_tile)
        r = self.row_off + float(row_in_tile)
        return (self.core_col_off <= c < self.core_col_stop
                and self.core_row_off <= r < self.core_row_stop)


@dataclass(frozen=True)
class TileGrid:
    """A tiling scheme for one mosaic, at one pyramid level.

    Parameters
    ----------
    tile_px :
        Side length of the window handed to the detector, in pixels.  The grid
        is square because detector backbones are.
    margin_px :
        Context band on **each** side of the tile.  Consecutive tiles therefore
        overlap by ``2 * margin_px`` and the grid advances by
        ``stride_px = tile_px - 2 * margin_px``.  The margin exists so a crater
        near a tile edge is still seen whole by at least one tile; it is the
        reason neighbouring tiles share image data, and hence the reason
        :mod:`crater.splits` must treat adjacent tiles as correlated.
    transform :
        ``affine.Affine`` of the raster at this level, in projected metres on
        the south polar stereographic CRS of ``crater.body``.  Pixels must be
        square (the whole project assumes one pixel scale).
    scale_label :
        Label for this pyramid level, e.g. ``"L0"``.  Defaults to ``f"L{level}"``.
        It is part of every ``tile_id``, which is what keeps ids unique across a
        multiscale dataset.
    level :
        Pyramid level; the raster this grid applies to is downsampled by
        ``2 ** level`` from the full mosaic.
    pixel_scale_m :
        Pixel size in **projected** metres.  Derived from ``transform`` when not
        given; when given it is cross-checked against the transform and a
        mismatch raises, because a silently wrong pixel scale would corrupt
        every diameter in the survey.
    edge_policy :
        How the right/bottom raster edge is handled.  Partial edge tiles are
        **never silently dropped**; the choice is only how they are expressed:

        ``"partial"`` (default)
            Offsets are a uniform ``stride_px`` lattice.  The final tile in a
            row/column is clipped to the raster and emitted with
            ``partial=True`` and the ``pad_cols``/``pad_rows`` it would need to
            reach ``tile_px``.  Padding is reported, not applied.  Chosen as the
            default because a uniform stride makes the core-window partition
            provable and independent of the raster size.
        ``"shift"``
            The final offset is clamped to ``size - tile_px`` so the window is
            full size, at the cost of a larger-than-nominal overlap at the edge.
            A tile is then partial only when the raster itself is smaller than
            one tile.  Use this when the detector cannot accept a padded input.

    Notes
    -----
    This class describes geometry only; it never reads pixels and never touches
    the network.
    """

    tile_px: int
    margin_px: int
    transform: Affine
    scale_label: str | None = None
    level: int = 0
    pixel_scale_m: float | None = None
    edge_policy: str = "partial"

    #: Relative tolerance for the square-pixel and pixel-scale cross-checks.
    scale_tol: float = 1e-9

    def __post_init__(self) -> None:
        if int(self.tile_px) != self.tile_px or self.tile_px < 1:
            raise ValueError(f"tile_px must be a positive int, got {self.tile_px!r}")
        if int(self.margin_px) != self.margin_px or self.margin_px < 0:
            raise ValueError(
                f"margin_px must be a non-negative int, got {self.margin_px!r}")
        if self.edge_policy not in EDGE_POLICIES:
            raise ValueError(
                f"edge_policy must be one of {EDGE_POLICIES}, got {self.edge_policy!r}")
        if self.tile_px - 2 * self.margin_px <= 0:
            raise ValueError(
                f"margin_px={self.margin_px} leaves no stride for tile_px="
                f"{self.tile_px}: 2*margin must be < tile_px, otherwise the grid "
                "cannot advance and tiling would never terminate"
            )
        level = self.level
        if not isinstance(level, (int, np.integer)) or level < 0:
            raise ValueError(f"level must be a non-negative int, got {level!r}")
        a, b, c, d, e, f = _affine_coeffs(self.transform)
        sx = math.hypot(a, d)
        sy = math.hypot(b, e)
        if sx <= 0 or sy <= 0:
            raise ValueError("transform has a zero-length pixel axis")
        if abs(sx - sy) > self.scale_tol * max(sx, sy):
            raise ValueError(
                f"non-square pixels (x scale {sx!r}, y scale {sy!r}); the project "
                "assumes a single pixel scale in projected metres"
            )
        object.__setattr__(self, "level", int(level))
        object.__setattr__(self, "tile_px", int(self.tile_px))
        object.__setattr__(self, "margin_px", int(self.margin_px))
        if self.pixel_scale_m is None:
            object.__setattr__(self, "pixel_scale_m", float(sx))
        else:
            ps = float(self.pixel_scale_m)
            if ps <= 0:
                raise ValueError(f"pixel_scale_m must be positive, got {ps}")
            if abs(ps - sx) > self.scale_tol * max(ps, sx):
                raise ValueError(
                    f"pixel_scale_m={ps} disagrees with the transform's pixel size "
                    f"{sx}; refusing to guess which is right"
                )
            object.__setattr__(self, "pixel_scale_m", ps)
        if self.scale_label is None:
            object.__setattr__(self, "scale_label", f"L{self.level}")
        elif not str(self.scale_label).strip():
            raise ValueError("scale_label must be a non-empty string")

    # -- derived geometry -------------------------------------------------- #
    @property
    def stride_px(self) -> int:
        """Advance between consecutive tile offsets: ``tile_px - 2*margin_px``."""
        return self.tile_px - 2 * self.margin_px

    @property
    def overlap_px(self) -> int:
        """Nominal overlap between neighbouring tiles: ``2 * margin_px``."""
        return 2 * self.margin_px

    @property
    def tile_span_projected_m(self) -> float:
        """Side length of a full tile in **projected** metres."""
        return self.tile_px * float(self.pixel_scale_m)

    def tile_span_ground_m(self, lat_deg: float, k0: float = 1.0) -> float:
        """Side length of a full tile as a **ground** length at ``lat_deg``.

        Smaller than :attr:`tile_span_projected_m` by the factor ``k``: at 1 m/px
        a 1024 px tile spans 1024 projected m but only ~1022.8 ground m at
        85.9 S.  Reported separately because the two are routinely conflated.
        """
        return float(projected_to_ground_length(self.tile_span_projected_m,
                                                lat_deg, k0=k0))

    def at_level(self, level: int) -> "TileGrid":
        """The same tiling scheme applied to pyramid level ``level``.

        ``tile_px`` and ``margin_px`` are kept (the detector input size does not
        change with scale); the transform and pixel scale are coarsened by
        ``2 ** (level - self.level)``.  Only coarsening is supported: a level
        finer than this grid's would require data that does not exist.
        """
        delta = int(level) - self.level
        if delta < 0:
            raise ValueError(
                f"cannot refine from level {self.level} to {level}; pyramid levels "
                "only coarsen (level 0 is full resolution)"
            )
        return TileGrid(
            tile_px=self.tile_px,
            margin_px=self.margin_px,
            transform=downsample_transform(self.transform, delta),
            scale_label=f"L{int(level)}",
            level=int(level),
            pixel_scale_m=None,
            edge_policy=self.edge_policy,
            scale_tol=self.scale_tol,
        )

    # -- enumeration -------------------------------------------------------- #
    def iter_tiles(self, width: int, height: int) -> Iterator[Tile]:
        """Enumerate tiles over a ``width`` x ``height`` raster at this level."""
        return iter_tiles(self, width, height)

    def tiles(self, width: int, height: int) -> tuple[Tile, ...]:
        """Materialised :meth:`iter_tiles`."""
        return tile_list(self, width, height)

    def recommend_tile_size(self, latitude_deg: float, **kwargs
                            ) -> "TileSizeRecommendation":
        """:func:`recommend_tile_size` for this grid's pixel scale and tile size."""
        kwargs.setdefault("proposed_tile_px", self.tile_px)
        kwargs.setdefault("context_fraction", DEFAULT_CONTEXT_FRACTION)
        return recommend_tile_size(float(self.pixel_scale_m), latitude_deg, **kwargs)


def _axis_offsets(size: int, tile_px: int, stride: int, edge_policy: str) -> list[int]:
    """Tile offsets along one axis.  See :class:`TileGrid` for the policies."""
    if int(size) != size or size < 1:
        raise ValueError(f"raster size must be a positive int, got {size!r}")
    size = int(size)
    if size <= tile_px:
        # One window covers the whole axis; it is partial iff size < tile_px.
        return [0]
    if edge_policy == "shift":
        limit = size - tile_px
        offs = [o for o in range(0, size, stride) if o <= limit]
        if not offs:
            offs = [0]
        if offs[-1] != limit:
            offs.append(limit)
        return offs
    return list(range(0, size, stride))


def _axis_extents(size: int, tile_px: int, offs: Sequence[int]) -> list[int]:
    return [min(tile_px, size - o) for o in offs]


def _core_bounds(size: int, offs: Sequence[int], extents: Sequence[int]
                 ) -> list[tuple[int, int]]:
    """Exclusive-core start/stop for each offset along one axis.

    The boundary between consecutive tiles is placed at the midpoint of their
    overlap, which (a) reduces to ``offset + margin`` for a uniform-stride
    interior grid, (b) still works for the clipped final tile and for the
    ``"shift"`` policy's irregular last offset, and (c) guarantees the cores are
    disjoint, non-empty, contained in their tiles, and cover ``[0, size)``
    exactly.  Property (c) is what makes cross-tile duplicate handling exact
    instead of heuristic.
    """
    n = len(offs)
    starts = [0] * n
    for i in range(1, n):
        prev_stop = offs[i - 1] + extents[i - 1]
        if offs[i] >= prev_stop:
            raise ValueError(
                f"tiles at offsets {offs[i-1]} and {offs[i]} do not overlap; "
                "a gap would leave mosaic pixels with no owning tile"
            )
        starts[i] = (offs[i] + prev_stop) // 2
    stops = [starts[i + 1] for i in range(n - 1)] + [size]
    return list(zip(starts, stops))


def iter_tiles(grid: TileGrid, width: int, height: int) -> Iterator[Tile]:
    """Enumerate the tiles of ``grid`` over a raster of ``width`` x ``height``.

    ``width``/``height`` are in the pixels of ``grid.level``; use
    :func:`level_raster_size` to get them from the full-resolution size.

    Yields
    ------
    Tile
        In raster order (rows outer, columns inner).  Every mosaic pixel is in
        the exclusive core of exactly one yielded tile, and partial edge tiles
        are yielded with ``partial=True`` rather than dropped.
    """
    stride = grid.stride_px
    col_offs = _axis_offsets(width, grid.tile_px, stride, grid.edge_policy)
    row_offs = _axis_offsets(height, grid.tile_px, stride, grid.edge_policy)
    col_ext = _axis_extents(int(width), grid.tile_px, col_offs)
    row_ext = _axis_extents(int(height), grid.tile_px, row_offs)
    col_core = _core_bounds(int(width), col_offs, col_ext)
    row_core = _core_bounds(int(height), row_offs, row_ext)

    for j, (row_off, h) in enumerate(zip(row_offs, row_ext)):
        for i, (col_off, w) in enumerate(zip(col_offs, col_ext)):
            pad_cols = grid.tile_px - w
            pad_rows = grid.tile_px - h
            cs, ce = col_core[i]
            rs, re = row_core[j]
            yield Tile(
                tile_id=f"{grid.scale_label}_r{row_off:06d}_c{col_off:06d}",
                level=grid.level,
                scale_label=str(grid.scale_label),
                col_off=col_off,
                row_off=row_off,
                width=w,
                height=h,
                transform=grid.transform @ Affine.translation(col_off, row_off),
                partial=bool(pad_cols or pad_rows),
                pad_cols=pad_cols,
                pad_rows=pad_rows,
                core_col_off=cs,
                core_row_off=rs,
                core_width=ce - cs,
                core_height=re - rs,
            )


def tile_list(grid: TileGrid, width: int, height: int) -> tuple[Tile, ...]:
    """Materialised :func:`iter_tiles`."""
    return tuple(iter_tiles(grid, width, height))


# --------------------------------------------------------------------------- #
# Tile footprints
# --------------------------------------------------------------------------- #
def tile_corners_projected(tile: Tile) -> tuple[tuple[float, float], ...]:
    """The four corners of a tile in **projected metres**, upper-left first.

    Order is upper-left, upper-right, lower-right, lower-left in pixel space,
    which is a closed ring once repeated.  All four corners are computed from
    the tile's own transform rather than from an axis-aligned bounding box, so a
    rotated mosaic is handled correctly.
    """
    w, h = float(tile.width), float(tile.height)
    pts = [(0.0, 0.0), (w, 0.0), (w, h), (0.0, h)]
    return tuple(pixel_to_world(tile.transform, c, r) for c, r in pts)


def tile_bounds_projected(tile: Tile) -> tuple[float, float, float, float]:
    """Axis-aligned ``(x_min, y_min, x_max, y_max)`` of a tile, projected metres."""
    xs, ys = zip(*tile_corners_projected(tile))
    return (min(xs), min(ys), max(xs), max(ys))


def tile_corners_lonlat(tile: Tile, *, body: Body = MOON, lon_0: float = 0.0,
                        k0: float = 1.0) -> tuple[tuple[float, float], ...]:
    """The four corners of a tile as ``(lon, lat)`` degree pairs."""
    return tuple(_g.inverse(x, y, body=body, lon_0=lon_0, k0=k0)
                 for x, y in tile_corners_projected(tile))


def tile_footprint_lonlat(tile: Tile, n_per_edge: int = 8, *, body: Body = MOON,
                          lon_0: float = 0.0, k0: float = 1.0):
    """Densified tile outline in lon/lat, as closed ``(lon[], lat[])`` arrays.

    A straight line in the projected plane is **not** a straight line in
    lon/lat: near the pole a tile edge is an arc, and a four-corner polygon can
    miss or include ground that the tile does not or does cover.  Sampling
    ``n_per_edge`` points along each edge keeps that error bounded.  Footprints
    used for the split geometry are kept in projected metres precisely to avoid
    this issue (see :mod:`crater.splits`); this function exists for export and
    plotting, where lon/lat is expected.
    """
    if n_per_edge < 2:
        raise ValueError("n_per_edge must be >= 2 to sample an edge")
    w, h = float(tile.width), float(tile.height)
    t = np.linspace(0.0, 1.0, int(n_per_edge), endpoint=False)
    cols = np.concatenate([t * w, np.full_like(t, w), (1.0 - t) * w, np.zeros_like(t)])
    rows = np.concatenate([np.zeros_like(t), t * h, np.full_like(t, h), (1.0 - t) * h])
    cols = np.append(cols, 0.0)
    rows = np.append(rows, 0.0)
    x, y = pixel_to_world(tile.transform, cols, rows)
    return _g.inverse(x, y, body=body, lon_0=lon_0, k0=k0)


# --------------------------------------------------------------------------- #
# Tile size vs the approved diameter range
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TileSizeRecommendation:
    """Structured answer to "is this tile big enough for the target craters?".

    Nothing here resizes anything.  The caller decides; this object records what
    the geometry demands and, when the demand is not met, by how much and what
    the alternatives cost.

    Attributes
    ----------
    pixel_scale_projected_m, pixel_scale_ground_m :
        Projected pixel size as given, and the ground sampling it corresponds to
        at ``latitude_deg`` (``projected / k``).
    scale_factor_k :
        ``crater.geometry.point_scale_factor(latitude_deg)``.
    max_diameter_m, min_diameter_m :
        The approved target range (DECISIONS.md D-003: 20 m to 1000 m), as
        **ground** planimetric diameters.
    max_diameter_px, min_diameter_px :
        Those diameters in pixels, ``D * k / pixel_scale_projected_m``.  Note
        the multiplication by ``k``: the projected image of a ground circle is
        inflated, so a crater occupies *more* pixels than ``D / pixel_scale``.
    context_fraction :
        Context required on each side of a crater, as a fraction of its diameter.
    required_margin_px :
        Margin needed so that a crater whose centre lies anywhere in a tile's
        exclusive core still fits inside that tile together with its context:
        ``ceil(max_diameter_px * (0.5 + context_fraction))``.
    required_core_px :
        Minimum stride, i.e. how far the grid must advance per tile.  Defaults to
        one maximum diameter, so tiling is not dominated by overlap.
    required_tile_px :
        ``required_core_px + 2 * required_margin_px`` -- the honest answer.
    recommended_tile_px :
        ``required_tile_px`` rounded up per the ``round_to`` rule.
    proposed_tile_px :
        The tile size being checked, if one was given.
    fits :
        Whether ``proposed_tile_px`` meets ``required_tile_px``.
    shortfall_px :
        ``max(0, required_tile_px - proposed_tile_px)``.
    largest_diameter_that_fits_m :
        Largest ground diameter the proposed tile can hold with full context and
        the required core, at this pixel scale and latitude.
    min_level_that_fits :
        Smallest pyramid level at which ``required_tile_px`` drops to
        ``proposed_tile_px`` or below, or ``None`` if the proposed tile cannot
        work at any level up to ``max_level_searched``.
    min_diameter_px_at_min_level :
        How many pixels the *smallest* target crater would span at that level.
        This is the cost of the coarser level and it is usually where the
        20 m - 1000 m range breaks: one tile size and one scale frequently
        cannot serve both ends of the range.
    min_pixels_across :
        Floor for a measurable rim (config/project.yaml
        ``diameters.min_pixels_across``, proposed 8 px).
    smallest_crater_resolved, smallest_crater_resolved_at_min_level :
        Whether ``min_diameter_px`` clears that floor, at this level and at
        ``min_level_that_fits``.
    warnings :
        Every problem found, as plain sentences.  Empty means no problem found.
    ok :
        True only when the proposed tile fits **and** the smallest crater is
        resolved at this level.
    message :
        One-paragraph human summary, suitable for STATUS.md.
    """

    pixel_scale_projected_m: float
    pixel_scale_ground_m: float
    latitude_deg: float
    scale_factor_k: float
    max_diameter_m: float
    min_diameter_m: float
    max_diameter_px: float
    min_diameter_px: float
    context_fraction: float
    required_margin_px: int
    required_core_px: int
    required_tile_px: int
    recommended_tile_px: int
    proposed_tile_px: int | None
    proposed_tile_span_projected_m: float | None
    proposed_tile_span_ground_m: float | None
    fits: bool
    shortfall_px: int
    largest_diameter_that_fits_m: float
    min_level_that_fits: int | None
    pixel_scale_at_min_level_m: float | None
    min_diameter_px_at_min_level: float | None
    min_pixels_across: float
    smallest_crater_resolved: bool
    smallest_crater_resolved_at_min_level: bool | None
    warnings: tuple[str, ...]
    ok: bool
    message: str


def _round_up(value: int, round_to) -> int:
    if round_to is None or round_to == "none":
        return int(value)
    if round_to == "pow2":
        return 1 << max(0, int(math.ceil(math.log2(max(1, int(value))))))
    m = int(round_to)
    if m < 1:
        raise ValueError(f"round_to multiple must be >= 1, got {round_to!r}")
    return -(-int(value) // m) * m


def recommend_tile_size(pixel_scale_m: float, latitude_deg: float, *,
                        max_diameter_m: float = 1000.0,
                        min_diameter_m: float = 20.0,
                        context_fraction: float = DEFAULT_CONTEXT_FRACTION,
                        proposed_tile_px: int | None = None,
                        required_core_px: int | None = None,
                        min_pixels_across: float = 8.0,
                        round_to: int | str | None = "pow2",
                        k0: float = 1.0,
                        pixel_scale_is_projected: bool = True,
                        max_level_searched: int = 8,
                        emit_warning: bool = True) -> TileSizeRecommendation:
    """Report the tile size needed to hold the largest target crater plus context.

    The approved target range is 20 m to 1000 m (DECISIONS.md D-003).  Those are
    **ground** planimetric diameters, while a tile is measured in projected
    pixels, so the conversion goes through the point scale factor: a ground
    diameter ``D`` at latitude ``lat`` images to ``D * k(lat)`` projected metres
    and hence ``D * k / pixel_scale`` pixels (INTERFACES.md point 2, applied in
    the ground-to-projected direction).

    The requirement implemented is: *a crater whose centre lies anywhere in a
    tile's exclusive core must fit entirely inside that tile, with a context
    band of ``context_fraction * D`` still visible on every side.*  Worst case
    the centre sits on the core boundary, so

        required_margin_px = ceil(D_px * (0.5 + context_fraction))
        required_tile_px   = required_core_px + 2 * required_margin_px

    with ``required_core_px`` defaulting to one full ``D_px`` so that the grid
    advances by at least one maximum crater per tile.

    This surfaces a real constraint rather than hiding it.  At ~1 m/px a 1024 px
    tile spans ~1 km of projected extent, so a 1000 m crater does **not** fit --
    not with context, and barely without.  The function never resizes anything:
    it returns a :class:`TileSizeRecommendation`, and (unless
    ``emit_warning=False``) also raises a :class:`TileSizeWarning` so the
    problem cannot pass unnoticed in a pipeline log.

    Parameters
    ----------
    pixel_scale_m :
        Pixel size.  Interpreted as **projected** metres unless
        ``pixel_scale_is_projected=False``, in which case it is taken as a
        ground sampling and converted up by ``k``.
    latitude_deg :
        Latitude at which the tile must work.  **Required, with no default**:
        the ROI centre is deliberately unset (DECISIONS.md D-002) and must never
        be hard-coded.  Pass the equator-most (least negative) latitude of the
        ROI for the worst case, since ``k`` grows away from the pole.
    proposed_tile_px :
        A tile size to check.  When omitted, the recommendation is reported and
        ``fits`` describes the recommendation itself.
    max_level_searched :
        How far up the pyramid to look for a level at which the proposed tile
        would suffice.

    Returns
    -------
    TileSizeRecommendation
    """
    if pixel_scale_m <= 0:
        raise ValueError(f"pixel_scale_m must be positive, got {pixel_scale_m}")
    if not (min_diameter_m > 0 and max_diameter_m >= min_diameter_m):
        raise ValueError(
            f"need 0 < min_diameter_m <= max_diameter_m, got "
            f"{min_diameter_m} and {max_diameter_m}"
        )
    if context_fraction < 0:
        raise ValueError(f"context_fraction must be >= 0, got {context_fraction}")
    if proposed_tile_px is not None and proposed_tile_px < 1:
        raise ValueError("proposed_tile_px must be a positive int")
    if max_level_searched < 0:
        raise ValueError("max_level_searched must be >= 0")

    k = float(_g.point_scale_factor(latitude_deg, k0=k0))
    if pixel_scale_is_projected:
        ps_proj = float(pixel_scale_m)
        ps_ground = ps_proj / k
    else:
        ps_ground = float(pixel_scale_m)
        ps_proj = ps_ground * k

    def diameter_px(d_m: float, level: int = 0) -> float:
        """Ground diameter -> pixels at pyramid ``level``."""
        return d_m * k / (ps_proj * level_scale_factor(level))

    max_px = diameter_px(max_diameter_m)
    min_px = diameter_px(min_diameter_m)

    def required_for(level: int) -> tuple[int, int, int]:
        dpx = diameter_px(max_diameter_m, level)
        margin = int(math.ceil(dpx * (0.5 + context_fraction)))
        core = int(math.ceil(dpx)) if required_core_px is None else int(required_core_px)
        core = max(1, core)
        return margin, core, core + 2 * margin

    req_margin, req_core, req_tile = required_for(0)
    recommended = _round_up(req_tile, round_to)

    warns: list[str] = []
    proposed = None if proposed_tile_px is None else int(proposed_tile_px)
    effective = recommended if proposed is None else proposed
    fits = effective >= req_tile
    shortfall = max(0, req_tile - effective)

    # Largest ground diameter the proposed tile can hold, inverting the above.
    # tile >= core + 2*ceil(Dpx*(0.5+c)) with core = ceil(Dpx) when unconstrained.
    if required_core_px is None:
        denom = (1.0 + 2.0 * (0.5 + context_fraction))
    else:
        denom = 2.0 * (0.5 + context_fraction)
    budget = effective - (0 if required_core_px is None else int(required_core_px))
    largest_fit = max(0.0, budget * ps_proj / (k * denom)) if denom > 0 else float("inf")

    min_level: int | None = None
    for level in range(0, int(max_level_searched) + 1):
        if effective >= required_for(level)[2]:
            min_level = level
            break
    ps_at_level = None if min_level is None else ps_proj * level_scale_factor(min_level)
    min_px_at_level = (None if min_level is None
                       else diameter_px(min_diameter_m, min_level))

    resolved_now = min_px >= min_pixels_across
    resolved_at_level = (None if min_px_at_level is None
                         else min_px_at_level >= min_pixels_across)

    if not fits:
        warns.append(
            f"tile of {effective} px at {ps_proj:.4g} projected m/px spans "
            f"{effective * ps_proj:.1f} projected m "
            f"({effective * ps_ground:.1f} ground m) and CANNOT hold the "
            f"{max_diameter_m:g} m target crater ({max_px:.1f} px) with "
            f"{context_fraction:.0%} context plus a {req_core} px core: "
            f"{req_tile} px are required, short by {shortfall} px"
        )
    if not resolved_now:
        warns.append(
            f"the smallest target crater ({min_diameter_m:g} m) spans only "
            f"{min_px:.1f} px at {ps_proj:.4g} projected m/px, below the "
            f"{min_pixels_across:g} px floor for a defensible rim fit"
        )
    if min_level is None:
        warns.append(
            f"no pyramid level up to L{max_level_searched} lets a {effective} px "
            f"tile hold a {max_diameter_m:g} m crater with context"
        )
    elif min_level > 0:
        warns.append(
            f"a {effective} px tile would only suffice from pyramid level "
            f"L{min_level} ({ps_at_level:.4g} projected m/px), where the "
            f"{min_diameter_m:g} m end of the range spans "
            f"{min_px_at_level:.1f} px"
            + ("" if resolved_at_level else
               f" -- below the {min_pixels_across:g} px floor, so a SINGLE tile "
               "size and scale cannot serve the whole approved diameter range; "
               "a two-scale (or larger-tile) strategy is required")
        )
    if context_fraction == 0:
        warns.append(
            "context_fraction=0 means a crater may touch the tile edge exactly; "
            "no background is guaranteed for a rim fit"
        )

    ok = fits and resolved_now
    msg = (
        f"At {ps_proj:.4g} projected m/px (= {ps_ground:.4g} ground m/px, "
        f"k={k:.6f} at {latitude_deg:g} deg), the {max_diameter_m:g} m target "
        f"crater spans {max_px:.1f} px; with {context_fraction:.0%} context and "
        f"a {req_core} px core the tile must be {req_tile} px "
        f"(recommended {recommended} px). "
        + (f"Proposed {proposed} px: {'OK' if fits else 'INSUFFICIENT'}. "
           if proposed is not None else "")
        + ("No issues found." if ok else "Issues: " + " | ".join(warns))
    )
    if warns and emit_warning:
        _warnings.warn(msg, TileSizeWarning, stacklevel=2)

    return TileSizeRecommendation(
        pixel_scale_projected_m=ps_proj,
        pixel_scale_ground_m=ps_ground,
        latitude_deg=float(latitude_deg),
        scale_factor_k=k,
        max_diameter_m=float(max_diameter_m),
        min_diameter_m=float(min_diameter_m),
        max_diameter_px=float(max_px),
        min_diameter_px=float(min_px),
        context_fraction=float(context_fraction),
        required_margin_px=req_margin,
        required_core_px=req_core,
        required_tile_px=req_tile,
        recommended_tile_px=recommended,
        proposed_tile_px=proposed,
        proposed_tile_span_projected_m=None if proposed is None else proposed * ps_proj,
        proposed_tile_span_ground_m=None if proposed is None else proposed * ps_ground,
        fits=bool(fits),
        shortfall_px=int(shortfall),
        largest_diameter_that_fits_m=float(largest_fit),
        min_level_that_fits=min_level,
        pixel_scale_at_min_level_m=ps_at_level,
        min_diameter_px_at_min_level=min_px_at_level,
        min_pixels_across=float(min_pixels_across),
        smallest_crater_resolved=bool(resolved_now),
        smallest_crater_resolved_at_min_level=resolved_at_level,
        warnings=tuple(warns),
        ok=bool(ok),
        message=msg,
    )
