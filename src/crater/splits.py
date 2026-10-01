"""Geographic train/val/test splitting with provably independent terrain.

Why a geographic split, and why this API shape
----------------------------------------------
Crater terrain is spatially autocorrelated at every scale this survey works at:
secondary crater fields, ejecta blankets, illumination and the degradation state
of a slope all vary smoothly over hundreds of metres to kilometres.  If two
tiles that touch each other end up in different splits, the validation score
measures interpolation inside a texture the model has already seen, not
generalisation to new terrain.  Worse, a tiling with a context margin makes
neighbouring tiles share *literal pixels* (see :class:`crater.tiling.TileGrid`),
so a random tile split does not merely correlate train and validation -- it puts
the same image data on both sides.

Therefore:

* Splits are defined as **contiguous geographic blocks** in projected metres.
* A tile joins a split only if it lies **wholly inside** that split's region and
  is at least ``buffer_m`` from **every other** split's region.
* Anything in a buffer zone, straddling a boundary, or outside all regions is
  **discarded with a named reason** -- never quietly pushed into a split.

A random tile split is not expressible through this API
------------------------------------------------------
This is a deliberate, enforced property, not a convention:

1. The only way to define a split is :class:`SplitRegion`, which accepts a
   **single connected** polygon.  Its constructor rejects ``MultiPolygon`` and
   any geometry whose interior falls apart under an infinitesimal erosion, so
   "the union of these scattered tiles" cannot be made into a region.
2. Interior rings (holes) are rejected unless ``allow_holes=True`` is passed
   explicitly, because a hole is the obvious back door for cherry-picking
   individual tiles out of a block.  When enabled it is recorded and surfaced in
   the plan report, so it cannot be used silently.
3. :func:`assign_tiles_to_splits` derives membership **only** from geometry.
   There is no parameter that assigns a tile by id, no override hook, and no way
   to move a discarded tile into a split.
4. This module has no source of randomness at all: no generator, no seeded
   sampling, nothing that permutes or reorders tiles.  A test asserts this of
   the source text, so it cannot drift.
5. :func:`check_leakage` is a *verifier*, not an assigner.  It takes whatever
   split labels you claim and reports, by id, every way those labels violate
   independence.  A hand-made random split handed to it will light up.

The only residual way to approximate a random split is to declare a great many
tiny single-tile regions; :func:`plan_block_splits` reports per-split tile
counts and a compactness ratio so that such a scheme is visible in the report
rather than hidden.

Units, and the projected-vs-ground trap
---------------------------------------
All geometry here is in **projected metres** on the south polar stereographic
CRS of ``crater.body.MOON``.  Projected metres are the right working units for
split geometry -- they are planar, so shapely's predicates are exact, whereas
lon/lat degrees are neither metric nor planar near the pole.

But a *buffer* is a physical separation on the ground, and projected metres are
not ground metres: a projected length divides by the point scale factor ``k`` to
become a ground length (INTERFACES.md point 2), so a projected separation of
``B`` is only ``B / k`` metres of real terrain.  ``buffer_kind="ground"`` (the
default) therefore converts the requested ground buffer **up** by ``k`` at each
tile's own latitude before testing it in the plane.  Since ``k >= 1`` this is
conservative: the enforced separation is never less than asked for.  Pass
``buffer_kind="projected"`` to test the raw planar distance instead, which is
what you want when reproducing a published plane-geometry split.

Areas are likewise reported both ways: a projected area divides by ``k**2`` to
become true surface area.  The authoritative survey-area computation belongs to
``crater.area``; the areas here are split-sizing diagnostics.

All distances described as ground distances are planimetric great-circle on the
sphere, matching ``crater.geometry.DIAMETER_DISTANCE_KIND``.  No ROI centre is
hard-coded anywhere (DECISIONS.md D-002).
"""
from __future__ import annotations

import math
import warnings as _warnings
from dataclasses import dataclass
from typing import Iterable, Mapping, Sequence

from shapely.geometry import MultiPolygon, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from . import geometry as _g
from . import tiling as _t
from .body import MOON, Body

__all__ = [
    "SplitGeometryWarning",
    "BUFFER_KINDS",
    "DISCARD_REASONS",
    "FINDING_KINDS",
    "SplitRegion",
    "SplitTile",
    "TileAssignment",
    "SplitAssignment",
    "LeakageFinding",
    "LeakageReport",
    "SplitPlan",
    "assign_tiles_to_splits",
    "check_leakage",
    "block_regions",
    "plan_block_splits",
]


class SplitGeometryWarning(UserWarning):
    """Warning for split geometry that is legal but scientifically questionable."""


#: How ``buffer_m`` is interpreted.  See the module docstring.
BUFFER_KINDS = ("ground", "projected")

#: Every reason a tile can be withheld from all splits.  A discarded tile always
#: carries exactly one of these; there is no unlabelled discard path.
DISCARD_REASONS = (
    "outside_all_regions",          # the tile touches no split region at all
    "straddles_region_boundary",    # the tile overlaps two or more regions
    "partially_outside_region",     # the tile overlaps one region but is not inside it
    "within_buffer_of_other_split",  # inside one region, too close to another
)

#: Named leakage findings produced by :func:`check_leakage`.
FINDING_KINDS = (
    # (a) plain geometric correlation: same scale, same acquisition.
    "cross_split_footprint_proximity",
    # (b) the same catalogued crater appears in more than one split.
    "crater_id_in_multiple_splits",
    # (c) the same ground area at two different pyramid scales, split apart.
    "same_ground_area_at_different_scales",
    # (d) a different acquisition (source_id) of the same ground area.
    "alternate_acquisition_of_same_area",
    # bookkeeping error: one tile id claimed by two splits.
    "duplicate_tile_id_across_splits",
)


# --------------------------------------------------------------------------- #
# Split regions
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SplitRegion:
    """One contiguous geographic block, in projected metres.

    A region is the *only* way to define a split in this module, and it must be a
    single connected polygon.  That is what makes a random tile split
    inexpressible: a set of scattered tiles unions to a ``MultiPolygon`` (or, if
    they merely touch at corners, to a geometry whose interior disconnects under
    an infinitesimal erosion), and both are rejected here.

    Parameters
    ----------
    name :
        Split name, e.g. ``"train"``.  Must be non-empty and unique within a
        collection of regions.
    geometry :
        A shapely ``Polygon`` in projected metres.  Validated for being
        non-empty, valid, positive-area and connected.
    allow_holes :
        Interior rings are rejected by default: a hole lets a caller excise
        individual tiles from the middle of a block, which is a random split in
        disguise.  Set True only for a documented data gap (e.g. a permanently
        shadowed region with no usable imagery); the hole count is then recorded
        in :attr:`n_holes` and reported by :func:`plan_block_splits`.
    """

    name: str
    geometry: Polygon
    allow_holes: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("split region name must be a non-empty string")
        geom = self.geometry
        if isinstance(geom, MultiPolygon):
            raise ValueError(
                f"split region {self.name!r} is a MultiPolygon with "
                f"{len(geom.geoms)} parts; a split must be ONE contiguous "
                "geographic block. A union of scattered tiles is exactly the "
                "random split this module exists to prevent."
            )
        if not isinstance(geom, Polygon):
            raise TypeError(
                f"split region {self.name!r} must be a shapely Polygon, got "
                f"{type(geom).__name__}"
            )
        if geom.is_empty:
            raise ValueError(f"split region {self.name!r} is empty")
        if not geom.is_valid:
            raise ValueError(
                f"split region {self.name!r} is not a valid polygon "
                "(self-intersecting or malformed ring)"
            )
        if geom.area <= 0.0:
            raise ValueError(f"split region {self.name!r} has zero area")
        n_holes = len(geom.interiors)
        if n_holes and not self.allow_holes:
            raise ValueError(
                f"split region {self.name!r} has {n_holes} interior ring(s). "
                "Holes are refused by default because they allow individual "
                "tiles to be excised from a block; pass allow_holes=True and "
                "document the data gap if that is genuinely what you mean."
            )
        # Connectivity: erode by a length that is negligible compared with the
        # region (1e-6 of its own root-area).  Two blocks that merely kiss at a
        # corner, or a ring that pinches to a point, fall into two pieces.
        eps = math.sqrt(geom.area) * 1e-6
        eroded = geom.buffer(-eps)
        if eroded.is_empty:
            raise ValueError(
                f"split region {self.name!r} is degenerate: it vanishes under an "
                f"erosion of {eps:.3e} m, so it has effectively zero width"
            )
        parts = getattr(eroded, "geoms", None)
        if parts is not None and len(parts) > 1:
            raise ValueError(
                f"split region {self.name!r} is not contiguous: its interior "
                f"splits into {len(parts)} pieces under an infinitesimal "
                "erosion (blocks touching only at a point are not contiguous)"
            )
        object.__setattr__(self, "name", self.name.strip())

    @classmethod
    def from_bounds(cls, name: str, x_min: float, y_min: float,
                    x_max: float, y_max: float) -> "SplitRegion":
        """A rectangular block from projected-metre bounds."""
        if not (x_max > x_min and y_max > y_min):
            raise ValueError(
                f"from_bounds needs x_max > x_min and y_max > y_min, got "
                f"({x_min}, {y_min}, {x_max}, {y_max})"
            )
        return cls(name=name, geometry=box(x_min, y_min, x_max, y_max))

    @property
    def n_holes(self) -> int:
        """Number of interior rings (data gaps) in this region."""
        return len(self.geometry.interiors)

    @property
    def projected_area_m2(self) -> float:
        """Area of the region in projected metres squared (not ground area)."""
        return float(self.geometry.area)

    @property
    def compactness(self) -> float:
        """Polsby-Popper compactness ``4*pi*A/P**2``; 1.0 for a disc.

        Reported, not enforced.  A very low value means a long thin block, which
        maximises the length of boundary along which tiles must be discarded and
        is a hint that the region is being used to trace around tiles.
        """
        p = float(self.geometry.length)
        return float(4.0 * math.pi * self.geometry.area / (p * p)) if p > 0 else 0.0

    def centroid_lonlat(self, *, body: Body = MOON, lon_0: float = 0.0,
                        k0: float = 1.0) -> tuple[float, float]:
        """Region centroid as ``(lon, lat)`` degrees, via ``geometry.inverse``."""
        c = self.geometry.centroid
        return _g.inverse(c.x, c.y, body=body, lon_0=lon_0, k0=k0)


# --------------------------------------------------------------------------- #
# Tiles carrying split membership and provenance
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SplitTile:
    """A tile footprint with the split it is claimed to belong to.

    This is the input to :func:`check_leakage`.  It carries the three pieces of
    provenance that make the four leakage mechanisms distinguishable:

    level :
        Pyramid level.  Two tiles at different levels covering the same ground
        are a *multiscale* leak (finding kind (c)), which a naive footprint check
        would either miss (if it only compares tiles of equal size) or report
        without naming the mechanism.
    source_id :
        The acquisition the pixels came from (e.g. an NAC product id).  Two
        different acquisitions of the same ground in different splits are an
        *alternate acquisition* leak (finding kind (d)): the terrain is
        identical, only the illumination and viewing geometry differ, which is
        precisely the kind of near-duplicate a detector memorises.
    crater_ids :
        Catalogue ids of craters falling in this tile, used for finding kind (b).
    """

    tile_id: str
    footprint: Polygon
    split: str
    level: int = 0
    source_id: str = "unknown"
    crater_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.footprint, BaseGeometry) or self.footprint.is_empty:
            raise ValueError(f"tile {self.tile_id!r} has an empty footprint")
        if not isinstance(self.split, str) or not self.split.strip():
            raise ValueError(f"tile {self.tile_id!r} has no split label")
        object.__setattr__(self, "crater_ids", tuple(self.crater_ids))

    @classmethod
    def from_tile(cls, tile, split: str, *, source_id: str = "unknown",
                  crater_ids: Sequence[str] = ()) -> "SplitTile":
        """Build from a :class:`crater.tiling.Tile`, using its projected corners."""
        return cls(
            tile_id=tile.tile_id,
            footprint=footprint_of(tile),
            split=split,
            level=int(getattr(tile, "level", 0)),
            source_id=source_id,
            crater_ids=tuple(crater_ids),
        )


def footprint_of(tile) -> Polygon:
    """Projected-metre footprint polygon of a tile-like object.

    Accepts anything carrying a shapely ``footprint``, or a
    :class:`crater.tiling.Tile` (whose four projected corners are used, so a
    rotated mosaic is handled exactly rather than via a bounding box).
    """
    fp = getattr(tile, "footprint", None)
    if isinstance(fp, BaseGeometry):
        return fp
    if hasattr(tile, "transform") and hasattr(tile, "width"):
        return Polygon(_t.tile_corners_projected(tile))
    raise TypeError(
        f"cannot derive a footprint from {type(tile).__name__}; supply a "
        "crater.tiling.Tile or an object with a shapely `footprint`"
    )


def _tile_id_of(tile) -> str:
    tid = getattr(tile, "tile_id", None)
    if not isinstance(tid, str) or not tid:
        raise TypeError(f"{type(tile).__name__} has no usable tile_id")
    return tid


def _centroid_lat(geom: BaseGeometry, body: Body, lon_0: float, k0: float) -> float:
    c = geom.centroid
    return float(_g.inverse(c.x, c.y, body=body, lon_0=lon_0, k0=k0)[1])


def _scale_factor_for(geom: BaseGeometry, body: Body, lon_0: float, k0: float) -> float:
    return float(_g.point_scale_factor(_centroid_lat(geom, body, lon_0, k0), k0=k0))


def _required_projected_separation(buffer_m: float, buffer_kind: str, k: float) -> float:
    """Planar separation that enforces ``buffer_m`` under the chosen convention.

    For ``buffer_kind="ground"`` the requested ground separation must be scaled
    **up** by ``k`` before it is compared with a planar distance, because one
    projected metre is only ``1/k`` ground metres.  With ``k >= 1`` this errs
    towards discarding more tiles, which is the safe direction for leakage.
    """
    if buffer_kind == "projected":
        return float(buffer_m)
    if buffer_kind == "ground":
        return float(buffer_m) * float(k)
    raise ValueError(f"buffer_kind must be one of {BUFFER_KINDS}, got {buffer_kind!r}")


def _validate_regions(regions: Sequence[SplitRegion]) -> tuple[SplitRegion, ...]:
    regions = tuple(regions)
    if not regions:
        raise ValueError("at least one split region is required")
    for r in regions:
        if not isinstance(r, SplitRegion):
            raise TypeError(
                f"split regions must be SplitRegion instances, got "
                f"{type(r).__name__}; this is what guarantees contiguity"
            )
    names = [r.name for r in regions]
    if len(set(names)) != len(names):
        raise ValueError(f"split region names must be unique, got {names}")
    for i in range(len(regions)):
        for j in range(i + 1, len(regions)):
            inter = regions[i].geometry.intersection(regions[j].geometry)
            if inter.area > 0.0:
                raise ValueError(
                    f"split regions {regions[i].name!r} and {regions[j].name!r} "
                    f"overlap by {inter.area:.6g} projected m^2; overlapping "
                    "regions make split membership ambiguous and guarantee "
                    "leakage. Adjacent (edge-touching) regions are allowed."
                )
    return regions


# --------------------------------------------------------------------------- #
# Assignment
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TileAssignment:
    """The outcome for one tile, with the numbers behind it.

    ``split`` is ``None`` exactly when the tile was discarded, in which case
    ``reason`` is one of :data:`DISCARD_REASONS`.  Both are always set; there is
    no state in which a tile has been silently dropped or silently placed.
    """

    tile_id: str
    split: str | None
    reason: str
    detail: str
    footprint: Polygon
    level: int
    source_id: str
    crater_ids: tuple[str, ...]
    scale_factor_k: float
    required_separation_projected_m: float
    nearest_other_split: str | None
    nearest_other_distance_projected_m: float | None

    @property
    def discarded(self) -> bool:
        return self.split is None

    @property
    def nearest_other_distance_ground_m(self) -> float | None:
        """Nearest-other-split distance as a ground length (``/ k``)."""
        d = self.nearest_other_distance_projected_m
        return None if d is None else d / self.scale_factor_k

    @property
    def projected_area_m2(self) -> float:
        return float(self.footprint.area)

    @property
    def ground_area_m2(self) -> float:
        """Footprint area as true surface area: projected area ``/ k**2``."""
        return float(self.footprint.area) / (self.scale_factor_k ** 2)


@dataclass(frozen=True)
class SplitAssignment:
    """Result of :func:`assign_tiles_to_splits`."""

    assignments: tuple[TileAssignment, ...]
    regions: tuple[SplitRegion, ...]
    buffer_m: float
    buffer_kind: str

    @property
    def region_names(self) -> tuple[str, ...]:
        return tuple(r.name for r in self.regions)

    @property
    def splits(self) -> dict[str, tuple[str, ...]]:
        """Mapping split name -> tile ids, including empty splits."""
        out: dict[str, list[str]] = {r.name: [] for r in self.regions}
        for a in self.assignments:
            if a.split is not None:
                out[a.split].append(a.tile_id)
        return {k: tuple(v) for k, v in out.items()}

    @property
    def discarded(self) -> tuple[TileAssignment, ...]:
        """The discarded set: every tile withheld from all splits, with reasons."""
        return tuple(a for a in self.assignments if a.discarded)

    @property
    def discarded_ids(self) -> tuple[str, ...]:
        return tuple(a.tile_id for a in self.discarded)

    @property
    def reason_counts(self) -> dict[str, int]:
        """How many tiles were discarded for each of :data:`DISCARD_REASONS`."""
        counts = {r: 0 for r in DISCARD_REASONS}
        for a in self.discarded:
            counts[a.reason] = counts.get(a.reason, 0) + 1
        return counts

    def split_of(self, tile_id: str) -> str | None:
        for a in self.assignments:
            if a.tile_id == tile_id:
                return a.split
        raise KeyError(tile_id)

    def assignment_of(self, tile_id: str) -> TileAssignment:
        for a in self.assignments:
            if a.tile_id == tile_id:
                return a
        raise KeyError(tile_id)

    def split_tiles(self) -> tuple[SplitTile, ...]:
        """Assigned tiles as :class:`SplitTile`, ready for :func:`check_leakage`."""
        return tuple(
            SplitTile(tile_id=a.tile_id, footprint=a.footprint, split=a.split,
                      level=a.level, source_id=a.source_id,
                      crater_ids=a.crater_ids)
            for a in self.assignments if a.split is not None
        )


def assign_tiles_to_splits(tiles: Iterable, split_regions: Sequence[SplitRegion],
                           buffer_m: float, *, buffer_kind: str = "ground",
                           source_ids: Mapping[str, str] | None = None,
                           crater_ids: Mapping[str, Sequence[str]] | None = None,
                           body: Body = MOON, lon_0: float = 0.0,
                           k0: float = 1.0) -> SplitAssignment:
    """Assign tiles to contiguous split regions, discarding anything ambiguous.

    A tile is placed in split ``S`` if and only if **both** hold:

    1. its footprint lies **wholly inside** region ``S`` (shapely ``covers``;
       the region is used as given -- *eroded by nothing*, so the only margin in
       play is the explicit buffer below); and
    2. its footprint is at least ``buffer_m`` away from **every other** split
       region.

    Otherwise the tile is discarded with one of :data:`DISCARD_REASONS`.  There
    is no third outcome: a tile is either placed with both conditions proven or
    withheld with a reason naming why.  Note that condition 2 is tested against
    the *regions*, not against the tiles already assigned to them, so the result
    does not depend on iteration order and is reproducible.

    Parameters
    ----------
    tiles :
        :class:`crater.tiling.Tile` objects, or anything with ``tile_id`` and a
        shapely ``footprint`` in projected metres.
    split_regions :
        :class:`SplitRegion` blocks.  Must be pairwise non-overlapping (touching
        is fine); overlapping regions raise, since membership would be ambiguous.
    buffer_m :
        Required separation between a tile and any foreign region.  Interpreted
        as a ground distance by default -- see ``buffer_kind`` and the module
        docstring.  ``0`` is allowed and means "adjacency is acceptable", which
        it generally is not for correlated terrain.
    buffer_kind :
        ``"ground"`` (default) or ``"projected"``; see :data:`BUFFER_KINDS`.
    source_ids, crater_ids :
        Optional per-tile-id provenance, carried through to the result so the
        assignment can be fed straight to :func:`check_leakage`.

    Returns
    -------
    SplitAssignment
    """
    if buffer_m < 0:
        raise ValueError(f"buffer_m must be >= 0, got {buffer_m}")
    if buffer_kind not in BUFFER_KINDS:
        raise ValueError(f"buffer_kind must be one of {BUFFER_KINDS}, got {buffer_kind!r}")
    regions = _validate_regions(split_regions)

    out: list[TileAssignment] = []
    seen: set[str] = set()
    for tile in tiles:
        tid = _tile_id_of(tile)
        if tid in seen:
            raise ValueError(
                f"duplicate tile_id {tid!r} in the input; tile ids must be "
                "unique or the assignment is ambiguous"
            )
        seen.add(tid)
        fp = footprint_of(tile)
        if fp.is_empty or fp.area <= 0:
            raise ValueError(f"tile {tid!r} has an empty or zero-area footprint")
        level = int(getattr(tile, "level", 0))
        src = (source_ids or {}).get(tid, getattr(tile, "source_id", "unknown"))
        crat = tuple((crater_ids or {}).get(tid, getattr(tile, "crater_ids", ())))
        k = _scale_factor_for(fp, body, lon_0, k0)
        required = _required_projected_separation(buffer_m, buffer_kind, k)

        inside = [r for r in regions if r.geometry.covers(fp)]
        if len(inside) > 1:  # pragma: no cover - prevented by _validate_regions
            raise AssertionError(
                f"tile {tid!r} is covered by several regions "
                f"{[r.name for r in inside]}; regions should be disjoint"
            )
        if not inside:
            touching = [r.name for r in regions if r.geometry.intersects(fp)]
            if len(touching) >= 2:
                reason = "straddles_region_boundary"
                detail = (f"footprint overlaps {len(touching)} regions "
                          f"({', '.join(sorted(touching))}); a tile spanning a "
                          "split boundary shares pixels with both splits")
            elif len(touching) == 1:
                reason = "partially_outside_region"
                detail = (f"footprint overlaps region {touching[0]!r} but is not "
                          "wholly inside it")
            else:
                reason = "outside_all_regions"
                detail = "footprint lies outside every split region"
            dists = {r.name: float(fp.distance(r.geometry)) for r in regions}
            nearest = min(dists, key=dists.get) if dists else None
            out.append(TileAssignment(
                tile_id=tid, split=None, reason=reason, detail=detail,
                footprint=fp, level=level, source_id=src, crater_ids=crat,
                scale_factor_k=k, required_separation_projected_m=required,
                nearest_other_split=nearest,
                nearest_other_distance_projected_m=(
                    None if nearest is None else dists[nearest]),
            ))
            continue

        home = inside[0]
        others = [r for r in regions if r.name != home.name]
        dists = {r.name: float(fp.distance(r.geometry)) for r in others}
        nearest = min(dists, key=dists.get) if dists else None
        nearest_d = None if nearest is None else dists[nearest]
        if nearest is not None and nearest_d < required:
            out.append(TileAssignment(
                tile_id=tid, split=None,
                reason="within_buffer_of_other_split",
                detail=(f"inside {home.name!r} but only {nearest_d:.6g} projected m "
                        f"({nearest_d / k:.6g} ground m) from {nearest!r}; "
                        f"{required:.6g} projected m required for a "
                        f"{buffer_m:g} m {buffer_kind} buffer (k={k:.6f})"),
                footprint=fp, level=level, source_id=src, crater_ids=crat,
                scale_factor_k=k, required_separation_projected_m=required,
                nearest_other_split=nearest,
                nearest_other_distance_projected_m=nearest_d,
            ))
            continue

        out.append(TileAssignment(
            tile_id=tid, split=home.name, reason="", detail=(
                f"wholly inside {home.name!r}"
                + ("" if nearest is None else
                   f"; {nearest_d:.6g} projected m from the nearest other "
                   f"region {nearest!r} (>= {required:.6g} required)")),
            footprint=fp, level=level, source_id=src, crater_ids=crat,
            scale_factor_k=k, required_separation_projected_m=required,
            nearest_other_split=nearest,
            nearest_other_distance_projected_m=nearest_d,
        ))

    return SplitAssignment(assignments=tuple(out), regions=regions,
                           buffer_m=float(buffer_m), buffer_kind=buffer_kind)


# --------------------------------------------------------------------------- #
# Leakage audit
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LeakageFinding:
    """One named leakage finding, naming the offending ids.

    Attributes
    ----------
    kind :
        One of :data:`FINDING_KINDS`.
    ids :
        The primary offending identifiers: the pair of tile ids for geometric
        findings, or the crater id for a crater duplicated across splits.
    tile_ids :
        The tile ids involved, always populated.
    splits :
        The split names involved, sorted.
    measured_projected_m, required_projected_m :
        The planar separation found and the one required.  ``None`` for findings
        that are not distance-based.
    overlap_area_projected_m2 :
        Area of the shared footprint, in projected metres squared.  Non-zero
        means the two splits literally contain the same pixels.
    detail :
        A sentence a human can act on.
    """

    kind: str
    ids: tuple[str, ...]
    tile_ids: tuple[str, ...]
    splits: tuple[str, ...]
    detail: str
    measured_projected_m: float | None = None
    required_projected_m: float | None = None
    overlap_area_projected_m2: float = 0.0

    def __post_init__(self) -> None:
        if self.kind not in FINDING_KINDS:
            raise ValueError(f"unknown finding kind {self.kind!r}")


@dataclass(frozen=True)
class LeakageReport:
    """Structured result of :func:`check_leakage`.

    ``clean`` is True only when no finding of any kind was raised.  The report
    never summarises a finding away: ``findings`` holds every one, each naming
    its offending ids.  ``n_pairs_examined`` counts the *cross-split* candidate
    pairs that were distance-tested (same-split neighbours are expected and are
    not examined), and ``split_tile_counts`` is the tile count per claimed split.
    """

    findings: tuple[LeakageFinding, ...]
    n_tiles: int
    n_pairs_examined: int
    split_tile_counts: dict[str, int]
    buffer_m: float
    buffer_kind: str

    @property
    def clean(self) -> bool:
        """True iff no leakage of any kind was found."""
        return not self.findings

    @property
    def kinds_found(self) -> tuple[str, ...]:
        return tuple(k for k in FINDING_KINDS
                     if any(f.kind == k for f in self.findings))

    def by_kind(self, kind: str) -> tuple[LeakageFinding, ...]:
        """All findings of one of :data:`FINDING_KINDS`."""
        if kind not in FINDING_KINDS:
            raise ValueError(f"unknown finding kind {kind!r}")
        return tuple(f for f in self.findings if f.kind == kind)

    def counts(self) -> dict[str, int]:
        return {k: len(self.by_kind(k)) for k in FINDING_KINDS}

    @property
    def summary(self) -> str:
        if self.clean:
            return (f"No leakage found across {self.n_tiles} tiles "
                    f"({self.n_pairs_examined} cross-split candidate pairs "
                    f"examined) with a "
                    f"{self.buffer_m:g} m {self.buffer_kind} buffer.")
        parts = [f"{k}={n}" for k, n in self.counts().items() if n]
        return (f"LEAKAGE: {len(self.findings)} finding(s) across "
                f"{self.n_tiles} tiles -- " + ", ".join(parts))


def check_leakage(split_tiles: Iterable[SplitTile], buffer_m: float, *,
                  buffer_kind: str = "ground", body: Body = MOON,
                  lon_0: float = 0.0, k0: float = 1.0) -> LeakageReport:
    """Audit a claimed split for every way train/val/test can share information.

    This is a *verifier*: it takes the split labels as claimed and reports what
    is wrong with them.  It never changes an assignment, and it has no notion of
    a "good enough" split -- a non-empty ``findings`` list means the split is not
    independent.

    Four mechanisms are detected as separate named findings
    ------------------------------------------------------
    (a) ``cross_split_footprint_proximity``
        Two tiles in different splits, at the **same** pyramid level and from the
        **same** acquisition, whose footprints overlap or sit closer than the
        buffer.  This is the everyday failure: with a context margin, adjacent
        tiles share pixels outright.
    (c) ``same_ground_area_at_different_scales``
        As (a) but the two tiles are at **different** pyramid levels.  A 512 px
        level-2 tile and a 512 px level-0 tile can cover the same massif; the
        model sees the same terrain, merely resampled.  Reported separately
        because a footprint check that only compares equal-sized tiles, or that
        keys on tile id, misses it entirely.
    (d) ``alternate_acquisition_of_same_area``
        As (a) but the two tiles come from **different** ``source_id``s.  The
        terrain is identical and only the illumination and viewing geometry
        differ -- a near-duplicate, not an independent sample.
    (b) ``crater_id_in_multiple_splits``
        The same catalogued crater id appears in tiles belonging to more than one
        split.  Caught by id, so it fires even when the footprints do not come
        close (e.g. a crater listed in two acquisitions' label files).

    Cases (a), (c) and (d) **partition** the violating cross-split pairs: a pair
    differing in neither level nor source yields (a), one differing in level
    yields (c), one differing in source yields (d), and a pair differing in both
    yields both (c) and (d) because both mechanisms genuinely apply.  Every
    violating pair therefore produces at least one finding, and no pair is
    reported twice under the same kind.

    A fifth kind, ``duplicate_tile_id_across_splits``, catches the bookkeeping
    error of one tile id claimed by two splits.

    Parameters
    ----------
    split_tiles :
        :class:`SplitTile` objects (or anything with ``tile_id``, ``footprint``,
        ``split``, and optionally ``level``, ``source_id``, ``crater_ids``).
    buffer_m, buffer_kind :
        The separation the split claims to enforce; see the module docstring for
        the ground-vs-projected distinction.  For a pair, the stricter of the two
        tiles' scale factors is used, so the test never under-enforces.

    Returns
    -------
    LeakageReport
    """
    if buffer_m < 0:
        raise ValueError(f"buffer_m must be >= 0, got {buffer_m}")
    if buffer_kind not in BUFFER_KINDS:
        raise ValueError(f"buffer_kind must be one of {BUFFER_KINDS}, got {buffer_kind!r}")

    tiles = list(split_tiles)
    findings: list[LeakageFinding] = []
    counts: dict[str, int] = {}
    for t in tiles:
        counts[t.split] = counts.get(t.split, 0) + 1

    # -- duplicate ids ----------------------------------------------------- #
    by_id: dict[str, list] = {}
    for t in tiles:
        by_id.setdefault(_tile_id_of(t), []).append(t)
    for tid, group in sorted(by_id.items()):
        claimed = sorted({t.split for t in group})
        if len(claimed) > 1:
            findings.append(LeakageFinding(
                kind="duplicate_tile_id_across_splits",
                ids=(tid,), tile_ids=(tid,), splits=tuple(claimed),
                detail=(f"tile id {tid!r} is claimed by splits "
                        f"{', '.join(claimed)}; one tile cannot be in two splits"),
            ))

    # -- (b) craters shared across splits ---------------------------------- #
    crater_map: dict[str, dict[str, list[str]]] = {}
    for t in tiles:
        for cid in getattr(t, "crater_ids", ()):
            crater_map.setdefault(str(cid), {}).setdefault(t.split, []).append(
                _tile_id_of(t))
    for cid, per_split in sorted(crater_map.items()):
        if len(per_split) > 1:
            splits = tuple(sorted(per_split))
            tids = tuple(sorted({tid for v in per_split.values() for tid in v}))
            findings.append(LeakageFinding(
                kind="crater_id_in_multiple_splits",
                ids=(cid,), tile_ids=tids, splits=splits,
                detail=(f"crater {cid!r} appears in splits {', '.join(splits)} "
                        f"via tiles {', '.join(tids)}; the same labelled object "
                        "cannot be both a training target and a test target"),
            ))

    # -- (a)/(c)/(d) geometric proximity ----------------------------------- #
    geoms = [footprint_of(t) for t in tiles]
    ks = [_scale_factor_for(gm, body, lon_0, k0) for gm in geoms]
    reqs = [_required_projected_separation(buffer_m, buffer_kind, k) for k in ks]
    max_req = max(reqs) if reqs else 0.0

    n_pairs = 0
    if tiles:
        tree = STRtree(geoms)
        seen_pairs: set[tuple[int, int]] = set()
        for i, gm in enumerate(geoms):
            probe = gm.buffer(max_req) if max_req > 0 else gm
            for j in tree.query(probe):
                j = int(j)
                if j == i:
                    continue
                pair = (i, j) if i < j else (j, i)
                if pair in seen_pairs:
                    continue
                seen_pairs.add(pair)
                a, b = tiles[pair[0]], tiles[pair[1]]
                if a.split == b.split:
                    continue
                ga, gb = geoms[pair[0]], geoms[pair[1]]
                n_pairs += 1
                required = max(reqs[pair[0]], reqs[pair[1]])
                dist = float(ga.distance(gb))
                overlap = float(ga.intersection(gb).area) if ga.intersects(gb) else 0.0
                if overlap <= 0.0 and dist >= required:
                    continue
                la, lb = int(getattr(a, "level", 0)), int(getattr(b, "level", 0))
                sa = str(getattr(a, "source_id", "unknown"))
                sb = str(getattr(b, "source_id", "unknown"))
                ids = (_tile_id_of(a), _tile_id_of(b))
                splits = tuple(sorted({a.split, b.split}))
                how = ("footprints overlap by "
                       f"{overlap:.6g} projected m^2" if overlap > 0 else
                       f"footprints are {dist:.6g} projected m apart, closer than "
                       f"the required {required:.6g}")
                kinds: list[tuple[str, str]] = []
                if la == lb and sa == sb:
                    kinds.append((
                        "cross_split_footprint_proximity",
                        f"tiles {ids[0]!r} ({a.split}) and {ids[1]!r} ({b.split}) "
                        f"are at the same level L{la} from the same source {sa!r} "
                        f"and {how}",
                    ))
                else:
                    if la != lb:
                        kinds.append((
                            "same_ground_area_at_different_scales",
                            f"tiles {ids[0]!r} ({a.split}, level L{la}) and "
                            f"{ids[1]!r} ({b.split}, level L{lb}) cover the same "
                            f"ground at different pyramid scales: {how}",
                        ))
                    if sa != sb:
                        kinds.append((
                            "alternate_acquisition_of_same_area",
                            f"tiles {ids[0]!r} ({a.split}, source {sa!r}) and "
                            f"{ids[1]!r} ({b.split}, source {sb!r}) are alternate "
                            f"acquisitions of the same ground: {how}",
                        ))
                for kind, detail in kinds:
                    findings.append(LeakageFinding(
                        kind=kind, ids=ids, tile_ids=ids, splits=splits,
                        detail=detail, measured_projected_m=dist,
                        required_projected_m=required,
                        overlap_area_projected_m2=overlap,
                    ))

    findings.sort(key=lambda f: (FINDING_KINDS.index(f.kind), f.ids, f.splits))
    return LeakageReport(
        findings=tuple(findings),
        n_tiles=len(tiles),
        n_pairs_examined=n_pairs,
        split_tile_counts=dict(sorted(counts.items())),
        buffer_m=float(buffer_m),
        buffer_kind=buffer_kind,
    )


# --------------------------------------------------------------------------- #
# Block construction and the 70/15/15 target
# --------------------------------------------------------------------------- #
def block_regions(bounds: tuple[float, float, float, float],
                  fractions: Sequence[float], names: Sequence[str],
                  axis: str = "y") -> tuple[SplitRegion, ...]:
    """Cut a projected-metre rectangle into contiguous bands.

    The bands are produced in increasing coordinate order along ``axis`` and are
    edge-adjacent, so they partition the rectangle exactly.  Each band is a
    single rectangle, hence trivially contiguous -- which is the point: this is
    the constructive counterpart to :class:`SplitRegion`'s refusal of scattered
    geometry.  There is deliberately no variant that assigns tiles.

    Parameters
    ----------
    bounds :
        ``(x_min, y_min, x_max, y_max)`` in projected metres.
    fractions :
        Band widths as fractions of the extent along ``axis``; must be positive
        and sum to 1.
    axis :
        ``"x"`` or ``"y"``.  Pick the axis that makes the bands cut *across* the
        dominant terrain gradient; on a polar mosaic a band in ``y`` near the
        central meridian is roughly a latitude band.
    """
    x0, y0, x1, y1 = (float(v) for v in bounds)
    if not (x1 > x0 and y1 > y0):
        raise ValueError(f"bounds must have positive extent, got {bounds}")
    fr = [float(f) for f in fractions]
    if len(fr) != len(names):
        raise ValueError(f"got {len(fr)} fractions for {len(names)} names")
    if not fr:
        raise ValueError("at least one band is required")
    if any(f <= 0 for f in fr):
        raise ValueError(f"all fractions must be > 0, got {fr}")
    if abs(sum(fr) - 1.0) > 1e-9:
        raise ValueError(
            f"fractions must sum to 1, got {sum(fr)!r}; this function will not "
            "renormalise a target ratio on your behalf"
        )
    if axis not in ("x", "y"):
        raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")

    span = (x1 - x0) if axis == "x" else (y1 - y0)
    lo = x0 if axis == "x" else y0
    out = []
    acc = 0.0
    for f, name in zip(fr, names):
        a = lo + acc * span
        b = lo + (acc + f) * span
        acc += f
        if axis == "x":
            out.append(SplitRegion.from_bounds(name, a, y0, b, y1))
        else:
            out.append(SplitRegion.from_bounds(name, x0, a, x1, b))
    return tuple(out)


@dataclass(frozen=True)
class SplitPlan:
    """A block split plan with its **achieved** ratio reported, not forced.

    The target ratio is an aspiration.  Whether it is met depends on the ROI
    shape, the tile size and the buffer -- a buffer wide enough to decorrelate
    terrain can easily consume a 15% band entirely.  This object therefore
    reports what the geometry actually delivered and sets ``feasible=False``
    rather than adjusting region sizes, moving tiles, or shrinking the buffer to
    hit the number.

    Attributes
    ----------
    regions, assignment :
        The blocks and the per-tile outcome (including the discarded set).
    target_ratio :
        What was asked for, per split name.
    region_projected_area_m2, region_area_ratio :
        Geometry of the blocks themselves, before any tile is assigned.  When
        these match the target but ``achieved_area_ratio`` does not, the loss is
        entirely buffer and boundary attrition.
    tile_counts, usable_projected_area_m2, usable_ground_area_m2 :
        Per split, over **assigned** tiles only.  Ground area is the projected
        area divided by ``k**2`` (INTERFACES.md point 2); it is the honest
        measure of how much Moon each split covers.  Overlapping tile footprints
        are summed as-is, so these are "tile area", not deduplicated coverage --
        the authoritative survey area is ``crater.area``'s job.
    achieved_area_ratio, achieved_count_ratio :
        Fractions of the total *usable* ground area and of the assigned tile
        count.  ``nan`` when nothing at all was assigned, rather than a
        plausible-looking zero.
    discarded_tiles, discarded_ground_area_m2, discarded_fraction :
        The cost of the buffer, stated explicitly.
    empty_splits :
        Splits that ended up with no tiles.  Any entry here makes the plan
        infeasible however good the other ratios look.
    max_abs_area_deviation, tolerance, feasible :
        ``feasible`` is True only when no split is empty and every achieved area
        ratio is within ``tolerance`` of its target.
    compactness :
        Polsby-Popper compactness per region; low values flag long thin blocks.
    warnings, message :
        Everything a reviewer needs to see, in plain sentences.
    """

    regions: tuple[SplitRegion, ...]
    assignment: SplitAssignment
    target_ratio: dict[str, float]
    region_projected_area_m2: dict[str, float]
    region_area_ratio: dict[str, float]
    tile_counts: dict[str, int]
    usable_projected_area_m2: dict[str, float]
    usable_ground_area_m2: dict[str, float]
    achieved_area_ratio: dict[str, float]
    achieved_count_ratio: dict[str, float]
    discarded_tiles: int
    discarded_ground_area_m2: float
    discarded_fraction: float
    empty_splits: tuple[str, ...]
    max_abs_area_deviation: float
    tolerance: float
    feasible: bool
    compactness: dict[str, float]
    buffer_m: float
    buffer_kind: str
    axis: str
    warnings: tuple[str, ...]
    message: str


def plan_block_splits(tiles: Iterable, *, buffer_m: float,
                      target_ratio: Sequence[float] = (0.70, 0.15, 0.15),
                      names: Sequence[str] = ("train", "val", "test"),
                      bounds: tuple[float, float, float, float] | None = None,
                      axis: str = "y", buffer_kind: str = "ground",
                      tolerance: float = 0.05,
                      source_ids: Mapping[str, str] | None = None,
                      crater_ids: Mapping[str, Sequence[str]] | None = None,
                      body: Body = MOON, lon_0: float = 0.0, k0: float = 1.0,
                      emit_warning: bool = True) -> SplitPlan:
    """Build contiguous blocks for a target ratio and report what was achieved.

    The default target is roughly 70/15/15.  The blocks are cut at exactly those
    fractions of the ROI extent along ``axis``, tiles are then assigned by
    :func:`assign_tiles_to_splits`, and the **achieved** ratio is computed from
    the usable ground area that survived the buffer.

    This function never forces the target.  It does not widen a starved band,
    shrink the buffer, reassign a discarded tile, or renormalise the numbers; if
    the geometry cannot support 70/15/15 it says so (``feasible=False``, with
    warnings naming the starved splits) and leaves the decision to a human.  A
    15% band that is narrower than ``2 * buffer_m + tile_extent`` cannot contain
    a single usable tile, and no amount of arithmetic changes that.

    Parameters
    ----------
    tiles :
        Tiles to assign (see :func:`assign_tiles_to_splits`).
    buffer_m, buffer_kind :
        Separation required between splits; ground metres by default.
    bounds :
        ROI rectangle in projected metres.  Defaults to the bounding box of the
        supplied tile footprints, which is the smallest honest ROI for this tile
        set.  No ROI centre is assumed (DECISIONS.md D-002).
    axis :
        Cut direction, ``"x"`` or ``"y"``.
    tolerance :
        Absolute tolerance on each achieved area ratio.

    Returns
    -------
    SplitPlan
    """
    tiles = list(tiles)
    if not tiles:
        raise ValueError("no tiles supplied; a split cannot be planned from nothing")
    if len(target_ratio) != len(names):
        raise ValueError(
            f"got {len(target_ratio)} target fractions for {len(names)} names")
    if len(set(names)) != len(names):
        raise ValueError(f"split names must be unique, got {list(names)}")
    if tolerance < 0:
        raise ValueError("tolerance must be >= 0")

    geoms = [footprint_of(t) for t in tiles]
    if bounds is None:
        xs0 = min(gm.bounds[0] for gm in geoms)
        ys0 = min(gm.bounds[1] for gm in geoms)
        xs1 = max(gm.bounds[2] for gm in geoms)
        ys1 = max(gm.bounds[3] for gm in geoms)
        bounds = (xs0, ys0, xs1, ys1)

    regions = block_regions(bounds, target_ratio, names, axis=axis)
    assignment = assign_tiles_to_splits(
        tiles, regions, buffer_m, buffer_kind=buffer_kind, source_ids=source_ids,
        crater_ids=crater_ids, body=body, lon_0=lon_0, k0=k0)

    target = {n: float(f) for n, f in zip(names, target_ratio)}
    region_area = {r.name: r.projected_area_m2 for r in regions}
    region_total = sum(region_area.values())
    region_ratio = {k: v / region_total for k, v in region_area.items()}
    compact = {r.name: r.compactness for r in regions}

    counts = {n: 0 for n in names}
    proj_area = {n: 0.0 for n in names}
    ground_area = {n: 0.0 for n in names}
    for a in assignment.assignments:
        if a.split is None:
            continue
        counts[a.split] += 1
        proj_area[a.split] += a.projected_area_m2
        ground_area[a.split] += a.ground_area_m2

    total_ground = sum(ground_area.values())
    total_count = sum(counts.values())
    nan = float("nan")
    area_ratio = ({n: ground_area[n] / total_ground for n in names}
                  if total_ground > 0 else {n: nan for n in names})
    count_ratio = ({n: counts[n] / total_count for n in names}
                   if total_count > 0 else {n: nan for n in names})

    discarded = assignment.discarded
    discarded_ground = sum(a.ground_area_m2 for a in discarded)
    denom = total_ground + discarded_ground
    discarded_fraction = (discarded_ground / denom) if denom > 0 else nan

    empty = tuple(n for n in names if counts[n] == 0)
    if total_ground > 0:
        max_dev = max(abs(area_ratio[n] - target[n]) for n in names)
    else:
        max_dev = nan
    feasible = (not empty) and (max_dev == max_dev) and (max_dev <= tolerance)

    warns: list[str] = []
    if empty:
        warns.append(
            f"split(s) {', '.join(empty)} received NO tiles: the band is too "
            f"narrow to hold a tile that is {buffer_m:g} m ({buffer_kind}) clear "
            "of its neighbours. The target ratio is not achievable with this "
            "ROI, tile size and buffer; it has NOT been forced."
        )
    if total_ground > 0 and not (max_dev <= tolerance):
        off = ", ".join(
            f"{n}: target {target[n]:.3f} achieved {area_ratio[n]:.3f}"
            for n in names if abs(area_ratio[n] - target[n]) > tolerance)
        warns.append(
            f"achieved area ratio deviates from the target by {max_dev:.3f} "
            f"(> tolerance {tolerance:.3f}): {off}. Reported as-is; region sizes "
            "were not adjusted to hit the target."
        )
    if discarded_fraction == discarded_fraction and discarded_fraction > 0.5:
        warns.append(
            f"{discarded_fraction:.1%} of the tiled ground area was discarded to "
            "the buffer zones; the usable dataset is less than half the ROI"
        )
    low = [n for n, c in compact.items() if c < 0.1]
    if low:
        warns.append(
            f"block(s) {', '.join(sorted(low))} have a compactness below 0.1, "
            "i.e. they are long and thin; such blocks maximise boundary "
            "attrition and are a sign the geometry is being used to trace "
            "around individual tiles rather than to define terrain"
        )
    for r in regions:
        if r.n_holes:
            warns.append(
                f"block {r.name!r} contains {r.n_holes} interior ring(s); holes "
                "must correspond to documented data gaps, not to excluded tiles"
            )

    ratio_txt = ", ".join(
        f"{n} {target[n]:.0%}->"
        + ("n/a" if area_ratio[n] != area_ratio[n] else f"{area_ratio[n]:.1%}")
        + f" ({counts[n]} tiles)"
        for n in names)
    message = (
        f"Block split along {axis} of a "
        f"{bounds[2] - bounds[0]:.0f} x {bounds[3] - bounds[1]:.0f} projected-m "
        f"ROI with a {buffer_m:g} m {buffer_kind} buffer: {ratio_txt}; "
        f"{len(discarded)} of {len(tiles)} tiles discarded "
        + ("" if discarded_fraction != discarded_fraction
           else f"({discarded_fraction:.1%} of ground area) ")
        + f"-- {'FEASIBLE' if feasible else 'TARGET NOT ACHIEVABLE'}."
        + ("" if not warns else " " + " ".join(warns))
    )
    if warns and emit_warning:
        _warnings.warn(message, SplitGeometryWarning, stacklevel=2)

    return SplitPlan(
        regions=regions,
        assignment=assignment,
        target_ratio=target,
        region_projected_area_m2=region_area,
        region_area_ratio=region_ratio,
        tile_counts=counts,
        usable_projected_area_m2=proj_area,
        usable_ground_area_m2=ground_area,
        achieved_area_ratio=area_ratio,
        achieved_count_ratio=count_ratio,
        discarded_tiles=len(discarded),
        discarded_ground_area_m2=float(discarded_ground),
        discarded_fraction=float(discarded_fraction),
        empty_splits=empty,
        max_abs_area_deviation=float(max_dev),
        tolerance=float(tolerance),
        feasible=bool(feasible),
        compactness=compact,
        buffer_m=float(buffer_m),
        buffer_kind=buffer_kind,
        axis=axis,
        warnings=tuple(warns),
        message=message,
    )
