"""YOLO box handling and pixel <-> ground diameter conversion.

Three representations are in play and this module is the only place they are
allowed to meet:

(a) **pixel xyxy** -- :class:`PixelBox`, continuous pixel coordinates inside a
    single tile.  ``(0.0, 0.0)`` is the *upper-left corner of the upper-left
    pixel*; ``(width_px, height_px)`` is the lower-right corner of the tile.
    ``x`` grows right (increasing projected easting), ``y`` grows *down*
    (decreasing projected northing), which is the usual north-up raster
    convention.  Pixel *centres* therefore sit at half-integers.
(b) **YOLO normalised** -- :class:`YoloBox`, ``(cx, cy, w, h)`` each in [0, 1],
    normalised by the tile size.  This is the on-disk label format.
(c) **catalogue crater** -- a lon/lat centre plus a *ground* diameter in metres.

Converting (c) -> (a) is the only direction that involves geodesy, and it is
done by sampling the rim with :func:`crater.geometry.crater_rim_samples` and
projecting those samples.  Degrees are never multiplied by a planar
metres-per-degree factor anywhere in this module (INTERFACES.md rule 1): at
86 S one degree of longitude is ~2.1 km, not ~30 km, so a planar factor would
be wrong by a factor of ~14.

Scale bookkeeping
-----------------
``TileFrame.pixel_size_m`` is **projected** metres per pixel, i.e. what a
GeoTIFF affine transform carries.  A length measured in projected metres must
be divided by :func:`crater.geometry.point_scale_factor` to become a ground
(planimetric great-circle) length -- see INTERFACES.md rule 2.  At the ROI
latitude ``k ~ 1.0012``, so ignoring it biases every diameter by +0.12%.
:func:`pixel_diameter_to_ground_m` is the only sanctioned conversion.

Honesty policy
--------------
Nothing in this module invents a number.  A conversion that cannot be
performed either raises :class:`ValueError` or returns the explicit
:data:`UNMEASURED` marker (INTERFACES.md rule 6).  In particular a box that
was clipped by a tile edge is *never* reported as a full-diameter
measurement; a clipped box bounds the crater from below only.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np

from . import geometry as g
from .body import MOON, Body

__all__ = [
    "UNMEASURED",
    "Unmeasured",
    "DIAMETER_MEASURE_LABEL",
    "APPROVED_DIAMETER_RANGE_M",
    "TileFrame",
    "PixelBox",
    "YoloBox",
    "ClipResult",
    "CraterBox",
    "GroundDiameter",
    "ProxyBias",
    "QualityFlags",
    "xyxy_to_yolo",
    "yolo_to_xyxy",
    "yolo_label_line",
    "parse_yolo_label_line",
    "clip_box_to_tile",
    "crater_to_pixel_box",
    "pixel_box_to_crater",
    "proxy_diameter_width_px",
    "proxy_diameter_height_px",
    "proxy_diameter_arithmetic_mean_px",
    "proxy_diameter_geometric_mean_px",
    "PIXEL_DIAMETER_PROXIES",
    "pixel_diameter_to_ground_m",
    "estimate_proxy_bias",
    "rim_discretisation_bias",
    "quality_flags",
]

#: The label every ground diameter produced here carries.  It is the short
#: form of :data:`crater.geometry.DIAMETER_DISTANCE_KIND`
#: ("planimetric_great_circle"); both are reported so a consumer cannot mistake
#: these for terrain-surface lengths (INTERFACES.md rule 3).
DIAMETER_MEASURE_LABEL = "planimetric"

#: Approved target range, DECISIONS.md D-003.  Used only for flagging; this
#: module never silently discards an out-of-range crater.
APPROVED_DIAMETER_RANGE_M: Tuple[float, float] = (20.0, 1000.0)

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Explicit "no measurement" marker
# --------------------------------------------------------------------------- #
class Unmeasured:
    """Singleton marker for "this quantity could not be measured".

    It is deliberately *not* ``None``, ``0.0`` or ``nan``: ``None`` is also
    used for "not supplied", ``0.0`` is a plausible-looking number, and ``nan``
    propagates silently through arithmetic.  ``bool(UNMEASURED)`` is ``False``
    so ``if diameter:`` guards behave sensibly.
    """

    __slots__ = ()
    _instance: Optional["Unmeasured"] = None

    def __new__(cls) -> "Unmeasured":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - trivial
        return "UNMEASURED"

    def __bool__(self) -> bool:
        return False


#: The marker instance.  Compare with ``is UNMEASURED``.
UNMEASURED = Unmeasured()

MaybeDiameter = Union["GroundDiameter", Unmeasured]


def _check_finite(name: str, value) -> float:
    if value is None:
        raise ValueError(f"{name} is required but was None")
    v = float(value)
    if not np.isfinite(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return v


# --------------------------------------------------------------------------- #
# Tile frame: the affine map between projected metres and tile pixels
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TileFrame:
    """One tile's georeferencing, in the project's south polar stereographic CRS.

    Parameters
    ----------
    tile_id:
        Opaque identifier, carried through to detections.
    x0_m, y0_m:
        Projected coordinates of the tile's upper-left *corner* (the corner of
        pixel (0, 0), not its centre).
    pixel_size_m:
        Square pixel size in **projected** metres.  Ground scale at latitude
        ``lat`` is ``pixel_size_m / point_scale_factor(lat)``.
    width_px, height_px:
        Tile size in pixels.
    lon_0, k0, body:
        Projection parameters, passed straight to :mod:`crater.geometry`.

    Notes
    -----
    This class is intentionally self-contained; it duplicates no logic from
    ``crater.tiling`` (owned by another agent) and makes no assumption about
    how tiles are laid out or where the ROI centre is (D-002).
    """

    tile_id: str
    x0_m: float
    y0_m: float
    pixel_size_m: float
    width_px: int
    height_px: int
    lon_0: float = 0.0
    k0: float = 1.0
    body: Body = MOON

    def __post_init__(self) -> None:
        for name in ("x0_m", "y0_m", "lon_0"):
            _check_finite(name, getattr(self, name))
        ps = _check_finite("pixel_size_m", self.pixel_size_m)
        if ps <= 0.0:
            raise ValueError(f"pixel_size_m must be positive, got {ps}")
        k0 = _check_finite("k0", self.k0)
        if k0 <= 0.0:
            raise ValueError(f"k0 must be positive, got {k0}")
        for name in ("width_px", "height_px"):
            v = getattr(self, name)
            if int(v) != v or v <= 0:
                raise ValueError(f"{name} must be a positive integer, got {v!r}")

    # -- affine map ------------------------------------------------------- #
    def to_pixel(self, x_m, y_m):
        """Projected metres -> continuous pixel (col, row)."""
        x = np.asarray(x_m, float)
        y = np.asarray(y_m, float)
        col = (x - self.x0_m) / self.pixel_size_m
        row = (self.y0_m - y) / self.pixel_size_m
        if np.ndim(x_m) or np.ndim(y_m):
            return col, row
        return float(col), float(row)

    def to_projected(self, col, row):
        """Continuous pixel (col, row) -> projected metres."""
        c = np.asarray(col, float)
        r = np.asarray(row, float)
        x = self.x0_m + c * self.pixel_size_m
        y = self.y0_m - r * self.pixel_size_m
        if np.ndim(col) or np.ndim(row):
            return x, y
        return float(x), float(y)

    def lonlat_to_pixel(self, lon_deg, lat_deg):
        x, y = g.forward(lon_deg, lat_deg, body=self.body,
                         lon_0=self.lon_0, k0=self.k0)
        return self.to_pixel(x, y)

    def pixel_to_lonlat(self, col, row):
        x, y = self.to_projected(col, row)
        return g.inverse(x, y, body=self.body, lon_0=self.lon_0, k0=self.k0)

    @property
    def tile_box(self) -> "PixelBox":
        """The tile's own extent as a :class:`PixelBox`."""
        return PixelBox(0.0, 0.0, float(self.width_px), float(self.height_px))

    def ground_pixel_scale_m(self, lat_deg) -> float:
        """Ground metres per pixel at ``lat_deg`` (projected scale / k)."""
        return self.pixel_size_m / g.point_scale_factor(lat_deg, k0=self.k0)


# --------------------------------------------------------------------------- #
# Boxes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class PixelBox:
    """Axis-aligned box in continuous tile-pixel coordinates (xyxy).

    ``x_min <= x_max`` and ``y_min <= y_max`` are enforced; a zero-extent box
    is allowed (a crater far smaller than one pixel) but is flagged downstream.
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "y_min", "x_max", "y_max"):
            _check_finite(name, getattr(self, name))
        if self.x_max < self.x_min or self.y_max < self.y_min:
            raise ValueError(
                "PixelBox requires x_min <= x_max and y_min <= y_max, got "
                f"({self.x_min}, {self.y_min}, {self.x_max}, {self.y_max})"
            )

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def centre(self) -> Tuple[float, float]:
        return (0.5 * (self.x_min + self.x_max), 0.5 * (self.y_min + self.y_max))

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x_min, self.y_min, self.x_max, self.y_max)


@dataclass(frozen=True)
class YoloBox:
    """YOLO normalised box: centre and size as fractions of the tile.

    All four values must lie in [0, 1] (``cx``, ``cy``) or (0, 1]
    (``w``, ``h``) to within ``tol``; a label file holding anything else is a
    bug, not something to clamp quietly.
    """

    cx: float
    cy: float
    w: float
    h: float
    class_id: int = 0

    def __post_init__(self) -> None:
        for name in ("cx", "cy", "w", "h"):
            _check_finite(name, getattr(self, name))
        if int(self.class_id) != self.class_id or self.class_id < 0:
            raise ValueError(f"class_id must be a non-negative int, got {self.class_id!r}")

    def validate(self, tol: float = 1e-9) -> "YoloBox":
        """Raise if the box is outside the YOLO domain.  Returns ``self``."""
        for name in ("cx", "cy", "w", "h"):
            v = getattr(self, name)
            if not (-tol <= v <= 1.0 + tol):
                raise ValueError(
                    f"YOLO {name}={v!r} is outside [0, 1]; normalise against the "
                    "correct tile size or clip the box first"
                )
        if self.w <= 0.0 or self.h <= 0.0:
            raise ValueError(f"YOLO w and h must be positive, got w={self.w}, h={self.h}")
        if self.cx - 0.5 * self.w < -tol or self.cx + 0.5 * self.w > 1.0 + tol:
            raise ValueError("YOLO box extends outside the tile in x; clip it first")
        if self.cy - 0.5 * self.h < -tol or self.cy + 0.5 * self.h > 1.0 + tol:
            raise ValueError("YOLO box extends outside the tile in y; clip it first")
        return self

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.cx, self.cy, self.w, self.h)


def xyxy_to_yolo(box: PixelBox, width_px: int, height_px: int,
                 class_id: int = 0, validate: bool = True) -> YoloBox:
    """Pixel xyxy -> YOLO normalised ``(cx, cy, w, h)``.

    Exact inverse of :func:`yolo_to_xyxy` for the same tile size.
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError("width_px and height_px must be positive")
    cx = 0.5 * (box.x_min + box.x_max) / width_px
    cy = 0.5 * (box.y_min + box.y_max) / height_px
    out = YoloBox(cx, cy, box.width / width_px, box.height / height_px,
                  class_id=class_id)
    return out.validate() if validate else out


def yolo_to_xyxy(yolo: YoloBox, width_px: int, height_px: int) -> PixelBox:
    """YOLO normalised -> pixel xyxy.  Exact inverse of :func:`xyxy_to_yolo`."""
    if width_px <= 0 or height_px <= 0:
        raise ValueError("width_px and height_px must be positive")
    half_w = 0.5 * yolo.w * width_px
    half_h = 0.5 * yolo.h * height_px
    cx = yolo.cx * width_px
    cy = yolo.cy * height_px
    return PixelBox(cx - half_w, cy - half_h, cx + half_w, cy + half_h)


def yolo_label_line(yolo: YoloBox, precision: int = 9) -> str:
    """One line of a YOLO ``.txt`` label file: ``class cx cy w h``."""
    return (f"{int(yolo.class_id)} "
            f"{yolo.cx:.{precision}f} {yolo.cy:.{precision}f} "
            f"{yolo.w:.{precision}f} {yolo.h:.{precision}f}")


def parse_yolo_label_line(line: str, validate: bool = True) -> YoloBox:
    """Inverse of :func:`yolo_label_line`.  Raises on a malformed line."""
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 whitespace-separated fields, got {len(parts)}: {line!r}")
    cls = int(parts[0])
    cx, cy, w, h = (float(p) for p in parts[1:])
    out = YoloBox(cx, cy, w, h, class_id=cls)
    return out.validate() if validate else out


# --------------------------------------------------------------------------- #
# Clipping policy
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ClipResult:
    """Outcome of applying the tile-edge clipping policy to one box.

    Attributes
    ----------
    box:
        The clipped box, or ``None`` when the object lies wholly outside the
        tile.  ``None`` is an explicit "not on this tile" report, not a silent
        drop: the caller still receives a record.
    full_box:
        The unclipped box, kept so the true extent is never lost.
    clipped:
        ``True`` when the tile edge cut the box, i.e. when ``box != full_box``.
    visible_fraction:
        Fraction of ``full_box``'s **area** inside the tile, in [0, 1].  This
        is a bounding-box fraction, not the fraction of the crater *disc*:
        for a circle cut by a straight edge the disc fraction is smaller than
        this for cuts past the centre and larger for shallow cuts.  It is used
        for flagging and sorting, never as a measurement.
    on_tile:
        ``visible_fraction > 0``.
    """

    box: Optional[PixelBox]
    full_box: PixelBox
    clipped: bool
    visible_fraction: float
    on_tile: bool


def clip_box_to_tile(box: PixelBox, width_px: int, height_px: int,
                     tol: float = 1e-9) -> ClipResult:
    """Clip ``box`` to ``[0, width_px] x [0, height_px]``.

    Policy (explicit, and the only one used in this project)
    -------------------------------------------------------
    1. A box fully inside the tile is returned unchanged with
       ``clipped=False`` and ``visible_fraction == 1``.
    2. A box crossing an edge is **truncated** to the tile and marked
       ``clipped=True``.  It is kept -- border craters are never dropped,
       because dropping them would bias the size-frequency distribution
       against large craters, which cross edges more often.
    3. A truncated box is not a diameter measurement.
       :func:`pixel_diameter_to_ground_m` refuses to turn one into a number and
       returns :data:`UNMEASURED`; the truncated extent is only a *lower
       bound* on the diameter.
    4. A box wholly outside the tile is reported with ``box=None`` and
       ``on_tile=False``.  Again a record is returned, never nothing.

    Degenerate (zero-area) boxes are handled by falling back to an
    intersection test, so a sub-pixel crater on the tile is still ``on_tile``.
    """
    if width_px <= 0 or height_px <= 0:
        raise ValueError("width_px and height_px must be positive")
    W = float(width_px)
    H = float(height_px)

    x_min = min(max(box.x_min, 0.0), W)
    x_max = min(max(box.x_max, 0.0), W)
    y_min = min(max(box.y_min, 0.0), H)
    y_max = min(max(box.y_max, 0.0), H)

    inter_w = max(0.0, x_max - x_min)
    inter_h = max(0.0, y_max - y_min)
    full_area = box.area

    if full_area > _EPS:
        visible = (inter_w * inter_h) / full_area
    else:
        # Degenerate box: area ratios are meaningless.  Fall back to "does it
        # touch the tile at all".
        touches = (box.x_max >= -tol and box.x_min <= W + tol
                   and box.y_max >= -tol and box.y_min <= H + tol)
        visible = 1.0 if touches else 0.0
    visible = float(min(max(visible, 0.0), 1.0))

    on_tile = visible > 0.0
    clipped = (box.x_min < -tol or box.y_min < -tol
               or box.x_max > W + tol or box.y_max > H + tol)

    clipped_box: Optional[PixelBox]
    if not on_tile:
        clipped_box = None
    elif not clipped:
        clipped_box = box
    else:
        clipped_box = PixelBox(x_min, y_min, x_max, y_max)
    return ClipResult(box=clipped_box, full_box=box, clipped=clipped,
                      visible_fraction=visible, on_tile=on_tile)


# --------------------------------------------------------------------------- #
# Catalogue crater -> pixel box
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CraterBox:
    """A catalogue crater expressed as a box on one tile.

    ``box`` is the clipped box actually drawn on the tile (``None`` if the
    crater misses the tile entirely); ``full_box`` is the unclipped envelope.
    ``centre_px`` is the projected crater *centre*, which for a clipped crater
    can legitimately lie outside the tile.
    """

    tile_id: str
    centre_lon_deg: float
    centre_lat_deg: float
    diameter_m: float
    box: Optional[PixelBox]
    full_box: PixelBox
    clipped: bool
    visible_fraction: float
    on_tile: bool
    centre_px: Tuple[float, float]
    n_rim_samples: int

    @property
    def clip(self) -> ClipResult:
        return ClipResult(self.box, self.full_box, self.clipped,
                          self.visible_fraction, self.on_tile)


def rim_discretisation_bias(n_samples: int) -> float:
    """Worst-case *relative* shortfall of a rim-sampled box extent.

    The box is the envelope of ``n_samples`` points on the rim, so along any
    given axis the extreme sample can sit up to half a sample step away from
    the true extremum.  The envelope half-width is therefore between
    ``cos(pi / n_samples)`` and ``1`` times the true radius, i.e. the box
    *under*-estimates the diameter by at most ``1 - cos(pi / n_samples)``.
    For ``n_samples=72`` that is 9.5e-4; for 360 it is 3.8e-5.

    This is a bias of the discretised representation, not of the proxy
    convention, and it is always negative (never an overestimate).
    """
    if n_samples < 3:
        raise ValueError("n_samples must be at least 3")
    return float(1.0 - np.cos(np.pi / n_samples))


def crater_to_pixel_box(centre_lon_deg: float, centre_lat_deg: float,
                        diameter_m: float, frame: TileFrame, *,
                        n_samples: int = 360,
                        body: Optional[Body] = None) -> CraterBox:
    """Build a tile box for a catalogue crater (lon/lat centre + ground diameter).

    The rim is sampled geodesically with
    :func:`crater.geometry.crater_rim_samples` and each sample is projected and
    mapped to pixels; the box is the axis-aligned envelope of those pixels.
    This is correct at any latitude, including across the pole and the
    antimeridian, because no step ever treats a degree as a fixed number of
    metres.

    ``n_samples`` defaults to 360 so the discretisation shortfall
    (:func:`rim_discretisation_bias`) is 3.8e-5 -- about 0.04 mm on a 1 m
    crater, i.e. far below any real measurement.
    """
    diameter_m = _check_finite("diameter_m", diameter_m)
    if diameter_m <= 0.0:
        raise ValueError(f"diameter_m must be positive, got {diameter_m}")
    _check_finite("centre_lon_deg", centre_lon_deg)
    _check_finite("centre_lat_deg", centre_lat_deg)
    b = frame.body if body is None else body

    lon, lat = g.crater_rim_samples(centre_lon_deg, centre_lat_deg, diameter_m,
                                    n_samples=n_samples, body=b)
    x, y = g.forward(lon, lat, body=b, lon_0=frame.lon_0, k0=frame.k0)
    col, row = frame.to_pixel(x, y)
    if not (np.all(np.isfinite(col)) and np.all(np.isfinite(row))):
        raise ValueError("rim projection produced non-finite pixel coordinates")

    full = PixelBox(float(np.min(col)), float(np.min(row)),
                    float(np.max(col)), float(np.max(row)))
    clip = clip_box_to_tile(full, frame.width_px, frame.height_px)
    centre_px = frame.lonlat_to_pixel(centre_lon_deg, centre_lat_deg)
    return CraterBox(
        tile_id=frame.tile_id,
        centre_lon_deg=float(centre_lon_deg),
        centre_lat_deg=float(centre_lat_deg),
        diameter_m=diameter_m,
        box=clip.box,
        full_box=clip.full_box,
        clipped=clip.clipped,
        visible_fraction=clip.visible_fraction,
        on_tile=clip.on_tile,
        centre_px=(float(centre_px[0]), float(centre_px[1])),
        n_rim_samples=int(n_samples),
    )


def pixel_box_to_crater(box: PixelBox, frame: TileFrame, *,
                        proxy: Union[str, Callable[[PixelBox], float]] = "geometric_mean",
                        clipped: Optional[bool] = None,
                        strict: bool = True,
                        ) -> Tuple[float, float, "MaybeDiameter"]:
    """Inverse direction: pixel box -> ``(lon_deg, lat_deg, ground diameter)``.

    The centre is unprojected from the box centre, and the diameter comes from
    ``proxy`` via :func:`pixel_diameter_to_ground_m` *at the box-centre
    latitude*, so the scale factor is evaluated where the crater actually is.

    A clipped box yields :data:`UNMEASURED` for the diameter while still
    returning a usable centre, because the centre of a half-visible crater is
    still informative for :mod:`crater.dedup` even when its size is not.
    Note that the returned centre is the *box* centre, which for a truncated
    box is displaced from the crater centre.

    ``clipped`` should be supplied by callers that still hold the flag (from
    :class:`CraterBox` or :class:`ClipResult`), because a box that has already
    been truncated to the tile is numerically indistinguishable from one that
    always fitted.  When it is ``None`` the flag is derived *conservatively*:
    a box that overflows the tile **or merely sits flush against a tile edge**
    is treated as clipped, so the failure mode is a refused measurement rather
    than a silently truncated diameter.
    """
    if clipped is None:
        tol = 1e-9
        clipped = (
            clip_box_to_tile(box, frame.width_px, frame.height_px).clipped
            or box.x_min <= tol or box.y_min <= tol
            or box.x_max >= frame.width_px - tol
            or box.y_max >= frame.height_px - tol
        )
    col, row = box.centre
    lon, lat = frame.pixel_to_lonlat(col, row)
    fn, name = _resolve_proxy(proxy)
    diameter = pixel_diameter_to_ground_m(
        fn(box), frame.pixel_size_m, lat, proxy=name, clipped=bool(clipped),
        k0=frame.k0, strict=strict)
    return float(lon), float(lat), diameter


# --------------------------------------------------------------------------- #
# Diameter proxies from a bounding box
# --------------------------------------------------------------------------- #
# Each proxy is a separate named function because the choice of convention is a
# scientific decision that must be recorded per product, not a keyword buried
# in a call.  All return PIXELS; use pixel_diameter_to_ground_m to get metres.
def proxy_diameter_width_px(box: PixelBox) -> float:
    """Proxy = box width (x extent).

    Convention: the horizontal extent only.
    Bias: unbiased for a circle whose box is tight.  For an elongated or
    obliquely illuminated crater it measures the x-projection of the shape, so
    it is biased by the crater's orientation in the tile -- high for a crater
    elongated along x, low for one elongated along y.  Near the pole the tile's
    x axis is not a fixed compass direction, so this bias is also a function of
    longitude, which makes ``width`` the worst choice for a polar survey.
    """
    return float(box.width)


def proxy_diameter_height_px(box: PixelBox) -> float:
    """Proxy = box height (y extent).

    Convention: the vertical extent only.
    Bias: the mirror image of :func:`proxy_diameter_width_px` -- same
    orientation sensitivity with the opposite sign.
    """
    return float(box.height)


def proxy_diameter_arithmetic_mean_px(box: PixelBox) -> float:
    """Proxy = ``(width + height) / 2``.

    Convention: the mean of the two box extents; equivalently the diameter of
    the circle with the same *perimeter* as the box.
    Bias: by AM >= GM it is always >= :func:`proxy_diameter_geometric_mean_px`,
    with equality only for a square box.  It therefore *over*-estimates the
    equal-area equivalent diameter of any non-square box, and the overestimate
    grows as ``(1 + e)/(2 sqrt(e))`` for aspect ratio ``e`` (1.03 at e=1.5,
    1.25 at e=4).  It is the more robust of the two to one extent being
    corrupted, because it is linear.
    """
    return 0.5 * (float(box.width) + float(box.height))


def proxy_diameter_geometric_mean_px(box: PixelBox) -> float:
    """Proxy = ``sqrt(width * height)``.

    Convention: the side of the square with the same *area* as the box, i.e.
    the equal-area equivalent diameter.  This is the convention this project
    recommends, because crater areas (and hence the size-frequency
    distribution) are what the science uses.
    Bias: always <= the arithmetic mean; unbiased for a circle.  Being
    multiplicative it collapses to 0 if either extent collapses, so a box
    clipped to a sliver gives a catastrophically small value -- which is one
    more reason a clipped box must not be converted at all.
    """
    return float(np.sqrt(float(box.width) * float(box.height)))


#: Registry so a product can record the proxy it used by name.
PIXEL_DIAMETER_PROXIES: dict = {
    "width": proxy_diameter_width_px,
    "height": proxy_diameter_height_px,
    "arithmetic_mean": proxy_diameter_arithmetic_mean_px,
    "geometric_mean": proxy_diameter_geometric_mean_px,
}


def _resolve_proxy(proxy: Union[str, Callable[[PixelBox], float]]):
    if callable(proxy):
        return proxy, getattr(proxy, "__name__", "callable")
    try:
        return PIXEL_DIAMETER_PROXIES[proxy], proxy
    except KeyError:
        raise ValueError(
            f"unknown diameter proxy {proxy!r}; choose one of "
            f"{sorted(PIXEL_DIAMETER_PROXIES)}"
        ) from None


# --------------------------------------------------------------------------- #
# Pixel diameter -> ground diameter
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GroundDiameter:
    """A ground diameter in metres, with the provenance needed to trust it.

    ``label`` is always :data:`DIAMETER_MEASURE_LABEL` ("planimetric") and
    ``distance_kind`` always :data:`crater.geometry.DIAMETER_DISTANCE_KIND`.
    These are great-circle lengths on the Moon 2000 sphere; they are **not**
    terrain-surface lengths, so on a sloped crater wall the true surface
    distance is longer.
    """

    value_m: float
    label: str = DIAMETER_MEASURE_LABEL
    distance_kind: str = g.DIAMETER_DISTANCE_KIND
    pixel_diameter_px: float = float("nan")
    projected_pixel_scale_m: float = float("nan")
    scale_factor_k: float = float("nan")
    latitude_deg: float = float("nan")
    proxy: str = "unspecified"


def pixel_diameter_to_ground_m(
    pixel_diameter_px: float,
    projected_pixel_scale_m: float,
    lat_deg: Optional[float],
    *,
    proxy: str = "unspecified",
    clipped: bool = False,
    k0: float = 1.0,
    strict: bool = True,
) -> MaybeDiameter:
    """Convert a pixel diameter to a ground (planimetric) diameter in metres.

    ``ground = pixel_diameter_px * projected_pixel_scale_m / k(lat_deg)``

    The division by ``k`` is mandatory (INTERFACES.md rule 2): the pixel scale
    of a polar stereographic product is in *projected* metres, which at 86 S
    are 0.12% shorter on the ground than they look.

    Returns
    -------
    GroundDiameter or UNMEASURED
        :data:`UNMEASURED` is returned -- never a number -- when
        * ``clipped`` is ``True``.  A truncated box bounds the diameter from
          below only; reporting it as a diameter would quietly bias the
          size-frequency distribution low.
        * ``strict=False`` and an input is invalid.

    Raises
    ------
    ValueError
        When ``strict`` (the default) and the latitude is missing, any input is
        non-finite, or the pixel diameter or pixel scale is non-positive.
    """
    if clipped:
        # Policy, not an error: see clip_box_to_tile.
        return UNMEASURED
    try:
        d_px = _check_finite("pixel_diameter_px", pixel_diameter_px)
        scale = _check_finite("projected_pixel_scale_m", projected_pixel_scale_m)
        if lat_deg is None:
            raise ValueError("lat_deg is required: the scale factor k depends on it")
        lat = _check_finite("lat_deg", lat_deg)
        if not (-90.0 <= lat <= 90.0):
            raise ValueError(f"lat_deg must be within [-90, 90], got {lat}")
        if d_px <= 0.0:
            raise ValueError(f"pixel_diameter_px must be positive, got {d_px}")
        if scale <= 0.0:
            raise ValueError(f"projected_pixel_scale_m must be positive, got {scale}")
        k = float(g.point_scale_factor(lat, k0=k0))
        if not np.isfinite(k) or k <= 0.0:
            raise ValueError(f"point scale factor is not usable at lat={lat}: k={k}")
    except ValueError:
        if strict:
            raise
        return UNMEASURED

    return GroundDiameter(
        value_m=d_px * scale / k,
        pixel_diameter_px=d_px,
        projected_pixel_scale_m=scale,
        scale_factor_k=k,
        latitude_deg=lat,
        proxy=proxy,
    )


# --------------------------------------------------------------------------- #
# Validating a proxy convention against known ground truth
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProxyBias:
    """Measured bias of one diameter proxy against a known circular crater."""

    proxy: str
    true_diameter_m: float
    estimated_diameter_m: float
    ratio: float            # estimated / true
    relative_bias: float    # ratio - 1
    centre_lon_deg: float
    centre_lat_deg: float
    n_rim_samples: int
    discretisation_bound: float


def estimate_proxy_bias(proxy: Union[str, Callable[[PixelBox], float]],
                        centre_lon_deg: float, centre_lat_deg: float,
                        diameter_m: float, frame: TileFrame, *,
                        n_samples: int = 360,
                        body: Optional[Body] = None) -> ProxyBias:
    """Measure a proxy's bias against a crater of *known* circular ground diameter.

    A synthetic crater is placed at ``(centre_lon_deg, centre_lat_deg)``, boxed
    with :func:`crater_to_pixel_box`, reduced to a pixel diameter by ``proxy``,
    converted back to metres with :func:`pixel_diameter_to_ground_m`, and
    compared with the input.  This is what makes the proxy convention a
    *validated* choice instead of an assumption.

    Raises
    ------
    ValueError
        If the synthetic crater is clipped by the tile edge -- a clipped box
        carries no diameter, so no bias could be attributed to the proxy.
        Place the crater well inside the frame.
    """
    fn, name = _resolve_proxy(proxy)
    cb = crater_to_pixel_box(centre_lon_deg, centre_lat_deg, diameter_m, frame,
                             n_samples=n_samples, body=body)
    if cb.clipped or cb.box is None:
        raise ValueError(
            "cannot estimate proxy bias from a clipped crater: the box is a "
            "lower bound on the diameter, not a measurement"
        )
    d_px = fn(cb.box)
    ground = pixel_diameter_to_ground_m(
        d_px, frame.pixel_size_m, centre_lat_deg, proxy=name, k0=frame.k0)
    assert isinstance(ground, GroundDiameter)  # unclipped + validated inputs
    ratio = ground.value_m / diameter_m
    return ProxyBias(
        proxy=name,
        true_diameter_m=float(diameter_m),
        estimated_diameter_m=ground.value_m,
        ratio=ratio,
        relative_bias=ratio - 1.0,
        centre_lon_deg=float(centre_lon_deg),
        centre_lat_deg=float(centre_lat_deg),
        n_rim_samples=int(n_samples),
        discretisation_bound=rim_discretisation_bias(n_samples),
    )


# --------------------------------------------------------------------------- #
# Quality flags
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class QualityFlags:
    """The per-detection flags this project carries to every downstream stage.

    Attributes
    ----------
    edge_flag:
        The box touches or crosses the tile boundary (within ``edge_margin_px``),
        or the object is off-tile.  Edge objects are kept but must be handled by
        :mod:`crater.dedup` across the tile overlap before any count is made.
    clipped:
        The tile edge actually truncated the box.
    low_confidence:
        Detector confidence below ``confidence_threshold``, or missing.
    unmeasured:
        **No valid diameter exists for this detection.**  True whenever the
        diameter is :data:`UNMEASURED`/missing/non-finite, or the box was
        clipped, or the object is off-tile.  Downstream size-frequency work
        must exclude ``unmeasured`` detections from the diameter histogram
        while still counting them as detections.
    reasons:
        Human-readable, machine-stable reason strings, sorted.  Extra context
        (e.g. a diameter outside the approved 20-1000 m range) appears here
        without changing the four headline flags.
    """

    edge_flag: bool
    clipped: bool
    low_confidence: bool
    unmeasured: bool
    reasons: Tuple[str, ...] = ()
    visible_fraction: Optional[float] = None


def quality_flags(*,
                  box: Optional[PixelBox] = None,
                  frame: Optional[TileFrame] = None,
                  confidence: Optional[float] = None,
                  ground_diameter: Optional[MaybeDiameter] = None,
                  visible_fraction: Optional[float] = None,
                  clipped: Optional[bool] = None,
                  on_tile: bool = True,
                  confidence_threshold: float = 0.25,
                  edge_margin_px: float = 1.0,
                  diameter_range_m: Tuple[float, float] = APPROVED_DIAMETER_RANGE_M,
                  ) -> QualityFlags:
    """Compute :class:`QualityFlags` for one detection.

    Every argument is keyword-only and optional so the helper can be used both
    on a catalogue-derived :class:`CraterBox` and on a raw detector output.
    Missing information is treated pessimistically: an absent confidence is
    ``low_confidence``, an absent diameter is ``unmeasured``.  Nothing here
    deletes a detection.
    """
    reasons = set()

    if clipped is None:
        clipped = False
        if box is not None and frame is not None:
            clipped = clip_box_to_tile(box, frame.width_px, frame.height_px).clipped
    clipped = bool(clipped)
    if clipped:
        reasons.add("clipped_by_tile_edge")

    # -- edge flag -------------------------------------------------------- #
    edge = clipped or not on_tile
    if not on_tile:
        reasons.add("off_tile")
    if box is not None and frame is not None and edge_margin_px >= 0.0:
        if (box.x_min <= edge_margin_px
                or box.y_min <= edge_margin_px
                or box.x_max >= frame.width_px - edge_margin_px
                or box.y_max >= frame.height_px - edge_margin_px):
            edge = True
            reasons.add("touches_tile_edge")
    if visible_fraction is not None and visible_fraction < 1.0:
        edge = True
        reasons.add("partially_visible")

    # -- confidence ------------------------------------------------------- #
    if confidence is None:
        low_conf = True
        reasons.add("confidence_missing")
    else:
        c = float(confidence)
        if not np.isfinite(c):
            low_conf = True
            reasons.add("confidence_non_finite")
        else:
            low_conf = c < confidence_threshold
            if low_conf:
                reasons.add("confidence_below_threshold")

    # -- measurability ---------------------------------------------------- #
    unmeasured = False
    if clipped:
        unmeasured = True
        reasons.add("diameter_is_lower_bound_only")
    if not on_tile:
        unmeasured = True
    if ground_diameter is None or ground_diameter is UNMEASURED:
        unmeasured = True
        reasons.add("diameter_unmeasured")
    elif isinstance(ground_diameter, GroundDiameter):
        v = ground_diameter.value_m
        if not np.isfinite(v) or v <= 0.0:
            unmeasured = True
            reasons.add("diameter_non_finite_or_non_positive")
        else:
            lo, hi = diameter_range_m
            if v < lo or v > hi:
                reasons.add("diameter_outside_approved_range")
    else:
        value = float(ground_diameter)
        if not np.isfinite(value) or value <= 0.0:
            unmeasured = True
            reasons.add("diameter_non_finite_or_non_positive")
        else:
            lo, hi = diameter_range_m
            if value < lo or value > hi:
                reasons.add("diameter_outside_approved_range")
    if box is not None and (box.width <= 0.0 or box.height <= 0.0):
        unmeasured = True
        reasons.add("degenerate_box")

    return QualityFlags(
        edge_flag=bool(edge),
        clipped=clipped,
        low_confidence=bool(low_conf),
        unmeasured=bool(unmeasured),
        reasons=tuple(sorted(reasons)),
        visible_fraction=None if visible_fraction is None else float(visible_fraction),
    )
