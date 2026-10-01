"""Tests for YOLO box handling and pixel <-> ground diameter conversion.

Every case is synthetic with a known answer: a crater is defined by a lon/lat
centre and a ground diameter, turned into a box, and measured back.  Nothing
here depends on imagery, the network, or the (still unset, D-002) ROI centre --
every tile frame is built explicitly by the test.
"""
import numpy as np
import pytest

from crater import boxes as bx
from crater import geometry as g
from crater.body import MOON

# The ROI latitude band (D-002 leaves the centre open, so the longitude here is
# arbitrary and only the latitude is load-bearing) and a couple of controls.
ROI_LON, ROI_LAT = 2.9, -85.9
DEEP_LAT = -86.0


def frame_with_centre_at_pixel(lon_deg, lat_deg, col, row, *,
                               pixel_size_m=1.0, width_px=1024, height_px=1024,
                               tile_id="T", lon_0=0.0):
    """A TileFrame placing (lon, lat) at continuous pixel (col, row)."""
    x, y = g.forward(lon_deg, lat_deg, lon_0=lon_0)
    return bx.TileFrame(tile_id=tile_id,
                        x0_m=x - col * pixel_size_m,
                        y0_m=y + row * pixel_size_m,
                        pixel_size_m=pixel_size_m,
                        width_px=width_px, height_px=height_px,
                        lon_0=lon_0)


def frame_centred_on(lon_deg, lat_deg, **kw):
    w = kw.get("width_px", 1024)
    h = kw.get("height_px", 1024)
    return frame_with_centre_at_pixel(lon_deg, lat_deg, w / 2.0, h / 2.0, **kw)


# --------------------------------------------------------------------- #
# Conventions and labelling
# --------------------------------------------------------------------- #
def test_diameter_label_matches_geometry_kind():
    """The module's label must be the short form of geometry's kind string,
    so a product cannot claim 'planimetric' while geometry means something else."""
    assert bx.DIAMETER_MEASURE_LABEL == "planimetric"
    assert bx.DIAMETER_MEASURE_LABEL in g.DIAMETER_DISTANCE_KIND


def test_ground_diameter_carries_both_labels():
    gd = bx.pixel_diameter_to_ground_m(100.0, 1.0, ROI_LAT)
    assert gd.label == "planimetric"
    assert gd.distance_kind == g.DIAMETER_DISTANCE_KIND


def test_unmeasured_is_a_distinct_falsy_singleton():
    assert bx.UNMEASURED is bx.Unmeasured()
    assert not bx.UNMEASURED
    assert bx.UNMEASURED is not None
    assert bx.UNMEASURED != 0.0


def test_tile_frame_pixel_affine_roundtrip():
    f = frame_centred_on(ROI_LON, ROI_LAT, pixel_size_m=1.5)
    for col, row in [(0.0, 0.0), (1024.0, 1024.0), (37.25, 901.75), (512.0, 512.0)]:
        x, y = f.to_projected(col, row)
        c2, r2 = f.to_pixel(x, y)
        assert (c2, r2) == pytest.approx((col, row), abs=1e-9)


def test_tile_frame_y_axis_points_down():
    """Row increases southward in projected y, the north-up raster convention."""
    f = frame_centred_on(ROI_LON, ROI_LAT)
    _, y_top = f.to_projected(0.0, 0.0)
    _, y_bottom = f.to_projected(0.0, 100.0)
    assert y_top > y_bottom


def test_tile_frame_rejects_invalid_geometry():
    for kw in ({"pixel_size_m": 0.0}, {"pixel_size_m": -1.0},
               {"pixel_size_m": float("nan")}, {"width_px": 0},
               {"height_px": -5}, {"k0": 0.0}, {"x0_m": float("inf")}):
        base = dict(tile_id="T", x0_m=0.0, y0_m=0.0, pixel_size_m=1.0,
                    width_px=8, height_px=8)
        base.update(kw)
        with pytest.raises(ValueError):
            bx.TileFrame(**base)


# --------------------------------------------------------------------- #
# xyxy <-> YOLO round trips
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("box", [
    bx.PixelBox(0.0, 0.0, 1024.0, 1024.0),     # the whole tile
    bx.PixelBox(10.0, 20.0, 30.0, 44.0),
    bx.PixelBox(511.5, 511.5, 512.5, 512.5),   # one pixel at the centre
    bx.PixelBox(0.0, 1000.0, 24.0, 1024.0),    # flush against two edges
    bx.PixelBox(100.25, 200.125, 300.375, 700.875),
])
def test_xyxy_yolo_roundtrip_is_exact(box):
    W = H = 1024
    yolo = bx.xyxy_to_yolo(box, W, H)
    back = bx.yolo_to_xyxy(yolo, W, H)
    assert back.as_tuple() == pytest.approx(box.as_tuple(), abs=1e-9)


def test_xyxy_yolo_roundtrip_on_non_square_tile():
    box = bx.PixelBox(13.0, 7.0, 613.0, 207.0)
    W, H = 800, 300
    yolo = bx.xyxy_to_yolo(box, W, H)
    assert yolo.cx == pytest.approx(313.0 / 800.0)
    assert yolo.cy == pytest.approx(107.0 / 300.0)
    assert yolo.w == pytest.approx(600.0 / 800.0)
    assert yolo.h == pytest.approx(200.0 / 300.0)
    assert bx.yolo_to_xyxy(yolo, W, H).as_tuple() == pytest.approx(box.as_tuple(), abs=1e-9)


def test_yolo_label_line_roundtrip():
    box = bx.PixelBox(100.0, 200.0, 340.0, 440.0)
    yolo = bx.xyxy_to_yolo(box, 1024, 1024, class_id=3)
    line = bx.yolo_label_line(yolo)
    assert line.split()[0] == "3"
    parsed = bx.parse_yolo_label_line(line)
    assert parsed.class_id == 3
    assert parsed.as_tuple() == pytest.approx(yolo.as_tuple(), abs=1e-9)
    back = bx.yolo_to_xyxy(parsed, 1024, 1024)
    assert back.as_tuple() == pytest.approx(box.as_tuple(), abs=1e-6)


def test_yolo_rejects_out_of_domain_and_overflowing_boxes():
    with pytest.raises(ValueError, match="outside"):
        bx.YoloBox(1.2, 0.5, 0.1, 0.1).validate()
    with pytest.raises(ValueError, match="outside the tile in x"):
        bx.YoloBox(0.95, 0.5, 0.2, 0.1).validate()
    with pytest.raises(ValueError, match="outside the tile in y"):
        bx.YoloBox(0.5, 0.02, 0.1, 0.2).validate()
    with pytest.raises(ValueError, match="positive"):
        bx.YoloBox(0.5, 0.5, 0.0, 0.1).validate()


def test_yolo_conversion_of_an_unclipped_box_is_rejected_not_clamped():
    """A crater crossing the edge must be clipped deliberately, not normalised
    into an invalid label silently."""
    box = bx.PixelBox(-50.0, 100.0, 50.0, 200.0)
    with pytest.raises(ValueError):
        bx.xyxy_to_yolo(box, 1024, 1024)


def test_parse_yolo_label_line_rejects_malformed():
    with pytest.raises(ValueError):
        bx.parse_yolo_label_line("0 0.5 0.5 0.1")
    with pytest.raises(ValueError):
        bx.parse_yolo_label_line("0 0.5 0.5 0.1 0.1 0.1")


def test_pixel_box_rejects_inverted_and_non_finite():
    with pytest.raises(ValueError):
        bx.PixelBox(10.0, 0.0, 5.0, 10.0)
    with pytest.raises(ValueError):
        bx.PixelBox(0.0, 0.0, float("nan"), 10.0)


# --------------------------------------------------------------------- #
# Catalogue crater -> box, and back to a ground diameter
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("diameter_m", [20.0, 50.0, 137.0, 300.0, 1000.0])
@pytest.mark.parametrize("lat", [ROI_LAT, DEEP_LAT, -70.0, -89.5])
def test_crater_box_is_square_and_sized_by_the_projected_scale(diameter_m, lat):
    """Stereographic maps a small ground circle to a circle, so the box is
    square with side D * k / pixel_size in projected pixels."""
    f = frame_centred_on(ROI_LON, lat, pixel_size_m=0.5, width_px=4096, height_px=4096)
    cb = bx.crater_to_pixel_box(ROI_LON, lat, diameter_m, f)
    assert not cb.clipped and cb.on_tile
    assert cb.visible_fraction == pytest.approx(1.0)
    expected_px = diameter_m * g.point_scale_factor(lat) / f.pixel_size_m
    assert cb.full_box.width == pytest.approx(expected_px, rel=1e-4)
    assert cb.full_box.height == pytest.approx(expected_px, rel=1e-4)
    assert cb.full_box.width == pytest.approx(cb.full_box.height, rel=1e-6)


@pytest.mark.parametrize("diameter_m", [20.0, 137.0, 1000.0])
@pytest.mark.parametrize("proxy", sorted(bx.PIXEL_DIAMETER_PROXIES))
def test_crater_roundtrip_through_box_recovers_the_ground_diameter(diameter_m, proxy):
    """crater -> box -> pixel proxy -> ground metres must return the input."""
    f = frame_centred_on(ROI_LON, ROI_LAT, pixel_size_m=1.0)
    cb = bx.crater_to_pixel_box(ROI_LON, ROI_LAT, diameter_m, f)
    d_px = bx.PIXEL_DIAMETER_PROXIES[proxy](cb.box)
    gd = bx.pixel_diameter_to_ground_m(d_px, f.pixel_size_m, ROI_LAT, proxy=proxy)
    assert gd.value_m == pytest.approx(diameter_m, rel=1e-4)


def test_crater_roundtrip_through_yolo_normalisation():
    """The full on-disk path: crater -> box -> YOLO line -> box -> diameter."""
    f = frame_centred_on(ROI_LON, ROI_LAT, pixel_size_m=1.0)
    D = 250.0
    cb = bx.crater_to_pixel_box(ROI_LON, ROI_LAT, D, f)
    line = bx.yolo_label_line(bx.xyxy_to_yolo(cb.box, f.width_px, f.height_px))
    box2 = bx.yolo_to_xyxy(bx.parse_yolo_label_line(line), f.width_px, f.height_px)
    assert box2.as_tuple() == pytest.approx(cb.box.as_tuple(), abs=1e-5)
    gd = bx.pixel_diameter_to_ground_m(
        bx.proxy_diameter_geometric_mean_px(box2), f.pixel_size_m, ROI_LAT)
    assert gd.value_m == pytest.approx(D, rel=1e-5)


@pytest.mark.parametrize("diameter_m", [20.0, 137.0, 1000.0])
@pytest.mark.parametrize("lat", [ROI_LAT, DEEP_LAT, -89.0])
def test_full_catalogue_roundtrip_crater_to_box_to_crater(diameter_m, lat):
    """(c) -> (a) -> (c): centre and ground diameter must both come back."""
    f = frame_centred_on(ROI_LON, lat, pixel_size_m=0.5, width_px=4096,
                         height_px=4096)
    cb = bx.crater_to_pixel_box(ROI_LON, lat, diameter_m, f)
    lon2, lat2, gd = bx.pixel_box_to_crater(cb.box, f, clipped=cb.clipped)
    # The box centre is the centre of the *projected* rim circle, which sits a
    # second-order distance from the projection of the spherical centre
    # (2.5 mm on a 1 km crater at 86 S).  Scale the tolerance with the crater.
    assert g.great_circle_distance(ROI_LON, lat, lon2, lat2) < 1e-5 * diameter_m
    assert gd.value_m == pytest.approx(diameter_m, rel=1e-4)
    assert gd.label == "planimetric"
    assert gd.proxy == "geometric_mean"


def test_box_to_crater_of_a_clipped_box_gives_a_centre_but_no_diameter():
    f = frame_with_centre_at_pixel(ROI_LON, ROI_LAT, 0.0, 512.0)
    cb = bx.crater_to_pixel_box(ROI_LON, ROI_LAT, 300.0, f)
    lon2, lat2, gd = bx.pixel_box_to_crater(cb.box, f, clipped=cb.clipped)
    assert gd is bx.UNMEASURED
    # With no flag supplied the conservative default must reach the same
    # verdict, because the box sits flush against the tile edge.
    assert bx.pixel_box_to_crater(cb.box, f)[2] is bx.UNMEASURED
    assert np.isfinite(lon2) and np.isfinite(lat2)
    # The centre of the *visible* half is offset from the true centre, as it
    # must be -- it is the box centre, not the crater centre.
    assert 50.0 < g.great_circle_distance(ROI_LON, ROI_LAT, lon2, lat2) < 120.0


def test_crater_box_near_the_antimeridian_does_not_wrap_apart():
    """A crater straddling lon = 180 must give a compact box, not one spanning
    the tile -- the rim is projected, never differenced in degrees."""
    f = frame_centred_on(180.0, ROI_LAT, pixel_size_m=1.0)
    cb = bx.crater_to_pixel_box(180.0, ROI_LAT, 300.0, f)
    assert not cb.clipped
    assert cb.full_box.width == pytest.approx(300.0 * g.point_scale_factor(ROI_LAT),
                                             rel=1e-4)


def test_crater_box_spanning_the_pole_is_finite():
    f = frame_centred_on(0.0, -89.995, pixel_size_m=1.0, width_px=4096, height_px=4096)
    cb = bx.crater_to_pixel_box(0.0, -89.995, 1000.0, f)
    assert cb.on_tile and not cb.clipped
    assert np.isfinite(cb.full_box.as_tuple()).all()
    assert cb.full_box.width == pytest.approx(1000.0, rel=1e-3)


def test_crater_box_rejects_bad_input():
    f = frame_centred_on(ROI_LON, ROI_LAT)
    for bad in (0.0, -10.0, float("nan"), float("inf")):
        with pytest.raises(ValueError):
            bx.crater_to_pixel_box(ROI_LON, ROI_LAT, bad, f)
    with pytest.raises(ValueError):
        bx.crater_to_pixel_box(ROI_LON, float("nan"), 100.0, f)


# --------------------------------------------------------------------- #
# The 86 S scale factor
# --------------------------------------------------------------------- #
def test_scale_factor_is_applied_and_matters_at_86_south():
    """Forgetting to divide by k inflates every diameter by 0.128% at 86 S.
    That is ~1.3 m on a 1 km crater and 0.026 m on a 20 m one: small, but a
    systematic bias in the same direction for every crater in the survey."""
    f = frame_centred_on(ROI_LON, DEEP_LAT, pixel_size_m=1.0, width_px=4096,
                         height_px=4096)
    D = 500.0
    cb = bx.crater_to_pixel_box(ROI_LON, DEEP_LAT, D, f)
    d_px = bx.proxy_diameter_geometric_mean_px(cb.box)

    k = g.point_scale_factor(DEEP_LAT)
    assert k == pytest.approx(1.0012, abs=5e-4)

    correct = bx.pixel_diameter_to_ground_m(d_px, f.pixel_size_m, DEEP_LAT)
    assert correct.value_m == pytest.approx(D, rel=1e-5)
    assert correct.scale_factor_k == pytest.approx(k, rel=1e-12)

    naive_projected_metres = d_px * f.pixel_size_m  # the bug this guards against
    assert naive_projected_metres / D == pytest.approx(k, rel=1e-5)
    assert naive_projected_metres - D > 0.6  # a real, one-signed bias


def test_scale_factor_differs_between_latitudes_for_the_same_crater():
    """The same ground diameter occupies more pixels further from the pole."""
    D = 300.0
    boxes = {}
    for lat in (-89.9, DEEP_LAT, -70.0):
        f = frame_centred_on(ROI_LON, lat, pixel_size_m=0.5, width_px=4096,
                             height_px=4096)
        cb = bx.crater_to_pixel_box(ROI_LON, lat, D, f)
        boxes[lat] = bx.proxy_diameter_geometric_mean_px(cb.box)
        gd = bx.pixel_diameter_to_ground_m(boxes[lat], f.pixel_size_m, lat)
        assert gd.value_m == pytest.approx(D, rel=1e-4)   # k removes the difference
    assert boxes[-70.0] > boxes[DEEP_LAT] > boxes[-89.9]


def test_pixel_diameter_conversion_rejects_invalid_scale_and_geometry():
    with pytest.raises(ValueError, match="lat_deg is required"):
        bx.pixel_diameter_to_ground_m(100.0, 1.0, None)
    with pytest.raises(ValueError, match="finite"):
        bx.pixel_diameter_to_ground_m(100.0, 1.0, float("nan"))
    with pytest.raises(ValueError, match="finite"):
        bx.pixel_diameter_to_ground_m(float("inf"), 1.0, ROI_LAT)
    with pytest.raises(ValueError, match="positive"):
        bx.pixel_diameter_to_ground_m(100.0, 0.0, ROI_LAT)
    with pytest.raises(ValueError, match="positive"):
        bx.pixel_diameter_to_ground_m(100.0, -1.0, ROI_LAT)
    with pytest.raises(ValueError, match="positive"):
        bx.pixel_diameter_to_ground_m(0.0, 1.0, ROI_LAT)
    with pytest.raises(ValueError, match=r"\[-90, 90\]"):
        bx.pixel_diameter_to_ground_m(100.0, 1.0, -95.0)


@pytest.mark.parametrize("args", [
    (100.0, 1.0, None), (100.0, 1.0, float("nan")), (100.0, 0.0, ROI_LAT),
    (float("nan"), 1.0, ROI_LAT), (-3.0, 1.0, ROI_LAT),
])
def test_invalid_conversion_returns_unmeasured_not_a_number(args):
    """Non-strict mode must still never produce a plausible-looking value."""
    out = bx.pixel_diameter_to_ground_m(*args, strict=False)
    assert out is bx.UNMEASURED


# --------------------------------------------------------------------- #
# Clipping policy
# --------------------------------------------------------------------- #
def test_clip_leaves_an_interior_box_untouched():
    res = bx.clip_box_to_tile(bx.PixelBox(10.0, 10.0, 20.0, 20.0), 1024, 1024)
    assert not res.clipped and res.on_tile
    assert res.visible_fraction == pytest.approx(1.0)
    assert res.box is res.full_box


def test_clip_halves_a_box_centred_on_the_edge():
    res = bx.clip_box_to_tile(bx.PixelBox(-50.0, 100.0, 50.0, 200.0), 1024, 1024)
    assert res.clipped and res.on_tile
    assert res.visible_fraction == pytest.approx(0.5)
    assert res.box.as_tuple() == pytest.approx((0.0, 100.0, 50.0, 200.0))
    assert res.full_box.x_min == -50.0          # the true extent is preserved


def test_clip_of_a_corner_box_multiplies_the_fractions():
    res = bx.clip_box_to_tile(bx.PixelBox(-25.0, -75.0, 75.0, 25.0), 1024, 1024)
    assert res.visible_fraction == pytest.approx(0.75 * 0.25)


def test_clip_reports_an_off_tile_box_rather_than_dropping_it():
    res = bx.clip_box_to_tile(bx.PixelBox(2000.0, 2000.0, 2100.0, 2100.0), 1024, 1024)
    assert res.box is None
    assert not res.on_tile
    assert res.visible_fraction == 0.0
    assert res.full_box.x_min == 2000.0          # still reported, not discarded


def test_clip_keeps_a_subpixel_box_on_the_tile():
    """A degenerate (zero-area) box has no area ratio; it must not vanish."""
    res = bx.clip_box_to_tile(bx.PixelBox(5.0, 5.0, 5.0, 5.0), 1024, 1024)
    assert res.on_tile and res.visible_fraction == 1.0 and not res.clipped


def test_clipped_crater_at_a_tile_edge_is_flagged_and_not_measured():
    """Headline clipping case: a 300 m crater whose centre sits on the left
    edge.  Half the box is visible; the diameter is explicitly unmeasured."""
    f = frame_with_centre_at_pixel(ROI_LON, ROI_LAT, 0.0, 512.0, pixel_size_m=1.0)
    cb = bx.crater_to_pixel_box(ROI_LON, ROI_LAT, 300.0, f)
    assert cb.clipped and cb.on_tile
    assert cb.visible_fraction == pytest.approx(0.5, abs=1e-6)
    assert cb.box.x_min == 0.0
    assert cb.box.width == pytest.approx(0.5 * cb.full_box.width, rel=1e-6)

    d_px = bx.proxy_diameter_geometric_mean_px(cb.box)
    assert bx.pixel_diameter_to_ground_m(d_px, f.pixel_size_m, ROI_LAT,
                                        clipped=cb.clipped) is bx.UNMEASURED
    # And the naive (wrong) answer that policy forbids would be far too small.
    wrong = bx.pixel_diameter_to_ground_m(d_px, f.pixel_size_m, ROI_LAT)
    assert wrong.value_m < 0.8 * 300.0

    flags = bx.quality_flags(box=cb.box, frame=f, confidence=0.9,
                             ground_diameter=bx.UNMEASURED,
                             visible_fraction=cb.visible_fraction,
                             clipped=cb.clipped, on_tile=cb.on_tile)
    assert flags.clipped and flags.edge_flag and flags.unmeasured
    assert not flags.low_confidence
    assert "clipped_by_tile_edge" in flags.reasons
    assert "diameter_is_lower_bound_only" in flags.reasons


def test_crater_entirely_off_the_tile_is_reported_not_dropped():
    f = frame_with_centre_at_pixel(ROI_LON, ROI_LAT, -5000.0, 512.0)
    cb = bx.crater_to_pixel_box(ROI_LON, ROI_LAT, 300.0, f)
    assert cb.box is None and not cb.on_tile
    assert cb.visible_fraction == 0.0
    assert cb.diameter_m == 300.0                     # the record survives
    flags = bx.quality_flags(frame=f, confidence=0.9, on_tile=False,
                             clipped=cb.clipped, visible_fraction=0.0)
    assert flags.unmeasured and flags.edge_flag
    assert "off_tile" in flags.reasons


def test_clip_is_idempotent():
    f = frame_with_centre_at_pixel(ROI_LON, ROI_LAT, 0.0, 512.0)
    cb = bx.crater_to_pixel_box(ROI_LON, ROI_LAT, 300.0, f)
    again = bx.clip_box_to_tile(cb.box, f.width_px, f.height_px)
    assert not again.clipped
    assert again.box.as_tuple() == pytest.approx(cb.box.as_tuple())


# --------------------------------------------------------------------- #
# Proxy conventions and their bias
# --------------------------------------------------------------------- #
def test_proxies_are_four_separate_functions_with_distinct_conventions():
    box = bx.PixelBox(0.0, 0.0, 200.0, 50.0)   # aspect ratio e = 4
    assert bx.proxy_diameter_width_px(box) == 200.0
    assert bx.proxy_diameter_height_px(box) == 50.0
    assert bx.proxy_diameter_arithmetic_mean_px(box) == 125.0
    assert bx.proxy_diameter_geometric_mean_px(box) == pytest.approx(100.0)
    assert set(bx.PIXEL_DIAMETER_PROXIES) == {
        "width", "height", "arithmetic_mean", "geometric_mean"}


@pytest.mark.parametrize("e", [1.0, 1.1, 1.5, 2.0, 4.0, 9.0])
def test_arithmetic_mean_exceeds_geometric_mean_by_the_closed_form(e):
    """AM/GM = (1 + e) / (2 sqrt(e)) for aspect ratio e -- the documented bias
    of the arithmetic-mean convention against the equal-area diameter."""
    box = bx.PixelBox(0.0, 0.0, 100.0 * e, 100.0)
    am = bx.proxy_diameter_arithmetic_mean_px(box)
    gm = bx.proxy_diameter_geometric_mean_px(box)
    assert am / gm == pytest.approx((1.0 + e) / (2.0 * np.sqrt(e)), rel=1e-12)
    assert am >= gm
    # GM is exactly the equal-area equivalent diameter of the box.
    assert gm ** 2 == pytest.approx(box.area, rel=1e-12)


def test_geometric_mean_is_the_equal_area_diameter_of_an_ellipse():
    """For an ellipse with semi-axes a, b the equal-area circle has diameter
    2*sqrt(a*b) -- exactly the geometric mean of the box extents.  The
    arithmetic mean overestimates it by 4.2% at a 1.5:1 axis ratio."""
    a, b = 150.0, 100.0
    box = bx.PixelBox(0.0, 0.0, 2 * a, 2 * b)
    equal_area_d = 2.0 * np.sqrt(a * b)
    assert bx.proxy_diameter_geometric_mean_px(box) == pytest.approx(equal_area_d, rel=1e-12)
    am = bx.proxy_diameter_arithmetic_mean_px(box)
    assert am / equal_area_d == pytest.approx(1.0206, rel=1e-3)   # 1.5:1 axes
    assert am > equal_area_d


@pytest.mark.parametrize("proxy", sorted(bx.PIXEL_DIAMETER_PROXIES))
@pytest.mark.parametrize("diameter_m", [20.0, 300.0, 1000.0])
def test_proxy_bias_against_a_known_circle_is_within_the_discretisation_bound(
        proxy, diameter_m):
    """Validate, do not assume: a known circular crater must come back to
    within the rim-sampling bound, and the error must be a (tiny) shortfall,
    because an envelope of samples can never exceed the true extent."""
    f = frame_centred_on(ROI_LON, ROI_LAT, pixel_size_m=0.5, width_px=4096,
                         height_px=4096)
    bias = bx.estimate_proxy_bias(proxy, ROI_LON, ROI_LAT, diameter_m, f)
    assert bias.true_diameter_m == diameter_m
    assert bias.n_rim_samples == 360
    assert bias.discretisation_bound == pytest.approx(
        1.0 - np.cos(np.pi / 360.0), rel=1e-12)
    assert -bias.discretisation_bound <= bias.relative_bias <= 1e-9
    assert bias.ratio == pytest.approx(1.0 + bias.relative_bias, rel=1e-12)


def test_mean_and_geometric_mean_agree_for_a_circle_but_not_for_an_ellipse():
    """On a known circle the two means are indistinguishable (the projected
    rim of a small circle is a circle, so the box is square).  Their
    difference is therefore entirely a shape effect, not a scale effect."""
    f = frame_centred_on(ROI_LON, ROI_LAT, pixel_size_m=0.5, width_px=4096,
                         height_px=4096)
    am = bx.estimate_proxy_bias("arithmetic_mean", ROI_LON, ROI_LAT, 300.0, f)
    gm = bx.estimate_proxy_bias("geometric_mean", ROI_LON, ROI_LAT, 300.0, f)
    assert am.estimated_diameter_m == pytest.approx(gm.estimated_diameter_m, rel=1e-9)
    assert abs(am.relative_bias) < 1e-5 and abs(gm.relative_bias) < 1e-5


def test_coarse_rim_sampling_has_a_larger_but_still_bounded_bias():
    f = frame_centred_on(ROI_LON, ROI_LAT, pixel_size_m=0.5, width_px=4096,
                         height_px=4096)
    coarse = bx.estimate_proxy_bias("geometric_mean", ROI_LON, ROI_LAT, 300.0, f,
                                    n_samples=7)
    assert coarse.discretisation_bound == pytest.approx(bx.rim_discretisation_bias(7))
    assert -coarse.discretisation_bound <= coarse.relative_bias <= 1e-9
    assert abs(coarse.relative_bias) > 1e-3     # genuinely coarser than n=360


def test_proxy_bias_refuses_a_clipped_crater():
    f = frame_with_centre_at_pixel(ROI_LON, ROI_LAT, 0.0, 512.0)
    with pytest.raises(ValueError, match="clipped"):
        bx.estimate_proxy_bias("geometric_mean", ROI_LON, ROI_LAT, 300.0, f)


def test_unknown_proxy_name_is_rejected():
    f = frame_centred_on(ROI_LON, ROI_LAT)
    with pytest.raises(ValueError, match="unknown diameter proxy"):
        bx.estimate_proxy_bias("mean_radius", ROI_LON, ROI_LAT, 100.0, f)


def test_rim_discretisation_bias_values_and_guard():
    assert bx.rim_discretisation_bias(72) == pytest.approx(9.5178e-4, rel=1e-3)
    assert bx.rim_discretisation_bias(360) == pytest.approx(3.8077e-5, rel=1e-3)
    assert bx.rim_discretisation_bias(720) < bx.rim_discretisation_bias(360)
    with pytest.raises(ValueError):
        bx.rim_discretisation_bias(2)


# --------------------------------------------------------------------- #
# Quality flags
# --------------------------------------------------------------------- #
def test_quality_flags_clean_interior_detection():
    f = frame_centred_on(ROI_LON, ROI_LAT)
    cb = bx.crater_to_pixel_box(ROI_LON, ROI_LAT, 300.0, f)
    gd = bx.pixel_diameter_to_ground_m(
        bx.proxy_diameter_geometric_mean_px(cb.box), f.pixel_size_m, ROI_LAT)
    flags = bx.quality_flags(box=cb.box, frame=f, confidence=0.88,
                             ground_diameter=gd,
                             visible_fraction=cb.visible_fraction,
                             clipped=cb.clipped, on_tile=cb.on_tile)
    assert not (flags.edge_flag or flags.clipped or flags.low_confidence
                or flags.unmeasured)
    assert flags.reasons == ()
    assert flags.visible_fraction == pytest.approx(1.0)


def test_quality_flags_low_confidence_and_missing_confidence():
    f = frame_centred_on(ROI_LON, ROI_LAT)
    box = bx.PixelBox(400.0, 400.0, 600.0, 600.0)
    gd = bx.pixel_diameter_to_ground_m(200.0, 1.0, ROI_LAT)
    low = bx.quality_flags(box=box, frame=f, confidence=0.1, ground_diameter=gd)
    assert low.low_confidence and not low.unmeasured
    assert "confidence_below_threshold" in low.reasons
    missing = bx.quality_flags(box=box, frame=f, ground_diameter=gd)
    assert missing.low_confidence and "confidence_missing" in missing.reasons
    nan_conf = bx.quality_flags(box=box, frame=f, confidence=float("nan"),
                               ground_diameter=gd)
    assert nan_conf.low_confidence and "confidence_non_finite" in nan_conf.reasons


def test_quality_flags_edge_proximity_without_clipping():
    """A box flush against the tile boundary is not clipped but is an edge
    object, so dedup across the tile overlap must see it."""
    f = frame_centred_on(ROI_LON, ROI_LAT)
    box = bx.PixelBox(0.0, 400.0, 100.0, 500.0)
    flags = bx.quality_flags(box=box, frame=f, confidence=0.9,
                             ground_diameter=bx.pixel_diameter_to_ground_m(
                                 100.0, 1.0, ROI_LAT))
    assert flags.edge_flag and not flags.clipped and not flags.unmeasured
    assert "touches_tile_edge" in flags.reasons


def test_quality_flags_derives_clipped_from_box_and_frame():
    f = frame_centred_on(ROI_LON, ROI_LAT)
    flags = bx.quality_flags(box=bx.PixelBox(-10.0, 10.0, 10.0, 30.0), frame=f,
                             confidence=0.9)
    assert flags.clipped and flags.edge_flag and flags.unmeasured


def test_quality_flags_unmeasured_states():
    f = frame_centred_on(ROI_LON, ROI_LAT)
    box = bx.PixelBox(400.0, 400.0, 600.0, 600.0)
    assert bx.quality_flags(box=box, frame=f, confidence=0.9,
                            ground_diameter=None).unmeasured
    assert bx.quality_flags(box=box, frame=f, confidence=0.9,
                            ground_diameter=bx.UNMEASURED).unmeasured
    degenerate = bx.quality_flags(box=bx.PixelBox(5.0, 5.0, 5.0, 5.0), frame=f,
                                  confidence=0.9, ground_diameter=10.0)
    assert degenerate.unmeasured and "degenerate_box" in degenerate.reasons


def test_quality_flags_notes_out_of_range_diameter_without_calling_it_unmeasured():
    """D-003 approves 20-1000 m.  A 5 m detection is measured but out of scope;
    it must be flagged, not deleted and not called unmeasured."""
    f = frame_centred_on(ROI_LON, ROI_LAT)
    box = bx.PixelBox(400.0, 400.0, 405.0, 405.0)
    small = bx.quality_flags(box=box, frame=f, confidence=0.9,
                             ground_diameter=bx.pixel_diameter_to_ground_m(
                                 5.0, 1.0, ROI_LAT))
    assert not small.unmeasured
    assert "diameter_outside_approved_range" in small.reasons
    big = bx.quality_flags(box=box, frame=f, confidence=0.9,
                           ground_diameter=bx.pixel_diameter_to_ground_m(
                               5000.0, 1.0, ROI_LAT))
    assert "diameter_outside_approved_range" in big.reasons
    assert bx.APPROVED_DIAMETER_RANGE_M == (20.0, 1000.0)


def test_quality_flags_partial_visibility_sets_edge_flag():
    flags = bx.quality_flags(confidence=0.9, visible_fraction=0.8,
                             ground_diameter=10.0)
    assert flags.edge_flag and "partially_visible" in flags.reasons
