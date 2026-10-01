"""Tests for mosaic tiling, pixel/world mapping and multiscale recovery.

Every expectation here is derived analytically from the grid parameters rather
than from a previous run of the code, so the tests are a specification and not a
snapshot.  The synthetic mosaic is placed at a realistic ROI latitude (~85.9 S)
so that the projection scale factor is genuinely != 1 and cannot be ignored.
"""
import math
import warnings

import numpy as np
import pytest
from affine import Affine

from crater import geometry as g
from crater import tiling as tl
from crater.body import MOON

# A 1 m/px north-up mosaic centred near 85.9 S, 0 E.  The ROI centre is a test
# fixture only; nothing in the module hard-codes one (DECISIONS.md D-002).
ROI_LON, ROI_LAT = 0.0, -85.9
PIXEL_M = 1.0


def _mosaic_transform(width=4096, height=4096, pixel_m=PIXEL_M):
    cx, cy = g.forward(ROI_LON, ROI_LAT)
    return Affine(pixel_m, 0.0, cx - width * pixel_m / 2.0,
                  0.0, -pixel_m, cy + height * pixel_m / 2.0)


@pytest.fixture
def transform():
    return _mosaic_transform()


# --------------------------------------------------------------------- #
# TileGrid validation
# --------------------------------------------------------------------- #
def test_grid_derives_pixel_scale_from_transform(transform):
    grid = tl.TileGrid(tile_px=512, margin_px=64, transform=transform)
    assert grid.pixel_scale_m == pytest.approx(PIXEL_M, rel=1e-12)
    assert grid.stride_px == 512 - 2 * 64
    assert grid.overlap_px == 128
    assert grid.scale_label == "L0"
    assert grid.tile_span_projected_m == pytest.approx(512.0)


def test_grid_rejects_pixel_scale_that_contradicts_the_transform(transform):
    with pytest.raises(ValueError, match="disagrees"):
        tl.TileGrid(tile_px=512, margin_px=64, transform=transform,
                    pixel_scale_m=2.0)


def test_grid_rejects_margin_that_leaves_no_stride(transform):
    with pytest.raises(ValueError, match="no stride"):
        tl.TileGrid(tile_px=128, margin_px=64, transform=transform)


def test_grid_rejects_non_square_pixels(transform):
    skewed = Affine(1.0, 0.0, transform.c, 0.0, -2.0, transform.f)
    with pytest.raises(ValueError, match="non-square"):
        tl.TileGrid(tile_px=256, margin_px=16, transform=skewed)


def test_grid_rejects_unknown_edge_policy(transform):
    with pytest.raises(ValueError, match="edge_policy"):
        tl.TileGrid(tile_px=256, margin_px=16, transform=transform,
                    edge_policy="drop")


def test_grid_rejects_singular_transform():
    with pytest.raises(ValueError, match="singular"):
        tl.TileGrid(tile_px=256, margin_px=16,
                    transform=Affine(0.0, 0.0, 0.0, 0.0, 0.0, 0.0))


# --------------------------------------------------------------------- #
# Enumeration, edges and core ownership
# --------------------------------------------------------------------- #
def test_tile_count_is_the_analytic_one(transform):
    """Uniform stride: offsets are 0, s, 2s, ... while < size."""
    grid = tl.TileGrid(tile_px=512, margin_px=64, transform=transform)
    width, height = 2048, 1536
    s = grid.stride_px  # 384
    expected = math.ceil(width / s) * math.ceil(height / s)
    tiles = grid.tiles(width, height)
    assert len(tiles) == expected == 6 * 4
    assert len({t.tile_id for t in tiles}) == len(tiles)


def test_partial_edge_tile_is_emitted_with_a_flag_and_padding(transform):
    """2048 is not a multiple of the 384 stride: the last tile is clipped."""
    grid = tl.TileGrid(tile_px=512, margin_px=64, transform=transform)
    tiles = grid.tiles(2048, 2048)
    last = tiles[-1]
    assert last.col_off == 1920 and last.row_off == 1920
    assert last.width == 2048 - 1920 == 128
    assert last.partial is True
    assert last.pad_cols == 512 - 128 and last.pad_rows == 512 - 128
    # Interior tiles are whole.
    interior = [t for t in tiles if not t.partial]
    assert interior and all(t.width == t.height == 512 for t in interior)
    # Nothing was dropped: every offset on the lattice is present.
    assert sorted({t.col_off for t in tiles}) == list(range(0, 2048, 384))


def test_raster_smaller_than_one_tile_yields_one_partial_tile(transform):
    grid = tl.TileGrid(tile_px=512, margin_px=64, transform=transform)
    tiles = grid.tiles(100, 60)
    assert len(tiles) == 1
    t0 = tiles[0]
    assert (t0.width, t0.height) == (100, 60)
    assert t0.partial and (t0.pad_cols, t0.pad_rows) == (412, 452)
    assert (t0.core_width, t0.core_height) == (100, 60)


def test_shift_policy_keeps_edge_tiles_full_size(transform):
    grid = tl.TileGrid(tile_px=512, margin_px=64, transform=transform,
                       edge_policy="shift")
    tiles = grid.tiles(2048, 2048)
    assert all(t.width == t.height == 512 for t in tiles)
    assert not any(t.partial for t in tiles)
    # The last offset is clamped so the window ends exactly at the raster edge.
    assert max(t.col_off for t in tiles) == 2048 - 512


@pytest.mark.parametrize("policy", list(tl.EDGE_POLICIES))
@pytest.mark.parametrize("size", [(100, 100), (128, 60), (200, 137), (33, 512)])
def test_core_windows_tile_the_raster_exactly(transform, policy, size):
    """The exclusive cores must be disjoint and cover every mosaic pixel.

    This is the property that makes cross-tile duplicate handling exact: each
    detection belongs to exactly one tile, decided by its centre.
    """
    width, height = size
    grid = tl.TileGrid(tile_px=32, margin_px=4, transform=transform,
                       edge_policy=policy)
    hits = np.zeros((height, width), dtype=np.int32)
    for t in grid.iter_tiles(width, height):
        hits[t.core_row_off:t.core_row_stop, t.core_col_off:t.core_col_stop] += 1
        # The core must lie inside the tile's own read window.
        assert t.col_off <= t.core_col_off < t.core_col_stop <= t.col_stop
        assert t.row_off <= t.core_row_off < t.core_row_stop <= t.row_stop
    assert hits.min() == 1 and hits.max() == 1


def test_tile_transform_matches_the_mosaic_transform_at_its_offset(transform):
    grid = tl.TileGrid(tile_px=256, margin_px=32, transform=transform)
    for t in grid.tiles(900, 700):
        assert tl.pixel_to_world(t.transform, 0.0, 0.0) == pytest.approx(
            tl.pixel_to_world(transform, t.col_off, t.row_off), abs=1e-9)
        # A point in the middle of the tile maps to the same world point either way.
        assert tl.pixel_to_world(t.transform, 7.25, 3.5) == pytest.approx(
            tl.pixel_to_world(transform, t.col_off + 7.25, t.row_off + 3.5),
            abs=1e-9)


# --------------------------------------------------------------------- #
# Pixel <-> projected metres <-> lon/lat
# --------------------------------------------------------------------- #
def test_pixel_world_roundtrip_scalar_and_array(transform):
    rng = np.random.default_rng(20261001)
    cols = rng.uniform(0, 4096, 500)
    rows = rng.uniform(0, 4096, 500)
    x, y = tl.pixel_to_world(transform, cols, rows)
    c2, r2 = tl.world_to_pixel(transform, x, y)
    assert np.max(np.abs(c2 - cols)) < 1e-9
    assert np.max(np.abs(r2 - rows)) < 1e-9
    # Scalars come back as floats, not 0-d arrays.
    xs, ys = tl.pixel_to_world(transform, 3.0, 4.0)
    assert isinstance(xs, float) and isinstance(ys, float)


def test_pixel_centre_convention_is_half_a_pixel(transform):
    """Integer coordinates are pixel EDGES; the centre of pixel i is i + 0.5."""
    edge = tl.pixel_to_world(transform, 10, 20)
    centre = tl.pixel_to_world(transform, 10, 20, centre=True)
    assert centre[0] - edge[0] == pytest.approx(0.5 * PIXEL_M)
    assert centre[1] - edge[1] == pytest.approx(-0.5 * PIXEL_M)  # north-up: e < 0
    back = tl.world_to_pixel(transform, *centre, centre=True)
    assert back == pytest.approx((10.0, 20.0), abs=1e-9)
    assert "EDGES" in tl.PIXEL_COORDINATE_CONVENTION


def test_pixel_lonlat_roundtrip_and_agreement_with_geometry(transform):
    cols = np.array([0.0, 1.5, 2047.25, 4095.0])
    rows = np.array([0.0, 900.75, 2048.0, 4095.0])
    lon, lat = tl.pixel_to_lonlat(transform, cols, rows)
    # Agrees with doing it by hand through crater.geometry.
    x, y = tl.pixel_to_world(transform, cols, rows)
    lon_ref, lat_ref = g.inverse(x, y)
    assert np.allclose(lon, lon_ref) and np.allclose(lat, lat_ref)
    # Round trips back to the same pixels.
    c2, r2 = tl.lonlat_to_pixel(transform, lon, lat)
    assert np.max(np.abs(c2 - cols)) < 1e-5
    assert np.max(np.abs(r2 - rows)) < 1e-5
    # And the mosaic really is near the pole.
    assert np.all(lat < -85.0)


def test_tile_footprint_lonlat_is_a_closed_ring_through_the_corners(transform):
    grid = tl.TileGrid(tile_px=256, margin_px=16, transform=transform)
    t0 = grid.tiles(1024, 1024)[5]
    lon, lat = tl.tile_footprint_lonlat(t0, n_per_edge=5)
    assert len(lon) == 4 * 5 + 1
    assert lon[0] == pytest.approx(lon[-1]) and lat[0] == pytest.approx(lat[-1])
    corners = tl.tile_corners_lonlat(t0)
    assert (lon[0], lat[0]) == pytest.approx(corners[0], abs=1e-12)
    assert (lon[5], lat[5]) == pytest.approx(corners[1], abs=1e-12)
    # A projected rectangle is not a lon/lat rectangle: the edge midpoint does
    # not sit on the straight lon/lat line between the corners.
    xmin, ymin, xmax, ymax = tl.tile_bounds_projected(t0)
    assert xmax - xmin == pytest.approx(256 * PIXEL_M)
    assert ymax - ymin == pytest.approx(256 * PIXEL_M)


# --------------------------------------------------------------------- #
# Projected vs ground lengths (INTERFACES.md point 2)
# --------------------------------------------------------------------- #
def test_projected_metres_are_not_ground_metres_at_the_roi():
    k = g.point_scale_factor(ROI_LAT)
    assert 1.0009 < k < 1.0015
    assert tl.projected_to_ground_length(1000.0, ROI_LAT) == pytest.approx(
        1000.0 / k, rel=1e-15)
    assert tl.ground_to_projected_length(1000.0, ROI_LAT) == pytest.approx(
        1000.0 * k, rel=1e-15)
    # Round trip, and the direction of the inequality.
    assert tl.projected_to_ground_length(1000.0, ROI_LAT) < 1000.0
    assert tl.ground_to_projected_length(
        tl.projected_to_ground_length(1234.5, ROI_LAT), ROI_LAT) == pytest.approx(
        1234.5, rel=1e-15)
    # A nominal 1 m/px mosaic samples slightly less than a ground metre.
    gs = tl.ground_pixel_scale_m(1.0, ROI_LAT)
    assert gs < 1.0 and gs == pytest.approx(1.0 / k, rel=1e-15)
    # The error from ignoring k is ~0.12% -- 1.2 m on a 1 km crater.
    assert 1000.0 * k - 1000.0 == pytest.approx(1.28, abs=0.1)


def test_tile_ground_span_is_shorter_than_its_projected_span(transform):
    grid = tl.TileGrid(tile_px=1024, margin_px=64, transform=transform)
    assert grid.tile_span_projected_m == pytest.approx(1024.0)
    assert grid.tile_span_ground_m(ROI_LAT) < 1024.0
    assert grid.tile_span_ground_m(ROI_LAT) == pytest.approx(
        1024.0 / g.point_scale_factor(ROI_LAT), rel=1e-15)


# --------------------------------------------------------------------- #
# Multiscale pyramid arithmetic
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("level,factor", [(0, 1), (1, 2), (2, 4), (3, 8)])
def test_level_scale_factor_and_raster_size(level, factor):
    assert tl.level_scale_factor(level) == factor
    # Ceiling division: a partially filled coarse pixel still holds data.
    assert tl.level_raster_size(1000, 999, level) == (
        -(-1000 // factor), -(-999 // factor))


def test_level_scale_factor_rejects_negative_and_non_integer():
    with pytest.raises(ValueError):
        tl.level_scale_factor(-1)
    with pytest.raises(TypeError):
        tl.level_scale_factor(1.5)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_downsample_transform_coarsens_the_pixel_and_keeps_the_origin(transform, level):
    lv = tl.downsample_transform(transform, level)
    f = tl.level_scale_factor(level)
    assert lv.a == pytest.approx(transform.a * f, rel=1e-15)
    assert lv.e == pytest.approx(transform.e * f, rel=1e-15)
    assert (lv.c, lv.f) == (transform.c, transform.f)
    # One level-L pixel spans f level-0 pixels of the same ground.
    assert tl.pixel_to_world(lv, 1.0, 0.0) == pytest.approx(
        tl.pixel_to_world(transform, float(f), 0.0), abs=1e-9)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_level_pixel_mapping_is_exact_and_invertible(level):
    cols = np.array([0.0, 0.5, 1.0, 17.25, 511.5])
    rows = np.array([0.0, 3.5, 64.0, 100.125, 255.75])
    c0, r0 = tl.level_pixel_to_full_pixel(cols, rows, level)
    assert np.array_equal(c0, cols * tl.level_scale_factor(level))
    cb, rb = tl.full_pixel_to_level_pixel(c0, r0, level)
    # Powers of two: the round trip is exact, not approximate.
    assert np.array_equal(cb, cols) and np.array_equal(rb, rows)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_multiscale_pixel_recovery_roundtrips_exactly(level):
    off_c, off_r = 37, 59          # tile offset in LEVEL-L pixels
    p_c, p_r = 128.5, 64.25        # detection centre inside the tile, level-L px
    c_full, r_full = tl.recover_mosaic_pixel(p_c, p_r, off_c, off_r, level)
    f = tl.level_scale_factor(level)
    assert c_full == (off_c + p_c) * f
    assert r_full == (off_r + p_r) * f
    back = tl.mosaic_pixel_to_tile_pixel(c_full, r_full, off_c, off_r, level)
    assert back == (p_c, p_r)      # exact equality, not approx


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_offset_must_be_added_before_scaling(level):
    """Regression guard for the classic multiscale bug.

    Scaling the in-tile coordinate and *then* adding the tile offset is wrong by
    ``offset * (2**L - 1)`` pixels -- at level 3 with a 512 px tile that is
    thousands of pixels, i.e. kilometres on the ground.
    """
    off_c, off_r, p_c, p_r = 512, 1024, 10.5, 20.5
    f = tl.level_scale_factor(level)
    c_ok, r_ok = tl.recover_mosaic_pixel(p_c, p_r, off_c, off_r, level)
    c_bug, r_bug = off_c + p_c * f, off_r + p_r * f
    assert c_ok - c_bug == off_c * (f - 1)
    assert r_ok - r_bug == off_r * (f - 1)
    if level == 0:
        assert (c_ok, r_ok) == (c_bug, r_bug)   # no difference at full res
    else:
        assert c_ok != c_bug


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_multiscale_lonlat_recovery_roundtrips(transform, level):
    off_c, off_r = 11, 23
    p_c, p_r = 65.5, 200.25
    lon, lat = tl.recover_lonlat(p_c, p_r, off_c, off_r, level, transform)
    assert lat < -85.0 and -180.0 <= lon < 180.0
    # Same answer as going through the level-L transform directly.
    lv = tl.downsample_transform(transform, level)
    lon2, lat2 = tl.pixel_to_lonlat(lv, off_c + p_c, off_r + p_r)
    assert lon == pytest.approx(lon2, abs=1e-12)
    assert lat == pytest.approx(lat2, abs=1e-12)
    # Round trip back to in-tile level-L pixels.
    c_back, r_back = tl.lonlat_to_tile_pixel(lon, lat, off_c, off_r, level, transform)
    assert c_back == pytest.approx(p_c, abs=1e-6)
    assert r_back == pytest.approx(p_r, abs=1e-6)


@pytest.mark.parametrize("level", [0, 1, 2, 3])
def test_detections_in_real_level_tiles_recover_to_the_same_ground_point(
        transform, level):
    """End-to-end: tile a pyramid level, then recover a detection's lon/lat.

    The same ground point is detected in the level-L tile that contains it, and
    must come back to the same lon/lat at every level.
    """
    base = tl.TileGrid(tile_px=128, margin_px=16, transform=transform)
    width, height = 1024, 1024
    grid = base.at_level(level)
    lw, lh = tl.level_raster_size(width, height, level)
    tiles = grid.tiles(lw, lh)
    assert tiles and all(t.scale_label == f"L{level}" for t in tiles)
    # Target the ground point at full-resolution pixel (600.5, 400.5).
    target_full = (600.5, 400.5)
    lon_ref, lat_ref = tl.pixel_to_lonlat(transform, *target_full)
    c_l, r_l = tl.full_pixel_to_level_pixel(*target_full, level)
    owner = [t for t in tiles
             if t.contains_core(c_l - t.col_off, r_l - t.row_off)]
    assert len(owner) == 1, "exactly one tile core must own the point"
    t0 = owner[0]
    lon, lat = tl.recover_lonlat(c_l - t0.col_off, r_l - t0.row_off,
                                 t0.col_off, t0.row_off, level, transform)
    assert lon == pytest.approx(lon_ref, abs=1e-12)
    assert lat == pytest.approx(lat_ref, abs=1e-12)


def test_tile_ids_are_unique_across_pyramid_levels(transform):
    base = tl.TileGrid(tile_px=128, margin_px=16, transform=transform)
    ids = []
    for level in (0, 1, 2, 3):
        lw, lh = tl.level_raster_size(1024, 1024, level)
        ids += [t.tile_id for t in base.at_level(level).tiles(lw, lh)]
    assert len(ids) == len(set(ids))


def test_pyramid_levels_only_coarsen(transform):
    grid = tl.TileGrid(tile_px=128, margin_px=16, transform=transform, level=2,
                       scale_label="L2")
    with pytest.raises(ValueError, match="only coarsen"):
        grid.at_level(1)


# --------------------------------------------------------------------- #
# Tile size vs the approved 20 m - 1000 m diameter range (D-003)
# --------------------------------------------------------------------- #
def test_one_metre_pixel_1024_tile_cannot_hold_a_1000_m_crater():
    """The real constraint the project must surface, not hide.

    At 1 m/px a 1024 px tile spans 1024 projected m (~1022.7 ground m).  The
    1000 m crater alone is 1001 px once the scale factor is applied, so there is
    no room for the context a rim fit needs.
    """
    with pytest.warns(tl.TileSizeWarning, match="CANNOT hold"):
        rec = tl.recommend_tile_size(1.0, ROI_LAT, proposed_tile_px=1024)
    k = g.point_scale_factor(ROI_LAT)
    assert rec.max_diameter_px == pytest.approx(1000.0 * k / 1.0, rel=1e-15)
    assert rec.max_diameter_px > 1000.0            # k > 1 inflates it
    assert rec.proposed_tile_span_projected_m == pytest.approx(1024.0)
    assert rec.proposed_tile_span_ground_m < 1024.0
    assert rec.fits is False and rec.ok is False
    # Required = core (one diameter) + 2 * margin (half a diameter + context).
    d_px = 1000.0 * k
    assert rec.required_core_px == math.ceil(d_px)
    assert rec.required_margin_px == math.ceil(d_px * (0.5 + 0.25))
    assert rec.required_tile_px == rec.required_core_px + 2 * rec.required_margin_px
    assert rec.shortfall_px == rec.required_tile_px - 1024
    assert rec.recommended_tile_px == 4096        # next power of two >= 2504
    assert rec.recommended_tile_px >= rec.required_tile_px
    # Nothing was resized on our behalf.
    assert rec.proposed_tile_px == 1024
    assert rec.largest_diameter_that_fits_m < 1000.0


def test_recommendation_reports_the_scale_level_that_would_work_and_its_cost():
    """A coarser pyramid level fits the big craters but loses the small ones.

    This is the honest finding: one tile size at one scale cannot serve the whole
    approved 20 m - 1000 m range at 1 m/px.
    """
    with pytest.warns(tl.TileSizeWarning):
        rec = tl.recommend_tile_size(1.0, ROI_LAT, proposed_tile_px=1024)
    assert rec.min_level_that_fits == 2
    assert rec.pixel_scale_at_min_level_m == pytest.approx(4.0)
    assert rec.min_diameter_px_at_min_level == pytest.approx(
        20.0 * g.point_scale_factor(ROI_LAT) / 4.0, rel=1e-12)
    assert rec.min_diameter_px_at_min_level < rec.min_pixels_across
    assert rec.smallest_crater_resolved is True            # at level 0, 20 px
    assert rec.smallest_crater_resolved_at_min_level is False
    assert any("two-scale" in w for w in rec.warnings)


def test_a_large_enough_tile_passes_with_no_warnings():
    with warnings.catch_warnings():
        warnings.simplefilter("error")      # any warning fails the test
        rec = tl.recommend_tile_size(1.0, ROI_LAT, proposed_tile_px=4096)
    assert rec.fits is True and rec.ok is True
    assert rec.warnings == ()
    assert rec.min_level_that_fits == 0
    assert rec.largest_diameter_that_fits_m > 1000.0


def test_recommendation_can_be_asked_for_quietly_but_still_reports():
    rec = tl.recommend_tile_size(1.0, ROI_LAT, proposed_tile_px=1024,
                                 emit_warning=False)
    assert rec.fits is False
    assert rec.warnings  # the structured finding is still there


def test_coarse_mosaic_cannot_resolve_the_smallest_craters():
    """Kaguya TC is 7.4 m/px: a 20 m crater is ~3 px and cannot be measured."""
    with pytest.warns(tl.TileSizeWarning, match="below the 8 px floor"):
        rec = tl.recommend_tile_size(7.4, ROI_LAT, proposed_tile_px=1024)
    assert rec.min_diameter_px < 8.0
    assert rec.smallest_crater_resolved is False
    assert rec.fits is True          # the big end fits easily at this scale
    assert rec.ok is False           # but the recommendation is still not OK


def test_pixel_scale_may_be_given_as_a_ground_sampling():
    k = g.point_scale_factor(ROI_LAT)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        proj = tl.recommend_tile_size(1.0, ROI_LAT, proposed_tile_px=1024)
        ground = tl.recommend_tile_size(1.0, ROI_LAT, proposed_tile_px=1024,
                                        pixel_scale_is_projected=False)
    assert proj.pixel_scale_ground_m == pytest.approx(1.0 / k, rel=1e-15)
    assert ground.pixel_scale_projected_m == pytest.approx(1.0 * k, rel=1e-15)
    # Reading a projected scale as a ground scale makes craters look smaller.
    assert ground.max_diameter_px < proj.max_diameter_px


def test_recommendation_rejects_impossible_inputs():
    with pytest.raises(ValueError):
        tl.recommend_tile_size(0.0, ROI_LAT)
    with pytest.raises(ValueError):
        tl.recommend_tile_size(1.0, ROI_LAT, min_diameter_m=2000.0,
                               max_diameter_m=1000.0)
    with pytest.raises(ValueError):
        tl.recommend_tile_size(1.0, ROI_LAT, context_fraction=-0.1)


def test_grid_can_check_itself(transform):
    grid = tl.TileGrid(tile_px=1024, margin_px=64, transform=transform)
    with pytest.warns(tl.TileSizeWarning):
        rec = grid.recommend_tile_size(ROI_LAT)
    assert rec.proposed_tile_px == 1024 and rec.fits is False
