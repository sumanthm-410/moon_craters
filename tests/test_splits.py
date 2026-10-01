"""Tests for geographic splitting and leakage detection.

All geometry is synthetic and placed near the real ROI latitude (~85.9 S) so the
projection scale factor is genuinely != 1; every expected number below is derived
by hand from the box coordinates, not read off a previous run.

The headline cases are the ones a split must never get wrong: two adjacent tiles
of a real overlapping grid landing in different splits, the same crater appearing
twice, the same ground seen at two pyramid scales, an alternate acquisition of
the same ground, a buffer that is one pixel too small, and a 70/15/15 target that
the geometry cannot support.
"""
import inspect
import warnings

import pytest
from affine import Affine
from shapely.geometry import MultiPolygon, Polygon, box

from crater import geometry as g
from crater import splits as sp
from crater import tiling as tl

ROI_LON, ROI_LAT = 0.0, -85.9
#: Origin of the synthetic ROI in projected metres.
CX, CY = g.forward(ROI_LON, ROI_LAT)
#: Scale factor at the ROI; ~1.0012, i.e. 1 projected m is < 1 ground m.
K = g.point_scale_factor(ROI_LAT)


def fp(x0, y0, x1, y1):
    """A footprint box in projected metres, relative to the ROI origin."""
    return box(CX + x0, CY + y0, CX + x1, CY + y1)


def region(name, x0, y0, x1, y1):
    return sp.SplitRegion.from_bounds(name, CX + x0, CY + y0, CX + x1, CY + y1)


def stile(tid, geom, split, level=0, source_id="M1", crater_ids=()):
    return sp.SplitTile(tile_id=tid, footprint=geom, split=split, level=level,
                        source_id=source_id, crater_ids=tuple(crater_ids))


def grid_at(pixel_m=1.0, tile_px=100, margin_px=10, origin=(-500.0, 500.0)):
    """A real tile grid, with a context margin, anchored at the ROI."""
    tr = Affine(pixel_m, 0.0, CX + origin[0], 0.0, -pixel_m, CY + origin[1])
    return tl.TileGrid(tile_px=tile_px, margin_px=margin_px, transform=tr)


# --------------------------------------------------------------------- #
# A random tile split must be inexpressible
# --------------------------------------------------------------------- #
def test_scattered_tiles_cannot_become_a_split_region():
    """The union of two disjoint tiles is a MultiPolygon and is refused."""
    scattered = MultiPolygon([fp(0, 0, 100, 100), fp(500, 500, 600, 600)])
    with pytest.raises(ValueError, match="MultiPolygon"):
        sp.SplitRegion(name="train", geometry=scattered)


def test_corner_touching_blocks_are_not_contiguous():
    from shapely.ops import unary_union
    kissing = unary_union([fp(0, 0, 100, 100), fp(100, 100, 200, 200)])
    with pytest.raises(ValueError, match="MultiPolygon|not contiguous"):
        sp.SplitRegion(name="train", geometry=kissing)


def test_region_interior_must_stay_connected_under_erosion():
    """A hole that pinches a block in two is caught even though it is valid."""
    outer = fp(0, 0, 300, 100)
    neck = fp(100, 1e-9, 200, 100 - 1e-9)
    pinched = Polygon(outer.exterior.coords, [neck.exterior.coords])
    assert pinched.is_valid
    with pytest.raises(ValueError, match="not contiguous"):
        sp.SplitRegion(name="train", geometry=pinched, allow_holes=True)


def test_holes_are_refused_unless_explicitly_allowed():
    annulus = Polygon(fp(0, 0, 300, 300).exterior.coords,
                      [fp(100, 100, 200, 200).exterior.coords])
    with pytest.raises(ValueError, match="Holes are refused"):
        sp.SplitRegion(name="train", geometry=annulus)
    allowed = sp.SplitRegion(name="train", geometry=annulus, allow_holes=True)
    assert allowed.n_holes == 1          # recorded, so it can be reported


def test_degenerate_zero_width_region_is_refused():
    with pytest.raises(ValueError, match="degenerate"):
        sp.SplitRegion(name="train", geometry=box(0, 0, 1e9, 1e-6))


def test_self_intersecting_region_is_refused():
    bowtie = Polygon([(0, 0), (10, 10), (10, 0), (0, 10)])
    with pytest.raises(ValueError, match="not a valid polygon"):
        sp.SplitRegion(name="train", geometry=bowtie)


def test_module_contains_no_randomness():
    """There is no RNG, shuffle or seed anywhere in this module."""
    src = inspect.getsource(sp)
    assert "import random" not in src
    assert "shuffle" not in src
    assert "default_rng" not in src
    assert not hasattr(sp, "random")


def test_assignment_api_offers_no_override_hook():
    """Freeze the signature: nothing here lets a caller place a tile by hand."""
    params = set(inspect.signature(sp.assign_tiles_to_splits).parameters)
    assert params == {
        "tiles", "split_regions", "buffer_m", "buffer_kind",
        "source_ids", "crater_ids", "body", "lon_0", "k0",
    }


def test_regions_must_be_splitregion_instances():
    with pytest.raises(TypeError, match="contiguity"):
        sp.assign_tiles_to_splits([], [fp(0, 0, 10, 10)], buffer_m=0.0)


def test_overlapping_regions_are_refused():
    a = region("train", 0, 0, 100, 100)
    b = region("val", 50, 0, 150, 100)
    with pytest.raises(ValueError, match="overlap"):
        sp.assign_tiles_to_splits([], [a, b], buffer_m=0.0)


def test_adjacent_regions_are_allowed():
    a = region("train", 0, 0, 100, 100)
    b = region("val", 100, 0, 200, 100)
    out = sp.assign_tiles_to_splits([], [a, b], buffer_m=0.0)
    assert out.region_names == ("train", "val")


# --------------------------------------------------------------------- #
# Assignment: wholly inside, buffered, or discarded with a reason
# --------------------------------------------------------------------- #
class Fake:
    """Minimal tile-like object: an id plus a projected footprint."""

    def __init__(self, tile_id, footprint, level=0, source_id="M1", crater_ids=()):
        self.tile_id = tile_id
        self.footprint = footprint
        self.level = level
        self.source_id = source_id
        self.crater_ids = tuple(crater_ids)


#: Two blocks with a 500 m gap: train on x in [-1000, 0], val on x in [500, 1500].
TRAIN = ("train", -1000, -1000, 0, 1000)
VAL = ("val", 500, -1000, 1500, 1000)


def _two_regions():
    return [region(*TRAIN), region(*VAL)]


def test_tile_wholly_inside_and_clear_is_assigned():
    tile = Fake("t", fp(-300, -100, -200, 100))   # 700 m from val
    out = sp.assign_tiles_to_splits([tile], _two_regions(), buffer_m=100.0,
                                    buffer_kind="projected")
    a = out.assignment_of("t")
    assert a.split == "train" and a.reason == "" and not a.discarded
    assert a.nearest_other_split == "val"
    assert a.nearest_other_distance_projected_m == pytest.approx(700.0)
    assert out.splits == {"train": ("t",), "val": ()}
    assert out.discarded == ()


def test_tile_straddling_a_split_boundary_is_discarded_with_that_reason():
    """Two adjacent blocks, and a tile lying across the seam."""
    a = region("train", -500, -500, 0, 500)
    b = region("val", 0, -500, 500, 500)
    tile = Fake("seam", fp(-50, -50, 50, 50))
    out = sp.assign_tiles_to_splits([tile], [a, b], buffer_m=0.0)
    rec = out.assignment_of("seam")
    assert rec.split is None
    assert rec.reason == "straddles_region_boundary"
    assert "train" in rec.detail and "val" in rec.detail
    assert out.discarded_ids == ("seam",)
    assert out.reason_counts["straddles_region_boundary"] == 1


def test_tile_partly_outside_a_lone_region_is_discarded():
    tile = Fake("edge", fp(-1100, -100, -900, 100))   # hangs off train's left edge
    out = sp.assign_tiles_to_splits([tile], _two_regions(), buffer_m=0.0)
    assert out.assignment_of("edge").reason == "partially_outside_region"


def test_tile_outside_everything_is_discarded():
    tile = Fake("far", fp(5000, 5000, 5100, 5100))
    out = sp.assign_tiles_to_splits([tile], _two_regions(), buffer_m=0.0)
    assert out.assignment_of("far").reason == "outside_all_regions"


def test_duplicate_tile_ids_are_refused():
    tiles = [Fake("t", fp(-300, -100, -200, 100)), Fake("t", fp(-500, -100, -400, 100))]
    with pytest.raises(ValueError, match="duplicate tile_id"):
        sp.assign_tiles_to_splits(tiles, _two_regions(), buffer_m=0.0)


def test_buffer_exactly_large_enough_versus_one_pixel_too_small():
    """The decisive boundary case, with a 1 m pixel.

    The tile's right edge is at x = -200, the val region starts at x = +500, so
    the separation is exactly 700 projected m.  A 700 m buffer is exactly
    satisfied; moving the tile one pixel closer (or asking for one pixel more)
    must discard it.
    """
    regions = _two_regions()
    exact = Fake("exact", fp(-300, -100, -200, 100))        # 700 m away
    one_px_closer = Fake("closer", fp(-300, -100, -199, 100))  # 699 m away

    ok = sp.assign_tiles_to_splits([exact], regions, buffer_m=700.0,
                                   buffer_kind="projected")
    assert ok.assignment_of("exact").split == "train"
    assert ok.assignment_of("exact").required_separation_projected_m == 700.0

    short = sp.assign_tiles_to_splits([one_px_closer], regions, buffer_m=700.0,
                                      buffer_kind="projected")
    rec = short.assignment_of("closer")
    assert rec.split is None
    assert rec.reason == "within_buffer_of_other_split"
    assert rec.nearest_other_distance_projected_m == pytest.approx(699.0)
    assert "699" in rec.detail and "val" in rec.detail

    # Equivalently: ask for one pixel more than the geometry can give.
    greedy = sp.assign_tiles_to_splits([exact], regions, buffer_m=701.0,
                                       buffer_kind="projected")
    assert greedy.assignment_of("exact").reason == "within_buffer_of_other_split"


def test_ground_buffer_is_stricter_than_a_projected_buffer():
    """INTERFACES.md point 2 applied to a buffer.

    700 projected m is only 700/k = 699.1 ground m at 85.9 S.  A tile exactly
    700 projected m from the other block satisfies a 700 m *projected* buffer but
    NOT a 700 m *ground* buffer, which needs 700*k = 700.9 projected m.
    """
    regions = _two_regions()
    tile = Fake("t", fp(-300, -100, -200, 100))
    projected = sp.assign_tiles_to_splits([tile], regions, buffer_m=700.0,
                                          buffer_kind="projected")
    ground = sp.assign_tiles_to_splits([tile], regions, buffer_m=700.0,
                                       buffer_kind="ground")
    assert projected.assignment_of("t").split == "train"
    rec = ground.assignment_of("t")
    assert rec.split is None and rec.reason == "within_buffer_of_other_split"
    # k is taken at the tile's own centroid, so compare against that.
    assert rec.scale_factor_k == pytest.approx(K, rel=1e-5)
    assert rec.required_separation_projected_m == pytest.approx(
        700.0 * rec.scale_factor_k, rel=1e-12)
    assert rec.required_separation_projected_m > 700.0
    # The reported ground distance is the projected one divided by k.
    assert rec.nearest_other_distance_ground_m == pytest.approx(
        700.0 / rec.scale_factor_k, rel=1e-12)


def test_areas_are_reported_as_projected_and_as_true_ground_area():
    tile = Fake("t", fp(-300, -100, -200, 100))     # 100 x 200 projected m
    out = sp.assign_tiles_to_splits([tile], _two_regions(), buffer_m=0.0)
    rec = out.assignment_of("t")
    assert rec.projected_area_m2 == pytest.approx(100.0 * 200.0)
    assert rec.ground_area_m2 == pytest.approx(
        100.0 * 200.0 / rec.scale_factor_k ** 2, rel=1e-12)
    assert rec.ground_area_m2 == pytest.approx(
        100.0 * 200.0 / g.area_scale_factor(ROI_LAT), rel=1e-4)
    assert rec.ground_area_m2 < rec.projected_area_m2


def test_bad_buffer_arguments_are_refused():
    with pytest.raises(ValueError, match="buffer_m"):
        sp.assign_tiles_to_splits([], _two_regions(), buffer_m=-1.0)
    with pytest.raises(ValueError, match="buffer_kind"):
        sp.assign_tiles_to_splits([], _two_regions(), buffer_m=0.0,
                                  buffer_kind="metres")


# --------------------------------------------------------------------- #
# Leakage finding (a): adjacent grid tiles straddling a split boundary
# --------------------------------------------------------------------- #
def test_two_adjacent_grid_tiles_in_different_splits_must_be_caught():
    """The case that MUST be caught.

    A real grid with a 10 px context margin at 1 m/px: neighbouring tiles
    overlap by 2*margin = 20 px.  Putting them in different splits puts the same
    2000 m^2 of pixels in both, and the report must name both tile ids.
    """
    grid = grid_at(tile_px=100, margin_px=10)
    tiles = grid.tiles(300, 300)
    left = [t for t in tiles if t.row_off == 0 and t.col_off == 0][0]
    right = [t for t in tiles if t.row_off == 0 and t.col_off == grid.stride_px][0]
    a = sp.SplitTile.from_tile(left, "train")
    b = sp.SplitTile.from_tile(right, "val")
    assert a.footprint.intersects(b.footprint)

    rep = sp.check_leakage([a, b], buffer_m=50.0)
    assert rep.clean is False
    found = rep.by_kind("cross_split_footprint_proximity")
    assert len(found) == 1
    f = found[0]
    assert set(f.tile_ids) == {left.tile_id, right.tile_id}
    assert set(f.splits) == {"train", "val"}
    # Overlap is exactly 2*margin wide by one tile tall, at 1 m/px.
    assert f.overlap_area_projected_m2 == pytest.approx(
        grid.overlap_px * grid.tile_px * 1.0, rel=1e-9)
    assert f.measured_projected_m == 0.0
    assert "overlap" in f.detail
    assert "LEAKAGE" in rep.summary
    # No other mechanism is invented for this pair.
    assert rep.kinds_found == ("cross_split_footprint_proximity",)


def test_far_apart_tiles_in_different_splits_are_clean():
    a = stile("a", fp(0, 0, 100, 100), "train")
    b = stile("b", fp(5000, 5000, 5100, 5100), "val")
    rep = sp.check_leakage([a, b], buffer_m=100.0)
    assert rep.clean is True
    assert rep.findings == ()
    assert rep.split_tile_counts == {"train": 1, "val": 1}
    assert "No leakage" in rep.summary


def test_leakage_buffer_boundary_is_exact_to_one_pixel():
    """Separation exactly equal to the buffer passes; one pixel less fails."""
    a = stile("a", fp(0, 0, 100, 100), "train")
    exact = stile("b", fp(200, 0, 300, 100), "val")     # 100 m gap
    tight = stile("b", fp(199, 0, 299, 100), "val")     # 99 m gap
    assert sp.check_leakage([a, exact], buffer_m=100.0,
                            buffer_kind="projected").clean
    rep = sp.check_leakage([a, tight], buffer_m=100.0, buffer_kind="projected")
    assert not rep.clean
    f = rep.by_kind("cross_split_footprint_proximity")[0]
    assert f.measured_projected_m == pytest.approx(99.0)
    assert f.required_projected_m == pytest.approx(100.0)
    assert f.overlap_area_projected_m2 == 0.0
    # Under a ground buffer the "exact" pair also fails: 100 ground m needs
    # 100*k projected m of separation.
    assert not sp.check_leakage([a, exact], buffer_m=100.0,
                                buffer_kind="ground").clean


def test_same_split_neighbours_are_not_leakage():
    grid = grid_at()
    tiles = grid.tiles(300, 300)
    claimed = [sp.SplitTile.from_tile(t, "train") for t in tiles]
    assert sp.check_leakage(claimed, buffer_m=50.0).clean


# --------------------------------------------------------------------- #
# Leakage finding (b): the same crater in two splits
# --------------------------------------------------------------------- #
def test_the_same_crater_in_two_splits_is_caught_by_id():
    a = stile("a", fp(0, 0, 100, 100), "train", crater_ids=["C-7", "C-8"])
    b = stile("b", fp(9000, 9000, 9100, 9100), "test", crater_ids=["C-7"])
    rep = sp.check_leakage([a, b], buffer_m=100.0)
    assert not rep.clean
    # The footprints are far apart, so ONLY the crater mechanism fires.
    assert rep.kinds_found == ("crater_id_in_multiple_splits",)
    f = rep.by_kind("crater_id_in_multiple_splits")[0]
    assert f.ids == ("C-7",)
    assert set(f.tile_ids) == {"a", "b"}
    assert f.splits == ("test", "train")
    assert "C-7" in f.detail
    # C-8 lives in one split only and is not flagged.
    assert all("C-8" not in f.ids for f in rep.findings)


def test_a_crater_in_many_tiles_of_one_split_is_not_leakage():
    a = stile("a", fp(0, 0, 100, 100), "train", crater_ids=["C-1"])
    b = stile("b", fp(9000, 9000, 9100, 9100), "train", crater_ids=["C-1"])
    assert sp.check_leakage([a, b], buffer_m=100.0).clean


# --------------------------------------------------------------------- #
# Leakage finding (c): same ground at different pyramid scales
# --------------------------------------------------------------------- #
def test_the_same_ground_at_two_scales_in_two_splits_is_caught():
    """A level-0 tile in train and a level-2 tile in val over the same massif."""
    fine = stile("L0_a", fp(0, 0, 100, 100), "train", level=0)
    coarse = stile("L2_a", fp(50, 50, 450, 450), "val", level=2)
    rep = sp.check_leakage([fine, coarse], buffer_m=10.0)
    assert not rep.clean
    assert rep.kinds_found == ("same_ground_area_at_different_scales",)
    f = rep.by_kind("same_ground_area_at_different_scales")[0]
    assert set(f.tile_ids) == {"L0_a", "L2_a"}
    assert f.overlap_area_projected_m2 == pytest.approx(50.0 * 50.0)
    assert "L0" in f.detail and "L2" in f.detail
    # Not mislabelled as the plain same-scale case.
    assert rep.by_kind("cross_split_footprint_proximity") == ()


def test_multiscale_tiles_over_the_same_ground_in_one_split_are_clean():
    fine = stile("L0_a", fp(0, 0, 100, 100), "train", level=0)
    coarse = stile("L2_a", fp(50, 50, 450, 450), "train", level=2)
    assert sp.check_leakage([fine, coarse], buffer_m=10.0).clean


# --------------------------------------------------------------------- #
# Leakage finding (d): alternate acquisitions of the same ground
# --------------------------------------------------------------------- #
def test_alternate_acquisitions_of_the_same_ground_are_caught():
    same_ground = fp(0, 0, 100, 100)
    a = stile("M1_t0", same_ground, "train", source_id="M1104872100")
    b = stile("M2_t0", same_ground, "val", source_id="M1104872188")
    rep = sp.check_leakage([a, b], buffer_m=10.0)
    assert not rep.clean
    assert rep.kinds_found == ("alternate_acquisition_of_same_area",)
    f = rep.by_kind("alternate_acquisition_of_same_area")[0]
    assert set(f.tile_ids) == {"M1_t0", "M2_t0"}
    assert f.overlap_area_projected_m2 == pytest.approx(100.0 * 100.0)
    assert "M1104872100" in f.detail and "M1104872188" in f.detail


def test_a_pair_differing_in_both_scale_and_acquisition_names_both_mechanisms():
    a = stile("a", fp(0, 0, 100, 100), "train", level=0, source_id="M1")
    b = stile("b", fp(50, 50, 450, 450), "val", level=2, source_id="M2")
    rep = sp.check_leakage([a, b], buffer_m=10.0)
    assert set(rep.kinds_found) == {"same_ground_area_at_different_scales",
                                    "alternate_acquisition_of_same_area"}
    assert rep.counts()["cross_split_footprint_proximity"] == 0
    for f in rep.findings:
        assert set(f.tile_ids) == {"a", "b"}


def test_one_tile_id_claimed_by_two_splits_is_a_named_finding():
    a = stile("dup", fp(0, 0, 100, 100), "train")
    b = stile("dup", fp(9000, 9000, 9100, 9100), "val")
    rep = sp.check_leakage([a, b], buffer_m=10.0)
    assert rep.by_kind("duplicate_tile_id_across_splits")[0].ids == ("dup",)


def test_every_mechanism_can_fire_at_once_and_is_reported_separately():
    grid = grid_at()
    tiles = grid.tiles(300, 300)
    left, right = tiles[0], tiles[1]
    claimed = [
        sp.SplitTile.from_tile(left, "train", crater_ids=["C-1"]),
        sp.SplitTile.from_tile(right, "val", crater_ids=["C-1"]),
        stile("coarse", fp(2000, 2000, 2400, 2400), "train", level=2),
        stile("fine", fp(2100, 2100, 2200, 2200), "val", level=0),
        stile("acqA", fp(4000, 4000, 4100, 4100), "train", source_id="M1"),
        stile("acqB", fp(4000, 4000, 4100, 4100), "val", source_id="M2"),
    ]
    rep = sp.check_leakage(claimed, buffer_m=20.0)
    assert set(rep.kinds_found) == {
        "cross_split_footprint_proximity",
        "crater_id_in_multiple_splits",
        "same_ground_area_at_different_scales",
        "alternate_acquisition_of_same_area",
    }
    counts = rep.counts()
    assert counts["cross_split_footprint_proximity"] == 1
    assert counts["crater_id_in_multiple_splits"] == 1
    assert counts["same_ground_area_at_different_scales"] == 1
    assert counts["alternate_acquisition_of_same_area"] == 1


def test_leakage_rejects_bad_buffer_arguments():
    with pytest.raises(ValueError):
        sp.check_leakage([], buffer_m=-1.0)
    with pytest.raises(ValueError):
        sp.check_leakage([], buffer_m=0.0, buffer_kind="nautical_miles")


# --------------------------------------------------------------------- #
# Block construction
# --------------------------------------------------------------------- #
def test_block_regions_partition_the_rectangle_contiguously():
    bounds = (CX, CY, CX + 1000.0, CY + 2000.0)
    regions = sp.block_regions(bounds, (0.7, 0.15, 0.15), ("train", "val", "test"))
    assert [r.name for r in regions] == ["train", "val", "test"]
    total = sum(r.projected_area_m2 for r in regions)
    assert total == pytest.approx(1000.0 * 2000.0)
    # Exactly the requested fractions, and the bands are edge-adjacent.
    assert regions[0].projected_area_m2 / total == pytest.approx(0.70)
    assert regions[1].projected_area_m2 / total == pytest.approx(0.15)
    assert regions[0].geometry.touches(regions[1].geometry)
    assert regions[0].geometry.intersection(regions[1].geometry).area == 0.0


def test_block_regions_refuse_to_renormalise_a_bad_target():
    bounds = (CX, CY, CX + 1000.0, CY + 1000.0)
    with pytest.raises(ValueError, match="sum to 1"):
        sp.block_regions(bounds, (0.7, 0.15), ("train", "val"))
    with pytest.raises(ValueError, match="> 0"):
        sp.block_regions(bounds, (0.9, 0.1, 0.0), ("train", "val", "test"))


# --------------------------------------------------------------------- #
# The 70/15/15 target: reported, never forced
# --------------------------------------------------------------------- #
def test_seventy_fifteen_fifteen_is_reported_as_unachievable_when_it_is():
    """A 3 km ROI with 256 m tiles and a 200 m buffer cannot support 70/15/15.

    Each 15% band is 450 m wide; losing 200 m of buffer at each edge leaves
    50 m, which cannot contain a 256 m tile.  The honest answer is that val and
    test get nothing -- not a quietly widened band.
    """
    grid = grid_at(tile_px=256, margin_px=16, origin=(-1500.0, 1500.0))
    tiles = grid.tiles(3000, 3000)
    with pytest.warns(sp.SplitGeometryWarning, match="NO tiles"):
        plan = sp.plan_block_splits(tiles, buffer_m=200.0)

    assert plan.feasible is False
    assert set(plan.empty_splits) == {"val", "test"}
    assert plan.tile_counts["val"] == 0 and plan.tile_counts["test"] == 0
    assert plan.tile_counts["train"] > 0
    # The blocks themselves were cut at exactly the target: nothing was resized.
    assert plan.region_area_ratio["train"] == pytest.approx(0.70)
    assert plan.region_area_ratio["val"] == pytest.approx(0.15)
    assert plan.region_area_ratio["test"] == pytest.approx(0.15)
    # But the achieved ratio is reported as what it is.
    assert plan.achieved_area_ratio["train"] == pytest.approx(1.0)
    assert plan.achieved_area_ratio["val"] == 0.0
    assert plan.max_abs_area_deviation == pytest.approx(0.30, abs=1e-9)
    assert plan.discarded_tiles > 0
    assert any("NOT been forced" in w for w in plan.warnings)
    assert "TARGET NOT ACHIEVABLE" in plan.message
    # Every discarded tile still carries a reason.
    assert all(a.reason in sp.DISCARD_REASONS for a in plan.assignment.discarded)


def test_seventy_fifteen_fifteen_is_achievable_on_a_long_roi():
    """Given enough extent along the cut axis, the target is met and said so."""
    grid = grid_at(tile_px=512, margin_px=32, origin=(-1000.0, 10000.0))
    tiles = grid.tiles(2000, 20000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.SplitGeometryWarning)
        plan = sp.plan_block_splits(tiles, buffer_m=200.0, tolerance=0.10)
    assert plan.empty_splits == ()
    assert all(plan.tile_counts[n] > 0 for n in ("train", "val", "test"))
    assert sum(plan.achieved_area_ratio.values()) == pytest.approx(1.0)
    assert plan.achieved_area_ratio["train"] > plan.achieved_area_ratio["val"]
    assert plan.feasible is True
    # Achieved is still not identical to the target: attrition is real and shown.
    assert plan.achieved_area_ratio["train"] != pytest.approx(0.70, abs=1e-6)
    assert plan.discarded_tiles > 0
    # Ground area is the projected area divided by k**2.
    for name in ("train", "val", "test"):
        assert plan.usable_ground_area_m2[name] < plan.usable_projected_area_m2[name]
        assert plan.usable_ground_area_m2[name] == pytest.approx(
            plan.usable_projected_area_m2[name] / g.area_scale_factor(ROI_LAT),
            rel=1e-3)


def test_a_plan_built_by_this_api_is_leakage_free_by_construction():
    """The strong property: anything this assigner places passes the audit.

    Each assigned tile is at least the buffer away from every foreign region, and
    foreign tiles live inside those regions, so cross-split pairs cannot come
    closer than the buffer.  If this ever fails, one of the two halves of the
    module is wrong.
    """
    grid = grid_at(tile_px=512, margin_px=32, origin=(-1000.0, 10000.0))
    tiles = grid.tiles(2000, 20000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.SplitGeometryWarning)
        plan = sp.plan_block_splits(tiles, buffer_m=200.0, tolerance=0.10)
    rep = sp.check_leakage(plan.assignment.split_tiles(), buffer_m=200.0)
    assert rep.clean, rep.summary


def test_a_random_tile_split_of_the_same_grid_fails_the_audit():
    """Contrast case: label the very same tiles at random and watch it light up.

    This split cannot be expressed through the assignment API at all; it has to
    be hand-built, and the auditor rejects it.
    """
    grid = grid_at(tile_px=128, margin_px=16, origin=(-500.0, 500.0))
    tiles = grid.tiles(1000, 1000)
    # Deterministic interleaving -- the worst case of a "random" tile split.
    claimed = [sp.SplitTile.from_tile(t, ("train", "val", "test")[i % 3])
               for i, t in enumerate(tiles)]
    rep = sp.check_leakage(claimed, buffer_m=100.0)
    assert not rep.clean
    assert len(rep.by_kind("cross_split_footprint_proximity")) > 10
    overlapping = [f for f in rep.findings if f.overlap_area_projected_m2 > 0]
    assert overlapping, "interleaved tiles share pixels outright"


def test_plan_requires_tiles_and_matching_names():
    grid = grid_at()
    tiles = grid.tiles(300, 300)
    with pytest.raises(ValueError, match="no tiles"):
        sp.plan_block_splits([], buffer_m=0.0)
    with pytest.raises(ValueError, match="target fractions"):
        sp.plan_block_splits(tiles, buffer_m=0.0, target_ratio=(0.5, 0.5),
                             names=("a", "b", "c"))


def test_plan_accepts_an_explicit_roi_and_axis():
    grid = grid_at(tile_px=256, margin_px=16, origin=(-1500.0, 1500.0))
    tiles = grid.tiles(3000, 3000)
    bounds = (CX - 1500.0, CY - 1500.0, CX + 1500.0, CY + 1500.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", sp.SplitGeometryWarning)
        plan_x = sp.plan_block_splits(tiles, buffer_m=100.0, bounds=bounds, axis="x")
        plan_y = sp.plan_block_splits(tiles, buffer_m=100.0, bounds=bounds, axis="y")
    assert plan_x.axis == "x" and plan_y.axis == "y"
    # Cutting the same square ROI either way gives the same block areas.
    assert (sorted(plan_x.region_projected_area_m2.values())
            == pytest.approx(sorted(plan_y.region_projected_area_m2.values())))
