"""Tests for usable survey area on the lunar sphere.

The anchor of this file is an **analytic** area: the surface of a spherical cap
is ``2 pi R**2 (1 - cos theta)`` exactly, with no projection and no pixels in
it.  Every mask-based area here is checked against that closed form, so the
tests compare two independent routes to the same number rather than comparing
``area.py`` with itself.

The second anchor is the areal scale factor.  A pixel count is a *projected*
area; dividing by ``k**2`` is what makes it a surface area.  Several tests below
pin the size of the error that skipping that division makes -- 0.244 % at 86 S,
and much more further from the pole -- so that a regression which drops the
division fails loudly instead of shifting every crater density by a quarter of a
percent in silence.
"""
import numpy as np
import pytest

from crater import area as a
from crater import geometry as g
from crater.body import MOON

R = MOON.radius_m


def cap_grid(colatitude_deg, pixel_size_m, pad=1.05):
    """A grid comfortably containing the south-polar cap of that colatitude."""
    rho = 2.0 * R * np.tan(np.radians(colatitude_deg) / 2.0)
    return a.StereographicGrid.centred_on_projected_point(
        0.0, 0.0, rho * pad, pixel_size_m
    )


# --------------------------------------------------------------------- #
# The analytic reference
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("colat,expected", [
    (0.0, 0.0),
    (90.0, 2.0 * np.pi * R**2),            # hemisphere
    (180.0, 4.0 * np.pi * R**2),           # whole sphere
    (60.0, np.pi * R**2),                  # 1 - cos 60 = 1/2
])
def test_spherical_cap_area_closed_form(colat, expected):
    assert a.spherical_cap_area_m2(colat) == pytest.approx(expected, rel=1e-12)


def test_spherical_cap_rejects_impossible_colatitude():
    with pytest.raises(ValueError):
        a.spherical_cap_area_m2(-1.0)
    with pytest.raises(ValueError):
        a.spherical_cap_area_m2(181.0)


def test_small_cap_approaches_the_flat_disc_from_below():
    """A 15 km cap is 7e-6 smaller than a flat disc of the same arc radius --
    the sign matters: curvature makes the cap area smaller, so an answer that
    came out larger than pi*r**2 would be wrong."""
    theta = np.degrees(15_000.0 / R)
    cap = a.spherical_cap_area_m2(theta)
    disc = np.pi * 15_000.0**2
    assert cap < disc
    assert cap == pytest.approx(disc, rel=1e-5)


def test_colatitude_of_latitude():
    assert a.colatitude_of_latitude_deg(-90.0) == pytest.approx(0.0)
    assert a.colatitude_of_latitude_deg(-86.0) == pytest.approx(4.0)
    assert float(a.colatitude_of_latitude_deg(0.0)) == pytest.approx(90.0)


# --------------------------------------------------------------------- #
# Mask-based area converges to the analytic cap
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("colat", [1.0, 4.0, 10.0])
def test_mask_area_matches_spherical_cap(colat):
    """The headline check: a validity mask integrated with the k**2 division
    reproduces 2 pi R**2 (1 - cos theta)."""
    rho = 2.0 * R * np.tan(np.radians(colat) / 2.0)
    grid = cap_grid(colat, pixel_size_m=2.0 * rho * 1.05 / 1001)
    mask = a.mask_from_colatitude(grid, colat)
    assert a.true_area_m2(mask, grid) == pytest.approx(
        a.spherical_cap_area_m2(colat), rel=2e-4
    )


def test_mask_area_converges_as_the_pixel_shrinks():
    """Refining the grid must reduce the error towards zero.

    The error of a single pixelated disc is not monotone in resolution -- it
    depends on where the boundary happens to fall between pixel centres -- so the
    comparison is made on the RMS over four nearby grid sizes at each scale,
    which removes that phase dependence.
    """
    colat = 4.0
    exact = a.spherical_cap_area_m2(colat)
    rho = 2.0 * R * np.tan(np.radians(colat) / 2.0)

    def rms_error(sizes):
        errs = []
        for n in sizes:
            grid = cap_grid(colat, pixel_size_m=2.0 * rho * 1.05 / n)
            errs.append(a.true_area_m2(a.mask_from_colatitude(grid, colat), grid) / exact - 1.0)
        return float(np.sqrt(np.mean(np.square(errs))))

    coarse = rms_error((81, 97, 113, 129))
    fine = rms_error((1001, 1051, 1101, 1151))
    assert fine < coarse / 10.0
    assert fine < 1e-4


def test_single_pixel_area_is_projected_area_over_k_squared():
    """The whole correction, on one pixel, written out by hand."""
    grid = a.StereographicGrid.centred_on_lonlat(2.9, -85.9, half_extent_m=50.0,
                                                 pixel_size_m=100.0)
    assert grid.shape == (1, 1)
    lat = float(grid.pixel_latitudes()[0, 0])
    expected = 100.0**2 / g.area_scale_factor(lat)
    mask = np.ones(grid.shape, dtype=bool)
    assert a.true_area_m2(mask, grid) == pytest.approx(expected, rel=1e-12)
    # and the correction is a real one, not a no-op
    assert expected < 100.0**2


# --------------------------------------------------------------------- #
# The error a naive pixel count makes
# --------------------------------------------------------------------- #
def test_naive_pixel_count_is_wrong_by_0_24_percent_at_86_south():
    """INTERFACES.md rule 2: a projected area at 86 S is 0.244 % too large.

    The number is pinned to five digits, not merely 'small': a bias that is
    constant in sign shifts every crater density and every R value the same way,
    so it must be removed rather than absorbed into an error bar.
    """
    rho = 2.0 * R * np.tan(np.radians(4.0) / 2.0)
    grid = a.StereographicGrid.centred_on_projected_point(0.0, 0.0, rho * 1.02, 100.0)
    # A thin annulus hugging colatitude 4 deg, i.e. latitude -86 exactly.
    mask = a.mask_from_colatitude(grid, 4.005, 3.995)
    assert mask.sum() > 10_000

    expected = float(g.area_scale_factor(-86.0)) - 1.0
    assert expected == pytest.approx(0.00244, abs=1e-5)
    assert a.naive_area_relative_error(mask, grid) == pytest.approx(expected, rel=1e-4)

    naive = a.naive_projected_area_m2(mask, grid)
    true = a.true_area_m2(mask, grid)
    assert naive > true                       # always an over-estimate
    assert naive / true - 1.0 == pytest.approx(0.002440, abs=2e-6)


@pytest.mark.parametrize("lat,floor", [(-80.0, 0.015), (-70.0, 0.063), (-45.0, 0.37)])
def test_naive_error_is_far_worse_away_from_the_pole(lat, floor):
    """0.24 % is the best case. The same mistake costs 1.5 %, 6 % and 37 %."""
    colat = 90.0 + lat
    rho = 2.0 * R * np.tan(np.radians(colat) / 2.0)
    grid = a.StereographicGrid.centred_on_projected_point(0.0, 0.0, rho * 1.02,
                                                          rho * 0.001)
    mask = a.mask_from_colatitude(grid, colat + 0.005, colat - 0.005)
    err = a.naive_area_relative_error(mask, grid)
    assert err == pytest.approx(float(g.area_scale_factor(lat)) - 1.0, rel=1e-3)
    assert err > floor


def test_naive_error_of_an_empty_mask_is_nan_not_zero():
    """No pixels means no measurable error; a fabricated 0.0 would read as
    'no distortion' (INTERFACES.md rule 6)."""
    grid = cap_grid(1.0, pixel_size_m=1000.0)
    empty = np.zeros(grid.shape, dtype=bool)
    assert np.isnan(a.naive_area_relative_error(empty, grid))
    assert a.true_area_m2(empty, grid) == 0.0


def test_km2_helper_is_a_pure_unit_change():
    grid = cap_grid(2.0, pixel_size_m=500.0)
    mask = a.mask_from_colatitude(grid, 2.0)
    assert a.true_area_km2(mask, grid) == pytest.approx(
        a.true_area_m2(mask, grid) / 1.0e6, rel=1e-15
    )


# --------------------------------------------------------------------- #
# Mask assembly: nodata, exclusions, overlapping sources
# --------------------------------------------------------------------- #
def test_overlapping_source_coverage_is_counted_once():
    """Two strips sharing ground must contribute that ground once; summing the
    two footprint areas would double-count the seam."""
    grid = a.StereographicGrid.centred_on_projected_point(0.0, 0.0, 10_000.0, 200.0)
    left = np.zeros(grid.shape, dtype=bool)
    right = np.zeros(grid.shape, dtype=bool)
    left[:, :60] = True
    right[:, 40:] = True                      # 20 columns of overlap

    union = a.union_coverage([left, right])
    assert union.all()
    union_area = a.true_area_m2(union, grid)
    summed = a.true_area_m2(left, grid) + a.true_area_m2(right, grid)
    overlap = np.zeros(grid.shape, dtype=bool)
    overlap[:, 40:60] = True
    assert summed - union_area == pytest.approx(a.true_area_m2(overlap, grid), rel=1e-12)
    assert union_area < summed


def test_usable_mask_removes_nodata_and_excluded_terrain():
    grid = a.StereographicGrid.centred_on_projected_point(0.0, 0.0, 10_000.0, 200.0)
    cov = np.ones(grid.shape, dtype=bool)
    nodata = np.zeros(grid.shape, dtype=bool)
    nodata[:5, :] = True                      # a fill-value margin
    shadow = np.zeros(grid.shape, dtype=bool)
    shadow[-7:, :] = True                     # permanently shadowed, unmappable

    usable = a.usable_mask([cov], nodata_masks=[nodata], excluded_masks=[shadow])
    assert not usable[:5, :].any()
    assert not usable[-7:, :].any()
    assert usable[5:-7, :].all()

    full = a.true_area_m2(cov, grid)
    kept = a.true_area_m2(usable, grid)
    assert kept < full
    assert kept + a.true_area_m2(nodata | shadow, grid) == pytest.approx(full, rel=1e-12)


def test_overlapping_exclusions_are_not_subtracted_twice():
    """Two exclusion layers over the same pixels remove it once, not twice --
    the OR, not the sum."""
    grid = a.StereographicGrid.centred_on_projected_point(0.0, 0.0, 5_000.0, 200.0)
    cov = np.ones(grid.shape, dtype=bool)
    e1 = np.zeros(grid.shape, dtype=bool); e1[:, :10] = True
    e2 = np.zeros(grid.shape, dtype=bool); e2[:, 5:15] = True
    usable = a.usable_mask([cov], excluded_masks=[e1, e2])
    assert usable.sum() == grid.width * (grid.width - 15)


def test_masks_must_be_boolean_and_the_right_shape():
    """A float nodata array is truthy for every non-zero fill value; refusing it
    is cheaper than debugging an area that is silently 3 % too large."""
    grid = a.StereographicGrid.centred_on_projected_point(0.0, 0.0, 1_000.0, 200.0)
    with pytest.raises(TypeError):
        a.true_area_m2(np.ones(grid.shape, dtype=float), grid)
    with pytest.raises(ValueError):
        a.true_area_m2(np.ones((3, 3), dtype=bool), grid)


def test_union_coverage_needs_a_shape_when_empty():
    with pytest.raises(ValueError):
        a.union_coverage([])
    assert a.union_coverage([], shape=(2, 3)).shape == (2, 3)


# --------------------------------------------------------------------- #
# Diameter-dependent counting area A_i (the edge rule)
# --------------------------------------------------------------------- #
def test_edge_rule_is_documented():
    assert "D/2" in a.EDGE_RULE
    assert a.EDGE_DISTANCE_KIND == g.DIAMETER_DISTANCE_KIND


def test_usable_area_by_diameter_matches_the_shrunken_analytic_cap():
    """Eroding a cap of ground radius r0 by D/2 leaves a cap of radius r0 - D/2,
    whose area is known in closed form."""
    colat = 0.5                                   # ~15.2 km ground radius
    r0 = np.radians(colat) * R
    grid = cap_grid(colat, pixel_size_m=40.0, pad=1.1)
    mask = a.mask_from_colatitude(grid, colat)

    diameters = np.array([200.0, 1000.0, 5000.0, 10_000.0])
    areas = a.usable_area_m2_by_diameter(grid, mask, diameters)
    for d, got in zip(diameters, areas):
        shrunk_colat = np.degrees((r0 - d / 2.0) / R)
        assert got == pytest.approx(a.spherical_cap_area_m2(shrunk_colat), rel=5e-3)


def test_usable_area_shrinks_monotonically_with_diameter():
    colat = 0.5
    grid = cap_grid(colat, pixel_size_m=60.0, pad=1.1)
    mask = a.mask_from_colatitude(grid, colat)
    diameters = np.array([0.0, 20.0, 100.0, 300.0, 1000.0, 3000.0, 20_000.0])
    areas = a.usable_area_m2_by_diameter(grid, mask, diameters)
    assert np.all(np.diff(areas) <= 0.0)
    assert areas[0] == pytest.approx(a.true_area_m2(mask, grid), rel=1e-12)


def test_usable_area_reaches_exactly_zero_and_is_not_clipped():
    """Past the inradius no crater centre can satisfy the edge rule. The answer
    is 0, and a 0 must survive to the caller so the bin is reported as having no
    counting area rather than being divided by an invented epsilon."""
    colat = 0.2
    grid = cap_grid(colat, pixel_size_m=200.0, pad=1.1)
    mask = a.mask_from_colatitude(grid, colat)
    huge = 4.0 * np.radians(colat) * R           # diameter > cap diameter
    assert a.usable_area_m2_by_diameter(grid, mask, np.array([huge]))[0] == 0.0


def test_grid_edge_counts_as_a_survey_boundary():
    """A mask that fills the array is still bounded -- by the array. Treating the
    edge as interior would hand back the full area for every diameter."""
    grid = a.StereographicGrid.centred_on_projected_point(0.0, 0.0, 10_000.0, 100.0)
    full = np.ones(grid.shape, dtype=bool)
    a0 = a.true_area_m2(full, grid)
    a1 = a.usable_area_m2_by_diameter(grid, full, np.array([1000.0]))[0]
    assert a1 < a0
    # A 500 m ground buffer off a 20 km square: ~ (20-1)^2 / 20^2 of the area.
    assert a1 / a0 == pytest.approx((19.0 / 20.0) ** 2, rel=5e-3)


def test_edge_rule_selection_agrees_with_the_area_it_normalises():
    """The craters counted and the area they are divided by must come from the
    same rule: the fraction of uniformly scattered centres that pass must equal
    A_i / A_0."""
    rng = np.random.default_rng(20261001)
    colat = 0.5
    grid = cap_grid(colat, pixel_size_m=50.0, pad=1.1)
    mask = a.mask_from_colatitude(grid, colat)
    rho = 2.0 * R * np.tan(np.radians(colat) / 2.0)

    n = 40_000
    radius = rho * np.sqrt(rng.random(n))        # uniform over the projected disc
    phi = rng.uniform(0.0, 2 * np.pi, n)
    x, y = radius * np.cos(phi), radius * np.sin(phi)

    d = 3000.0
    passed = a.crater_passes_edge_rule(grid, mask, x, y, d)
    expected = (
        a.usable_area_m2_by_diameter(grid, mask, np.array([d]))[0]
        / a.true_area_m2(mask, grid)
    )
    assert passed.mean() == pytest.approx(expected, abs=0.01)


def test_crater_outside_the_grid_fails_the_edge_rule():
    grid = cap_grid(0.5, pixel_size_m=100.0)
    mask = a.mask_from_colatitude(grid, 0.5)
    far = 10.0 * R
    assert not a.crater_passes_edge_rule(grid, mask, far, far, 20.0)


def test_boundary_distance_is_a_ground_distance_not_a_projected_one():
    """The distance transform works in projected metres; dividing by k is what
    makes the buffer a real D/2 on the sphere."""
    grid = a.StereographicGrid.centred_on_projected_point(0.0, 0.0, 20_000.0, 100.0)
    mask = np.ones(grid.shape, dtype=bool)
    dist = a.boundary_distance_m(grid, mask)
    k = grid.point_scale_factor()
    row = col = grid.height // 2
    centre = (row, col)
    # Pixels from this centre to the nearest unusable pixel of the padded array,
    # less the half pixel by which the boundary itself sits nearer.
    n_px = min(row + 1, grid.height - row, col + 1, grid.width - col)
    projected = (n_px - 0.5) * grid.pixel_size_m
    assert dist[centre] == pytest.approx(projected / k[centre], rel=1e-12)
    assert dist[centre] < projected            # k > 1 away from the pole


def test_bin_representative_diameter_choice():
    low, high = np.array([20.0]), np.array([40.0])
    assert a.bin_representative_diameters_m(low, high, "high")[0] == 40.0
    assert a.bin_representative_diameters_m(low, high, "geometric")[0] == pytest.approx(
        np.sqrt(800.0)
    )
    with pytest.raises(ValueError):
        a.bin_representative_diameters_m(low, high, "midpoint")


def test_negative_or_nonfinite_diameters_are_rejected():
    grid = cap_grid(0.5, pixel_size_m=200.0)
    mask = a.mask_from_colatitude(grid, 0.5)
    with pytest.raises(ValueError):
        a.usable_area_m2_by_diameter(grid, mask, np.array([-1.0]))
    with pytest.raises(ValueError):
        a.usable_area_m2_by_diameter(grid, mask, np.array([np.nan]))


# --------------------------------------------------------------------- #
# Grid bookkeeping
# --------------------------------------------------------------------- #
def test_grid_rejects_degenerate_geometry():
    with pytest.raises(ValueError):
        a.StereographicGrid(0.0, 0.0, 0.0, 10, 10)
    with pytest.raises(ValueError):
        a.StereographicGrid(0.0, 0.0, 100.0, 0, 10)


def test_grid_pixel_centres_round_trip_through_the_projection():
    grid = a.StereographicGrid.centred_on_lonlat(2.9, -85.9, 5_000.0, 250.0)
    x, y = grid.pixel_centres_xy()
    lon, lat = grid.pixel_centres_lonlat()
    x2, y2 = g.forward(lon, lat)
    assert np.allclose(x2, x, atol=1e-6)
    assert np.allclose(y2, y, atol=1e-6)


def test_grid_is_centred_where_asked():
    lon0, lat0 = -3.6, -85.8
    grid = a.StereographicGrid.centred_on_lonlat(lon0, lat0, 4_000.0, 200.0)
    x, y = grid.pixel_centres_xy()
    cx, cy = g.forward(lon0, lat0)
    assert x.mean() == pytest.approx(cx, abs=1e-6)
    assert y.mean() == pytest.approx(cy, abs=1e-6)


def test_true_pixel_areas_decrease_away_from_the_pole():
    """k grows away from the pole, so the ground footprint of a fixed projected
    pixel shrinks. A grid whose pixel areas were constant would be planar."""
    grid = a.StereographicGrid.centred_on_projected_point(0.0, 0.0, 200_000.0, 2_000.0)
    areas = grid.pixel_true_areas_m2()
    colat = grid.pixel_colatitudes()
    near = areas[colat < np.percentile(colat, 5)]
    far = areas[colat > np.percentile(colat, 95)]
    assert far.max() < near.min()
    assert np.all(areas <= grid.projected_pixel_area_m2)
