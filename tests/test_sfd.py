"""Tests for the crater size-frequency distribution and the R-plot.

The scientific anchor of this file is an identity, not a tolerance.  For a
differential power law ``dN/dD = C D**-3`` the expected count in a bin
``[lo, hi)`` is ``C/2 (lo**-2 - hi**-2)``, and substituting that into
``R = D_g**3 N / (A dD)`` gives::

    R = C / (2 A) * (1 + r) / sqrt(r),     r = hi / lo

with no dependence on the bin's absolute diameter.  So for constant-ratio log
bins, **R is exactly flat for a -3 slope** -- that is what R is for.  The key
test here draws a synthetic ``D**-3`` population and requires the pipeline to
return that flat line, both bin by bin against the analytic value and as a
fitted log-log slope of zero.  A control population with a -4 slope must *not*
come out flat, so the test cannot be passed by a function that returns a
constant.

Everything else guards a specific way the result can be wrong while still
looking plausible: units silently mixed between metres and kilometres, a crater
on a bin edge counted twice or not at all, ``sqrt(N)`` error bars at N = 1,
an epsilon slipped in to keep ``log(0)`` finite, measurement error and counting
error merged into one bar, and the data being ``log10``-transformed before being
handed to an already-logarithmic axis.
"""
import os

import numpy as np
import pytest

from crater import area as a
from crater import sfd

# A population that is exactly a power law by construction: D**-3 differential,
# inverse-CDF sampled, so the expected count per bin is known in closed form.
D_MIN_M, D_MAX_M = 20.0, 1280.0        # 1280 = 20 * 2**6, so the octave grid
BINS_PER_OCTAVE = 4                    # closes exactly on both ends
N_FULL_BINS = 6 * BINS_PER_OCTAVE
AREA_KM2 = 950.0


@pytest.fixture(scope="module")
def edges():
    return sfd.log_bin_edges_m(
        D_MIN_M, D_MAX_M, bins_per_octave=BINS_PER_OCTAVE, anchor_m=D_MIN_M
    )


def power_law_sample(n, slope=-3.0, d_min=D_MIN_M, d_max=D_MAX_M, seed=20261001):
    """Inverse-CDF draw from ``dN/dD ∝ D**slope`` on ``[d_min, d_max)``."""
    rng = np.random.default_rng(seed)
    p = slope + 1.0                       # exponent of the integrated CDF
    lo, hi = d_min**p, d_max**p
    return (lo + rng.random(n) * (hi - lo)) ** (1.0 / p)


def expected_counts(n, edges_m, slope=-3.0, d_min=D_MIN_M, d_max=D_MAX_M):
    """Expected number of those draws in each bin (exact, no simulation)."""
    p = slope + 1.0
    lo, hi = np.asarray(edges_m[:-1]), np.asarray(edges_m[1:])
    cdf = lambda d: (np.clip(d, d_min, d_max) ** p - d_min**p) / (d_max**p - d_min**p)
    return n * (cdf(hi) - cdf(lo))


# --------------------------------------------------------------------- #
# Binning: convention, units, edges
# --------------------------------------------------------------------- #
def test_bins_are_a_constant_ratio_geometric_progression(edges):
    ratio = edges[1:] / edges[:-1]
    assert np.allclose(ratio, 2.0 ** (1.0 / BINS_PER_OCTAVE), rtol=1e-14)
    assert edges[0] == pytest.approx(D_MIN_M, rel=1e-14)
    assert edges[-1] > D_MAX_M            # strictly, so D_max is never dropped
    assert np.all(np.diff(edges) > 0)


def test_bin_grid_is_data_independent_so_two_series_share_it():
    """Comparing a catalogue with a detector bin by bin only means something on
    identical edges, so the grid must not depend on either series' range."""
    wide = sfd.log_bin_edges_m(20.0, 1000.0, bins_per_octave=4, anchor_m=1.0)
    narrow = sfd.log_bin_edges_m(60.0, 400.0, bins_per_octave=4, anchor_m=1.0)
    common = np.intersect1d(np.round(wide, 9), np.round(narrow, 9))
    assert len(common) == len(narrow)      # narrow's edges all lie on wide's grid


def test_configurable_progression():
    octave = sfd.log_bin_edges_m(10.0, 100.0, bins_per_octave=1, anchor_m=10.0)
    assert np.allclose(octave[1:] / octave[:-1], 2.0)
    fine = sfd.log_bin_edges_m(10.0, 100.0, bins_per_octave=18, base=10.0, anchor_m=10.0)
    assert np.allclose(fine[1:] / fine[:-1], 10.0 ** (1.0 / 18))


@pytest.mark.parametrize("bad", [
    dict(d_min_m=0.0, d_max_m=10.0),
    dict(d_min_m=10.0, d_max_m=1.0),
    dict(d_min_m=1.0, d_max_m=10.0, bins_per_octave=0),
    dict(d_min_m=1.0, d_max_m=10.0, base=1.0),
])
def test_bin_edges_reject_nonsense(bad):
    with pytest.raises(ValueError):
        sfd.log_bin_edges_m(**bad)


def test_crater_exactly_on_a_bin_boundary_goes_into_the_bin_it_opens(edges):
    """Half-open [D_low, D_high): the edge value belongs to the upper bin.
    A closed-closed rule would count it twice; open-open would lose it."""
    assert sfd.BIN_INTERVAL_CONVENTION == "half_open_low_inclusive"
    on_edge = edges[3]
    idx = sfd.bin_index(np.array([on_edge]), edges)[0]
    assert idx == 3
    assert edges[idx] <= on_edge < edges[idx + 1]

    counts, below, above = sfd.bin_counts(np.array([on_edge]), edges)
    assert counts.sum() == 1 and below == 0 and above == 0
    assert counts[3] == 1 and counts[2] == 0


def test_every_crater_lands_in_exactly_one_bin(edges):
    """No double counting and no silent loss: the counts plus the two
    out-of-range tallies must account for every crater."""
    d = power_law_sample(5000)
    d = np.concatenate([d, [D_MIN_M, edges[-1], 1.0, 1e9]])
    counts, below, above = sfd.bin_counts(d, edges)
    assert counts.sum() + below + above == d.size
    assert below == 1 and above == 2      # 1.0 m; edges[-1] and 1e9 m


def test_out_of_range_craters_are_reported_not_discarded(edges):
    counts, below, above = sfd.bin_counts(np.array([1.0, 5.0, 1e6]), edges)
    assert counts.sum() == 0 and below == 2 and above == 1


def test_bin_index_rejects_unsorted_edges():
    with pytest.raises(ValueError):
        sfd.bin_index(np.array([5.0]), np.array([10.0, 1.0, 20.0]))


# --------------------------------------------------------------------- #
# Units, by hand
# --------------------------------------------------------------------- #
def test_table_columns_are_exactly_the_documented_formulae():
    """One bin, all six quantities computed by hand in kilometres."""
    edges_m = np.array([1000.0, 2000.0])          # 1 km to 2 km
    d = np.array([1100.0, 1300.0, 1700.0, 1990.0])
    t = sfd.size_frequency_table(d, edges_m, 100.0, label="hand")

    assert t.d_low_km[0] == pytest.approx(1.0)
    assert t.d_high_km[0] == pytest.approx(2.0)
    assert t.d_g_km[0] == pytest.approx(np.sqrt(2.0), rel=1e-15)
    assert t.delta_d_km[0] == pytest.approx(1.0)
    assert t.n[0] == 4
    assert t.area_km2[0] == pytest.approx(100.0)
    assert t.r[0] == pytest.approx(np.sqrt(2.0) ** 3 * 4 / (100.0 * 1.0), rel=1e-14)
    assert t.r[0] == pytest.approx(0.11313708498984761, rel=1e-13)


def test_r_is_dimensionless_so_a_consistent_unit_rescale_leaves_it_alone():
    """R = D_g^3 N / (A dD) has units L^3 / (L^2 L). Scaling every length by f
    (and the area by f^2) must not move R -- the check that catches a metre
    multiplied by a kilometre, which is a factor of 1000 in R."""
    edges_m = np.array([100.0, 200.0, 400.0])
    d = np.array([120.0, 150.0, 180.0, 250.0, 390.0])
    base = sfd.size_frequency_table(d, edges_m, 10.0)
    f = 7.0
    scaled = sfd.size_frequency_table(d * f, edges_m * f, 10.0 * f**2)
    assert np.allclose(base.r, scaled.r, rtol=1e-13)
    assert np.allclose(base.n, scaled.n)


def test_metres_are_converted_to_kilometres_once():
    """Guard against the diameter being reported in metres: a 1 km bin must not
    appear as 1000."""
    t = sfd.size_frequency_table(np.array([1500.0]), np.array([1000.0, 2000.0]), 1.0)
    assert t.d_low_km[0] == pytest.approx(1.0)
    assert t.d_g_km[0] < 2.0


def test_area_must_be_scalar_or_one_value_per_bin(edges):
    d = power_law_sample(100)
    with pytest.raises(ValueError):
        sfd.size_frequency_table(d, edges, np.ones(3))
    with pytest.raises(ValueError):
        sfd.size_frequency_table(d, edges, -1.0)


def test_duplicate_crater_ids_are_rejected(edges):
    """N is a count of UNIQUE included craters; a duplicated detection inflates
    the density without changing the area."""
    d = power_law_sample(10)
    with pytest.raises(ValueError, match="UNIQUE"):
        sfd.size_frequency_table(d, edges, 1.0, crater_ids=["a"] * 10)
    sfd.size_frequency_table(d, edges, 1.0, crater_ids=list(range(10)))


def test_nonpositive_or_nonfinite_diameters_are_rejected(edges):
    with pytest.raises(ValueError):
        sfd.size_frequency_table(np.array([0.0, 50.0]), edges, 1.0)
    with pytest.raises(ValueError):
        sfd.size_frequency_table(np.array([np.nan]), edges, 1.0)


# --------------------------------------------------------------------- #
# Poisson counting uncertainty
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("n,low,high", [
    (0, 0.0, 3.6889),        # pure upper limit
    (1, 0.0253, 5.5716),
    (2, 0.2422, 7.2247),
    (3, 0.6187, 8.7673),
    (5, 1.6235, 11.6683),
    (10, 4.7954, 18.3904),
])
def test_garwood_interval_matches_published_values(n, low, high):
    """Textbook exact Poisson limits (chi-square inversion, 95 % two-sided)."""
    got_low, got_high, _ = sfd.poisson_interval(n, level=0.95)
    assert got_low == pytest.approx(low, abs=1e-4)
    assert got_high == pytest.approx(high, abs=1e-4)


def test_sqrt_n_would_be_badly_wrong_at_low_counts():
    """At N = 1 the normal approximation gives [0, 2]; the exact interval is
    [0.025, 5.57]. The upper bound is nearly three times larger -- which is the
    whole large-diameter end of a crater count."""
    low, high, _ = sfd.poisson_interval(1)
    assert low > 0.0
    assert high > 2.5 * (1 + np.sqrt(1))
    low0, high0, ul = sfd.poisson_interval(0)
    assert ul and low0 == 0.0 and high0 > 3.0   # sqrt(N) would give [0, 0]


def test_interval_brackets_the_count_and_widens_with_confidence():
    for n in (1, 4, 17, 100):
        low, high, _ = sfd.poisson_interval(n)
        assert low < n < high
        wide_low, wide_high, _ = sfd.poisson_interval(n, level=0.99)
        assert wide_low < low and wide_high > high


def test_interval_approaches_sqrt_n_at_large_counts():
    """The exact interval is not exotic: at N = 10000 it converges on the normal
    approximation, which is the sanity check that it is the same quantity."""
    n = 10_000
    low, high, _ = sfd.poisson_interval(n)
    assert low == pytest.approx(n - 1.96 * np.sqrt(n), rel=2e-3)
    assert high == pytest.approx(n + 1.96 * np.sqrt(n), rel=2e-3)


@pytest.mark.parametrize("lam", [0.3, 1.0, 3.0, 10.0, 50.0, 200.0])
def test_poisson_interval_covers_the_true_rate_at_the_nominal_level(lam):
    """Over 40000 synthetic draws the 95 % interval must contain the true rate
    about 95 % of the time. Garwood is conservative by construction (the Poisson
    distribution is discrete), so coverage is at least nominal and a little
    above -- but it must not be wildly above, which would mean uselessly wide
    error bars, nor below, which would mean overconfident ones."""
    rng = np.random.default_rng(424242)
    k = rng.poisson(lam, 40_000)
    low, high, _ = sfd.poisson_interval(k, level=0.95)
    coverage = float(np.mean((low <= lam) & (lam <= high)))
    assert coverage >= 0.95
    assert coverage <= 0.999


def test_poisson_interval_rejects_non_counts():
    with pytest.raises(ValueError):
        sfd.poisson_interval(-1)
    with pytest.raises(ValueError):
        sfd.poisson_interval(2.5)
    with pytest.raises(ValueError):
        sfd.poisson_interval(3, level=1.0)


# --------------------------------------------------------------------- #
# Zero-count bins
# --------------------------------------------------------------------- #
def test_zero_count_bin_is_a_flagged_upper_limit_not_a_fabricated_value():
    """R = 0 has no place on a log axis, and no epsilon may be inserted to give
    it one. The bin must come back as NaN + a flag + a finite upper limit."""
    edges_m = np.array([100.0, 200.0, 400.0, 800.0])
    d = np.array([120.0, 150.0])                  # nothing above 200 m
    t = sfd.size_frequency_table(d, edges_m, 50.0)

    assert t.n[1] == 0 and t.n[2] == 0
    assert np.isnan(t.r[1]) and np.isnan(t.r[2])
    assert t.flag[1] == sfd.FLAG_ZERO_COUNT
    assert bool(t.is_upper_limit[1]) and not bool(t.is_upper_limit[0])
    assert np.isfinite(t.r_poisson_high[1]) and t.r_poisson_high[1] > 0.0
    # Nothing anywhere is an infinity, and nothing was quietly set to zero.
    for col in (t.r, t.r_poisson_low, t.r_poisson_high):
        assert not np.any(np.isinf(col))
    assert t.r[0] > 0.0


def test_log10_of_a_zero_count_bin_is_nan_not_minus_inf():
    """The explicit reason for NaN over 0.0: a downstream log10 must propagate
    'unknown', not a -inf that silently rescales an axis."""
    t = sfd.size_frequency_table(
        np.array([120.0]), np.array([100.0, 200.0, 400.0]), 10.0
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        logged = np.log10(t.r)
    assert np.isnan(logged[1])
    assert not np.any(np.isneginf(logged))


def test_a_bin_with_no_counting_area_is_flagged_separately():
    """The edge rule can erode A_i to zero for the largest craters. That is a
    different statement from 'no craters found' and gets its own flag."""
    edges_m = np.array([100.0, 200.0, 400.0])
    d = np.array([120.0, 250.0])
    t = sfd.size_frequency_table(d, edges_m, np.array([50.0, 0.0]))
    assert t.flag[1] == sfd.FLAG_NO_AREA
    assert np.isnan(t.r[1]) and np.isnan(t.r_poisson_high[1])
    assert not bool(t.is_upper_limit[1])          # not an upper limit; unknown
    assert t.n[1] == 1                            # the crater was still counted


# --------------------------------------------------------------------- #
# The key scientific validation: a -3 slope gives a flat R-plot
# --------------------------------------------------------------------- #
def test_analytic_r_is_identically_constant_for_a_minus_three_slope(edges):
    """Before any sampling: the expected R of a D**-3 population is the same
    number in every constant-ratio bin. This is the identity the R-plot exists
    to exploit, so it is checked on its own."""
    n = 1_000_000
    exp_n = expected_counts(n, edges)[:N_FULL_BINS]
    lo_km, hi_km = edges[:-1] / 1000.0, edges[1:] / 1000.0
    d_g = np.sqrt(lo_km * hi_km)[:N_FULL_BINS]
    dd = (hi_km - lo_km)[:N_FULL_BINS]
    exp_r = d_g**3 * exp_n / (AREA_KM2 * dd)
    assert np.allclose(exp_r, exp_r[0], rtol=1e-12)

    # and it equals the closed form C/(2A) * (1+r)/sqrt(r)
    ratio = 2.0 ** (1.0 / BINS_PER_OCTAVE)
    c_m2 = 2.0 * n / (D_MIN_M**-2 - D_MAX_M**-2)   # dN/dD = C D^-3, C in m^2
    c_km2 = c_m2 / 1.0e6
    expected = c_km2 / (2.0 * AREA_KM2) * (1.0 + ratio) / np.sqrt(ratio)
    assert exp_r[0] == pytest.approx(expected, rel=1e-12)


def test_synthetic_minus_three_power_law_recovers_a_flat_rplot(edges):
    """THE validation. A population drawn from dN/dD ∝ D**-3, pushed through
    binning, areas and the R definition, must come back flat."""
    n = 1_000_000
    d = power_law_sample(n, slope=-3.0)
    t = sfd.size_frequency_table(d, edges, AREA_KM2, label="synthetic -3")

    full = slice(0, N_FULL_BINS)
    r = t.r[full]
    assert np.all(np.isfinite(r))

    exp_n = expected_counts(n, edges)[full]
    exp_r = t.d_g_km[full] ** 3 * exp_n / (AREA_KM2 * t.delta_d_km[full])
    assert np.allclose(exp_r, exp_r[0], rtol=1e-12)     # the target really is flat

    # 1. Bin by bin, within Poisson noise (sigma_R / R = 1/sqrt(expected N)).
    pulls = (r - exp_r) / (exp_r / np.sqrt(exp_n))
    assert np.max(np.abs(pulls)) < 4.0
    assert np.abs(np.mean(pulls)) < 0.6

    # 2. The departures from flat are counting noise and nothing else: the
    #    reduced chi-square against the constant analytic R is about 1.
    assert 0.4 < float(np.sum(pulls**2)) / len(r) < 2.0
    assert np.std(r) / np.mean(r) < 0.08               # bounded by that noise

    # 3. The level is right, not just the shape.
    assert np.average(r, weights=exp_n) == pytest.approx(exp_r[0], rel=0.01)

    # 4. A log-log fit has zero slope. This is the statement "flat R-plot".
    weights = np.sqrt(exp_n)
    slope = np.polyfit(np.log10(t.d_g_km[full]), np.log10(r), 1, w=weights)[0]
    assert slope == pytest.approx(0.0, abs=0.02)


def test_a_minus_four_slope_is_not_flat(edges):
    """Control: the flatness test above must be able to fail. A steeper
    differential slope must tilt R by one power of D per unit of slope."""
    n = 400_000
    d = power_law_sample(n, slope=-4.0, seed=99)
    t = sfd.size_frequency_table(d, edges, AREA_KM2)
    full = slice(0, N_FULL_BINS)
    ok = t.n[full] > 20
    slope = np.polyfit(
        np.log10(t.d_g_km[full][ok]), np.log10(t.r[full][ok]), 1, w=np.sqrt(t.n[full][ok])
    )[0]
    assert slope == pytest.approx(-1.0, abs=0.1)
    finite = t.r[full][np.isfinite(t.r[full])]
    assert np.std(finite) / np.mean(finite) > 0.5


def test_r_scales_inversely_with_the_counting_area(edges):
    """Halving the usable area doubles R: the area really is the denominator,
    not a decoration."""
    d = power_law_sample(20_000)
    t1 = sfd.size_frequency_table(d, edges, AREA_KM2)
    t2 = sfd.size_frequency_table(d, edges, AREA_KM2 / 2.0)
    good = np.isfinite(t1.r)
    assert np.allclose(t2.r[good], 2.0 * t1.r[good], rtol=1e-13)


def test_diameter_dependent_areas_from_the_area_module_feed_straight_in(edges):
    """End-to-end with crater.area: A_i from the edge rule, one value per bin,
    and R built on it."""
    colat = 0.5
    rho = 2.0 * a.MOON.radius_m * np.tan(np.radians(colat) / 2.0)
    grid = a.StereographicGrid.centred_on_projected_point(0.0, 0.0, rho * 1.1, 100.0)
    mask = a.mask_from_colatitude(grid, colat)

    rep = a.bin_representative_diameters_m(edges[:-1], edges[1:], "high")
    areas_km2 = a.usable_area_km2_by_diameter(grid, mask, rep)
    assert np.all(np.diff(areas_km2) <= 0.0)          # shrinks with D
    assert areas_km2[0] > areas_km2[-1]

    d = power_law_sample(50_000)
    t = sfd.size_frequency_table(d, edges, areas_km2, label="edge-rule areas")
    assert np.all(t.area_km2 == areas_km2)
    good = np.isfinite(t.r)
    assert np.all(t.r[good] > 0.0)


# --------------------------------------------------------------------- #
# Diameter measurement uncertainty, kept apart from Poisson
# --------------------------------------------------------------------- #
def test_monte_carlo_spread_is_reported_separately_from_poisson(edges):
    d = power_law_sample(20_000)
    unc = sfd.DiameterUncertainty(relative_sigma=0.10, n_draws=200, seed=5)
    t = sfd.size_frequency_table(d, edges, AREA_KM2, uncertainty=unc)

    assert t.has_monte_carlo
    poisson_width = t.r_poisson_high - t.r_poisson_low
    mc_width = t.r_mc_high - t.r_mc_low
    good = np.isfinite(t.r) & (t.n > 50)
    assert np.any(good)
    # Two genuinely different quantities, never one.
    assert not np.allclose(poisson_width[good], mc_width[good], rtol=0.2)
    quadrature = np.hypot(poisson_width, mc_width)
    for col in (poisson_width, mc_width):
        assert not np.allclose(col[good], quadrature[good], rtol=1e-6)

    cols = set(t.to_dataframe().columns)
    assert {"R_poisson_low", "R_poisson_high", "R_mc_p16", "R_mc_p84"} <= cols
    assert sfd.UNCERTAINTY_POLICY.endswith("separately")


def test_monte_carlo_moves_craters_between_bins(edges):
    """The point of propagating diameter error: with 25 % errors the count in a
    bin genuinely varies, and more so than with 2 % errors."""
    d = power_law_sample(20_000)
    small = sfd.monte_carlo_bin_counts(
        d, edges, sfd.DiameterUncertainty(relative_sigma=0.02, n_draws=120, seed=1)
    )
    large = sfd.monte_carlo_bin_counts(
        d, edges, sfd.DiameterUncertainty(relative_sigma=0.25, n_draws=120, seed=1)
    )
    busy = slice(0, N_FULL_BINS)
    assert np.mean(large.std(axis=0)[busy]) > 2.0 * np.mean(small.std(axis=0)[busy])


def test_zero_diameter_uncertainty_reproduces_the_plain_counts(edges):
    d = power_law_sample(5_000)
    unc = sfd.DiameterUncertainty(n_draws=5, seed=3)
    assert unc.is_zero
    draws = sfd.monte_carlo_bin_counts(d, edges, unc)
    counts, _, _ = sfd.bin_counts(d, edges)
    assert np.all(draws == counts)


def test_monte_carlo_is_reproducible_from_the_seed(edges):
    d = power_law_sample(2_000)
    kw = dict(relative_sigma=0.1, n_draws=30)
    one = sfd.monte_carlo_bin_counts(d, edges, sfd.DiameterUncertainty(seed=11, **kw))
    two = sfd.monte_carlo_bin_counts(d, edges, sfd.DiameterUncertainty(seed=11, **kw))
    other = sfd.monte_carlo_bin_counts(d, edges, sfd.DiameterUncertainty(seed=12, **kw))
    assert np.array_equal(one, two)
    assert not np.array_equal(one, other)


def test_perturbed_diameters_stay_positive_even_at_large_relative_error():
    """A Gaussian in D would put a 20 m crater with 30 % error at a negative
    diameter a few times in a thousand, and it could then never be binned."""
    rng = np.random.default_rng(0)
    unc = sfd.DiameterUncertainty(relative_sigma=0.30)
    d = np.full(200_000, 20.0)
    assert np.all(unc.draw(d, rng) > 0.0)


def test_absolute_and_relative_diameter_errors_combine_in_quadrature():
    unc = sfd.DiameterUncertainty(relative_sigma=0.1, absolute_sigma_m=4.0)
    s = unc.log_sigma(np.array([20.0, 100.0]))
    assert s[0] == pytest.approx(np.hypot(0.1, 4.0 / 20.0))
    assert s[1] == pytest.approx(np.hypot(0.1, 4.0 / 100.0))
    assert s[0] > s[1]                      # the floor hurts small craters most


def test_diameter_uncertainty_rejects_nonsense():
    with pytest.raises(ValueError):
        sfd.DiameterUncertainty(relative_sigma=-0.1)
    with pytest.raises(ValueError):
        sfd.DiameterUncertainty(n_draws=0)
    with pytest.raises(ValueError):
        sfd.DiameterUncertainty(percentiles=(84.0, 16.0))


# --------------------------------------------------------------------- #
# Comparing two series
# --------------------------------------------------------------------- #
def test_two_series_on_identical_bins(edges):
    d = power_law_sample(30_000)
    cat = sfd.size_frequency_table(d[d > 200.0], edges, AREA_KM2, label="catalogue")
    det = sfd.size_frequency_table(d, edges, AREA_KM2, label="detector")
    d_g = sfd.require_identical_bins(cat, det)
    assert np.allclose(d_g, cat.d_g_km)
    # Below the catalogue's completeness the detector finds more; above, they agree.
    high = cat.d_g_km > 0.3
    good = high & np.isfinite(cat.r) & np.isfinite(det.r)
    assert np.allclose(cat.r[good], det.r[good], rtol=1e-12)


def test_mismatched_bins_are_refused(edges):
    d = power_law_sample(1_000)
    other = sfd.log_bin_edges_m(20.0, 1280.0, bins_per_octave=3, anchor_m=20.0)
    t1 = sfd.size_frequency_table(d, edges, 1.0, label="a")
    t2 = sfd.size_frequency_table(d, other, 1.0, label="b")
    with pytest.raises(ValueError, match="bin edges"):
        sfd.require_identical_bins(t1, t2)


# --------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------- #
def test_csv_has_every_documented_column(tmp_path, edges):
    import pandas as pd

    d = power_law_sample(5_000)
    unc = sfd.DiameterUncertainty(relative_sigma=0.1, n_draws=40, seed=2)
    t1 = sfd.size_frequency_table(d, edges, AREA_KM2, label="detector", uncertainty=unc)
    t2 = sfd.size_frequency_table(d[d > 100], edges, AREA_KM2, label="catalogue",
                                  uncertainty=unc)
    path = sfd.write_diameter_bins_csv([t1, t2], str(tmp_path))
    assert os.path.basename(path) == "diameter_bins.csv"

    df = pd.read_csv(path)
    required = {"series", "D_low_km", "D_high_km", "D_g_km", "Delta_D_km", "N",
                "A_km2", "R", "R_poisson_low", "R_poisson_high", "is_upper_limit",
                "flag", "N_mc_mean", "N_mc_std", "R_mc_p16", "R_mc_p84"}
    assert required <= set(df.columns)
    assert set(df["series"]) == {"detector", "catalogue"}
    assert len(df) == 2 * t1.n_bins
    # R of a zero-count bin is empty, not 0 and not -inf.
    zero_rows = df[df["flag"] == sfd.FLAG_ZERO_COUNT]
    if len(zero_rows):
        assert zero_rows["R"].isna().all()
        assert (zero_rows["R_poisson_high"] > 0).all()


# --------------------------------------------------------------------- #
# The plot
# --------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def two_series(edges):
    d = power_law_sample(40_000)
    unc = sfd.DiameterUncertainty(relative_sigma=0.1, n_draws=40, seed=4)
    det = sfd.size_frequency_table(d, edges, AREA_KM2, label="detector", uncertainty=unc)
    cat = sfd.size_frequency_table(d[d > 150.0], edges, AREA_KM2, label="catalogue")
    return det, cat


def test_plot_writes_png_and_svg(tmp_path, two_series):
    res = sfd.plot_rplot(list(two_series), str(tmp_path), completeness_limit_m=40.0)
    assert os.path.basename(res.png_path) == "rplot.png"
    assert os.path.basename(res.svg_path) == "rplot.svg"
    assert os.path.getsize(res.png_path) > 10_000
    assert os.path.getsize(res.svg_path) > 10_000
    assert open(res.svg_path, encoding="utf-8").read(200).lstrip().startswith("<?xml")


def test_plot_does_not_double_log_the_data(tmp_path, two_series):
    """The guard. Both axes are logarithmic SCALES and the data handed to them
    are the RAW D_g and R values. Taking log10 first as well would plot
    log10(log10(R)) under tick labels that still claim to be R -- three decades
    compressed into half of one, and every fitted slope wrong."""
    det, cat = two_series
    res = sfd.plot_rplot([det, cat], str(tmp_path))

    assert res.axes.get_xscale() == "log"
    assert res.axes.get_yscale() == "log"

    for t in (det, cat):
        x, y = res.plotted[t.label]
        good = np.isfinite(t.r) & (t.r > 0.0)
        assert np.array_equal(y, t.r[good])          # raw R, bit for bit
        assert np.array_equal(x, t.d_g_km[good])     # raw D_g, bit for bit
        assert not np.allclose(y, np.log10(t.r[good]))
        assert np.all(y > 0.0)                       # a log10'd R would go negative

    # And the artists on the axes carry those same raw numbers.
    containers = {c.get_label(): c for c in res.axes.containers}
    for t in (det, cat):
        line = containers[t.label].lines[0]
        good = np.isfinite(t.r) & (t.r > 0.0)
        assert np.allclose(line.get_ydata(), t.r[good], rtol=0, atol=0)
        assert np.allclose(line.get_xdata(), t.d_g_km[good], rtol=0, atol=0)


def test_plot_never_receives_a_nan_zero_or_negative_value(tmp_path, two_series):
    """Nothing is pushed onto a log axis that cannot live there, and no epsilon
    was added to make one fit."""
    res = sfd.plot_rplot(list(two_series), str(tmp_path))
    for x, y in list(res.plotted.values()) + list(res.upper_limits.values()):
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        assert np.all(x > 0.0) and np.all(y > 0.0)


def test_plot_marks_zero_count_bins_as_upper_limits(tmp_path, edges):
    """An empty bin is drawn at its Poisson upper limit, as a downward caret --
    never as a point at R = 0, and never omitted without trace."""
    d = power_law_sample(300, slope=-3.0)
    d = d[d < 200.0]
    t = sfd.size_frequency_table(d, edges, AREA_KM2, label="sparse")
    assert np.any(t.is_upper_limit)

    res = sfd.plot_rplot(t, str(tmp_path))
    x_ul, y_ul = res.upper_limits["sparse"]
    assert len(x_ul) == int(np.count_nonzero(t.is_upper_limit))
    assert np.allclose(y_ul, t.r_poisson_high[t.is_upper_limit])
    # The upper-limit markers are a different artist from the measured series.
    labels = [c.get_label() for c in res.axes.containers]
    assert "sparse: upper limit (N = 0)" in labels
    assert "sparse" in labels


def test_plot_omits_bins_with_no_counting_area(tmp_path):
    edges_m = np.array([100.0, 200.0, 400.0])
    t = sfd.size_frequency_table(np.array([120.0, 250.0]), edges_m,
                                 np.array([50.0, 0.0]), label="eroded")
    res = sfd.plot_rplot(t, str(tmp_path))
    x, _ = res.plotted["eroded"]
    assert len(x) == 1                       # only the bin that has an area


def test_plot_marks_the_completeness_limit(tmp_path, two_series):
    res = sfd.plot_rplot(list(two_series), str(tmp_path), completeness_limit_m=40.0,
                         completeness_label="detector completeness")
    texts = [t.get_text() for t in res.axes.texts]
    assert "detector completeness" in texts
    xs = [ln.get_xdata()[0] for ln in res.axes.lines if len(set(ln.get_xdata())) == 1
          and len(ln.get_xdata()) == 2]
    assert any(np.isclose(x, 0.040) for x in xs)


def test_plot_legend_identifies_every_series(tmp_path, two_series):
    """Identity must not rest on colour alone: a legend is present and each
    series also carries its own marker shape."""
    res = sfd.plot_rplot(list(two_series), str(tmp_path))
    legend_labels = [t.get_text() for t in res.axes.get_legend().get_texts()]
    assert "detector" in legend_labels and "catalogue" in legend_labels
    markers = {c.lines[0].get_marker() for c in res.axes.containers
               if c.get_label() in ("detector", "catalogue")}
    assert len(markers) == 2


def test_plot_refuses_mismatched_bins(tmp_path, edges):
    d = power_law_sample(500)
    other = sfd.log_bin_edges_m(20.0, 1280.0, bins_per_octave=2, anchor_m=20.0)
    t1 = sfd.size_frequency_table(d, edges, 1.0, label="a")
    t2 = sfd.size_frequency_table(d, other, 1.0, label="b")
    with pytest.raises(ValueError):
        sfd.plot_rplot([t1, t2], str(tmp_path))


def test_plot_refuses_more_series_than_validated_colours(tmp_path, edges):
    d = power_law_sample(500)
    tables = [sfd.size_frequency_table(d, edges, 1.0, label=f"s{i}") for i in range(9)]
    with pytest.raises(ValueError, match="colours"):
        sfd.plot_rplot(tables, str(tmp_path))


def test_plot_writes_nothing_into_the_repository(tmp_path, two_series):
    """Plots go where the caller says; the module must not have a default path
    pointing at the working tree."""
    res = sfd.plot_rplot(list(two_series), str(tmp_path), basename="custom")
    assert os.path.dirname(res.png_path) == str(tmp_path)
    assert os.path.basename(res.png_path) == "custom.png"
