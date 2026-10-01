"""Tests for lunar projection and geodesic geometry.

The PROJ cross-checks matter: geometry.py implements the projection
analytically, so agreement with pyproj compares two independent
implementations rather than testing PROJ against itself.
"""
import numpy as np
import pytest
from pyproj import CRS, Transformer

from crater import geometry as g
from crater.body import MOON

# Points chosen to cover the ROI latitude, the pole, the antimeridian and
# mid-latitudes where projection distortion is large.
SAMPLE_POINTS = [
    (2.9, -85.9), (-3.6, -85.8), (0.0, -90.0), (120.0, -70.0),
    (-179.9, -89.99), (45.0, -30.0), (179.999, -86.0), (-0.001, -86.0),
]


@pytest.fixture(scope="module")
def proj_fwd():
    return Transformer.from_crs(
        CRS.from_proj4(MOON.geographic_proj4),
        CRS.from_proj4(MOON.polar_stereographic_proj4()),
        always_xy=True,
    )


# --------------------------------------------------------------------- #
# Longitude convention
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("raw,expected", [
    (0.0, 0.0), (180.0, -180.0), (-180.0, -180.0), (190.0, -170.0),
    (360.0, 0.0), (-360.0, 0.0), (540.0, -180.0), (359.9, -0.1),
    (-190.0, 170.0), (720.5, 0.5),
])
def test_wrap_longitude_canonical_domain(raw, expected):
    assert g.wrap_longitude(raw) == pytest.approx(expected, abs=1e-9)


def test_wrap_longitude_is_half_open():
    """Domain is [-180, 180): +180 must fold to -180 so binning is unambiguous."""
    many = g.wrap_longitude(np.linspace(-1000, 1000, 4001))
    assert np.all(many >= -180.0) and np.all(many < 180.0)


def test_longitude_difference_across_antimeridian():
    # 179.9E to -179.9E is 0.2 deg apart, not 359.8.
    assert g.longitude_difference(-179.9, 179.9) == pytest.approx(0.2, abs=1e-9)
    assert g.longitude_difference(179.9, -179.9) == pytest.approx(-0.2, abs=1e-9)


# --------------------------------------------------------------------- #
# Projection correctness
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("lon,lat", SAMPLE_POINTS)
def test_forward_matches_proj(lon, lat, proj_fwd):
    mx, my = g.forward(lon, lat)
    px, py = proj_fwd.transform(lon, lat)
    assert mx == pytest.approx(px, abs=1e-6)
    assert my == pytest.approx(py, abs=1e-6)


@pytest.mark.parametrize("lon,lat", SAMPLE_POINTS)
def test_roundtrip_lonlat_to_xy(lon, lat):
    x, y = g.forward(lon, lat)
    lon2, lat2 = g.inverse(x, y)
    assert lat2 == pytest.approx(lat, abs=1e-9)
    if lat > -89.9999:  # longitude is undefined at the pole
        assert g.longitude_difference(lon2, lon) == pytest.approx(0.0, abs=1e-9)


def test_roundtrip_xy_to_lonlat_dense():
    rng = np.random.default_rng(20261001)
    x = rng.uniform(-3e5, 3e5, 2000)
    y = rng.uniform(-3e5, 3e5, 2000)
    lon, lat = g.inverse(x, y)
    x2, y2 = g.forward(lon, lat)
    assert np.max(np.abs(x2 - x)) < 1e-6
    assert np.max(np.abs(y2 - y)) < 1e-6


def test_south_pole_maps_to_origin():
    for lon in (-180.0, -90.0, 0.0, 90.0, 179.0):
        x, y = g.forward(lon, -90.0)
        assert (x, y) == pytest.approx((0.0, 0.0), abs=1e-9)


def test_pole_inverse_longitude_is_conventional_not_nan():
    lon, lat = g.inverse(0.0, 0.0)
    assert lat == pytest.approx(-90.0, abs=1e-12)
    assert np.isfinite(lon)


def test_north_pole_is_rejected_not_silently_infinite():
    with pytest.raises(ValueError, match="singular"):
        g.forward(0.0, 90.0)


def test_latitude_out_of_range_rejected():
    with pytest.raises(ValueError, match="outside"):
        g.forward(0.0, -90.5)


def test_longitude_origin_shift_rotates_consistently():
    """A lon_0 shift must rotate the plane, not change distances."""
    a = g.forward(10.0, -86.0)
    b = g.forward(10.0, -86.0, lon_0=10.0)
    assert np.hypot(*a) == pytest.approx(np.hypot(*b), rel=1e-12)
    assert b[0] == pytest.approx(0.0, abs=1e-6)  # on the central meridian


# --------------------------------------------------------------------- #
# Scale factor
# --------------------------------------------------------------------- #
def test_scale_factor_is_unity_at_pole():
    assert g.point_scale_factor(-90.0) == pytest.approx(1.0, abs=1e-12)


def test_scale_factor_matches_numerical_derivative():
    """k must equal d(rho)/(R d(phi)) -- derived independently by differencing."""
    for lat in (-89.0, -86.0, -80.0, -60.0):
        h = 1e-6
        r1 = np.hypot(*g.forward(0.0, lat - h))
        r2 = np.hypot(*g.forward(0.0, lat + h))
        numeric = (r2 - r1) / (np.radians(2 * h) * MOON.radius_m)
        assert g.point_scale_factor(lat) == pytest.approx(numeric, rel=1e-6)


def test_scale_factor_conformal_meridian_equals_parallel():
    """Stereographic is conformal: parallel scale rho/(R cos phi) == k."""
    for lat in (-89.0, -85.9, -70.0, -45.0):
        rho = np.hypot(*g.forward(0.0, lat))
        parallel = rho / (MOON.radius_m * np.cos(np.radians(lat)))
        assert parallel == pytest.approx(g.point_scale_factor(lat), rel=1e-12)


def test_scale_distortion_at_roi_is_small_but_real():
    """At 85.9S distortion is ~0.12% -- small, yet not negligible for areas."""
    k = g.point_scale_factor(-85.9)
    assert 1.0009 < k < 1.0015
    assert g.area_scale_factor(-85.9) == pytest.approx(k * k, rel=1e-12)


# --------------------------------------------------------------------- #
# Geodesics and synthetic craters
# --------------------------------------------------------------------- #
@pytest.mark.parametrize("diameter_m", [20.0, 50.0, 137.0, 500.0, 1000.0])
@pytest.mark.parametrize("centre", [(2.9, -85.9), (0.0, -89.5), (120.0, -70.0)])
def test_synthetic_crater_rim_has_the_requested_diameter(diameter_m, centre):
    """A rim sampled from centre+diameter must measure back to that diameter."""
    lon0, lat0 = centre
    lon, lat = g.crater_rim_samples(lon0, lat0, diameter_m, n_samples=180)
    radii = g.great_circle_distance(lon0, lat0, lon, lat)
    # 0.1 mm: far below any achievable measurement, but above the float64
    # noise floor of the geodesic/haversine round trip (~5e-7 m near the pole).
    assert np.allclose(radii, diameter_m / 2.0, atol=1e-4)
    # Opposite rim points are one full diameter apart.
    n = len(lon)
    across = g.great_circle_distance(lon[: n // 2], lat[: n // 2],
                                     lon[n // 2:], lat[n // 2:])
    assert np.allclose(across, diameter_m, rtol=1e-6)


def test_rim_crossing_the_pole_stays_finite_and_correct():
    """A crater centred 200 m from the pole has rim points on both sides."""
    lon, lat = g.crater_rim_samples(0.0, -89.995, 1000.0, n_samples=72)
    assert np.all(np.isfinite(lon)) and np.all(np.isfinite(lat))
    assert np.all(lat <= -89.0)
    radii = g.great_circle_distance(0.0, -89.995, lon, lat)
    assert np.allclose(radii, 500.0, atol=1e-4)
    # The rim must genuinely wrap in longitude, not collapse to one side.
    assert np.ptp(g.wrap_longitude(lon)) > 180.0


def test_great_circle_distance_known_values():
    # Quarter circumference between pole and equator.
    d = g.great_circle_distance(0.0, -90.0, 0.0, 0.0)
    assert d == pytest.approx(np.pi / 2 * MOON.radius_m, rel=1e-12)
    # Antipodal points.
    d2 = g.great_circle_distance(0.0, 0.0, 180.0, 0.0)
    assert d2 == pytest.approx(np.pi * MOON.radius_m, rel=1e-9)
    # Identical points.
    assert g.great_circle_distance(2.9, -85.9, 2.9, -85.9) == pytest.approx(0.0, abs=1e-9)


def test_earth_radius_would_fail_this_test():
    """Guard against an Earth radius slipping in: 1 deg of latitude on the Moon
    is ~30.3 km, not ~111 km."""
    d = g.great_circle_distance(0.0, -86.0, 0.0, -85.0)
    assert d == pytest.approx(30325.0, abs=50.0)


def test_longitude_degrees_are_not_metres_near_the_pole():
    """At 86S one degree of longitude is ~2.1 km, not ~30 km. A planar
    deg->m conversion would be wrong by a factor of ~14 here."""
    d = g.great_circle_distance(0.0, -86.0, 1.0, -86.0)
    assert d == pytest.approx(2115.0, abs=30.0)
    equator = g.great_circle_distance(0.0, 0.0, 1.0, 0.0)
    assert equator / d == pytest.approx(1.0 / np.cos(np.radians(86.0)), rel=1e-3)


def test_rim_sampling_rejects_bad_input():
    with pytest.raises(ValueError):
        g.crater_rim_samples(0.0, -86.0, 100.0, n_samples=2)
    with pytest.raises(ValueError):
        g.crater_rim_samples(0.0, -86.0, -5.0)


def test_projected_rim_is_nearly_circular_at_roi():
    """Under a conformal projection a small ground circle stays near-circular;
    its projected radius is inflated by k, which is how pixel sizes arise."""
    lon0, lat0, D = 2.9, -85.9, 500.0
    lon, lat = g.crater_rim_samples(lon0, lat0, D, n_samples=360)
    x, y = g.forward(lon, lat)
    cx, cy = g.forward(lon0, lat0)
    r = np.hypot(x - cx, y - cy)
    assert (r.max() - r.min()) / r.mean() < 1e-3          # near-circular
    expected = (D / 2.0) * g.point_scale_factor(lat0)      # inflated by k
    assert r.mean() == pytest.approx(expected, rel=2e-3)


def test_longitudes_near_the_antimeridian_are_not_snapped():
    """Regression: a tolerant fold of +180 silently moved points near the
    antimeridian by up to 1.8e-3 deg (~3.8 m of ground at 86 S)."""
    for lon in (179.999, 179.9999, -179.999, 179.99):
        assert g.wrap_longitude(lon) == pytest.approx(lon, abs=1e-12)
    # and the fold of exactly +180 still holds
    assert g.wrap_longitude(180.0) == -180.0


def test_antimeridian_neighbours_stay_metres_apart_not_collapsed():
    d = g.great_circle_distance(179.999, -86.0, 180.0, -86.0)
    assert 0.0 < d < 10.0
    assert d == pytest.approx(2.1, abs=0.5)
