"""Lunar map geometry: polar stereographic projection and spherical geodesics.

The projection is implemented analytically here rather than only delegating to
PROJ, so that the test suite can cross-validate two independent
implementations.  Agreement between this module and ``pyproj`` is a real check;
a single wrapper around PROJ would only test PROJ against itself.

Conventions (fixed project-wide, see ``config/project.yaml``)
------------------------------------------------------------
* Body            : Moon 2000 sphere, R = 1737400 m.
* Latitude        : planetocentric, degrees, south negative.
* Longitude       : positive east, canonical domain [-180, 180).
* Projection      : south polar stereographic, lat_0 = lat_ts = -90, k0 = 1.
* Distances       : all "ground" distances are great-circle (planimetric on the
  sphere).  They are NOT terrain-surface distances; slope lengthening is not
  applied anywhere in this module.  Callers reporting a diameter must say which
  of the two they mean (see ``DIAMETER_DISTANCE_KIND``).
"""
from __future__ import annotations

import numpy as np

from .body import MOON, Body

#: Every distance produced by this module is planimetric on the sphere.
DIAMETER_DISTANCE_KIND = "planimetric_great_circle"

_EPS = 1e-12


# --------------------------------------------------------------------------- #
# Longitude handling
# --------------------------------------------------------------------------- #
def wrap_longitude(lon_deg):
    """Wrap longitude into the canonical domain [-180, 180).

    Works for scalars and arrays, and for values many turns outside the domain.
    """
    lon = np.asarray(lon_deg, dtype=float)
    out = (lon + 180.0) % 360.0 - 180.0
    # Floating-point modulo can return exactly 360.0 for a tiny negative
    # operand, yielding +180.0 and breaking the half-open domain.  Fold only
    # that exact case.  A tolerant comparison (e.g. np.isclose, whose default
    # rtol=1e-5 spans 1.8e-3 deg) would snap genuine longitudes near the
    # antimeridian onto it -- about 3.8 m of ground error at 86 S, which is a
    # fifth of the smallest crater this project measures.
    out = np.where(out >= 180.0, out - 360.0, out)
    return out if np.ndim(lon_deg) else float(out)


def longitude_difference(lon_a, lon_b):
    """Signed smallest difference ``lon_a - lon_b`` in degrees, in [-180, 180)."""
    return wrap_longitude(np.asarray(lon_a, float) - np.asarray(lon_b, float))


# --------------------------------------------------------------------------- #
# South polar stereographic
# --------------------------------------------------------------------------- #
def forward(lon_deg, lat_deg, body: Body = MOON, lon_0: float = 0.0, k0: float = 1.0):
    """Project lon/lat to south polar stereographic x/y in metres.

    Raises
    ------
    ValueError
        If any latitude is at or north of the antipodal point (+90), where the
        south polar stereographic projection is undefined (infinite radius).
    """
    lon = np.asarray(lon_deg, dtype=float)
    lat = np.asarray(lat_deg, dtype=float)
    if np.any(lat > 90.0 + 1e-9) or np.any(lat < -90.0 - 1e-9):
        raise ValueError("latitude outside [-90, 90]")
    if np.any(lat >= 90.0 - 1e-9):
        raise ValueError(
            "south polar stereographic is singular at the north pole; "
            "latitude +90 cannot be projected"
        )
    dlon = np.radians(wrap_longitude(lon - lon_0))
    phi = np.radians(lat)
    # rho = 2 a k0 tan(pi/4 + phi/2); zero at the south pole.
    rho = 2.0 * body.radius_m * k0 * np.tan(np.pi / 4.0 + phi / 2.0)
    x = rho * np.sin(dlon)
    y = rho * np.cos(dlon)
    if np.ndim(lon_deg) or np.ndim(lat_deg):
        return x, y
    return float(x), float(y)


def inverse(x_m, y_m, body: Body = MOON, lon_0: float = 0.0, k0: float = 1.0):
    """Unproject south polar stereographic x/y in metres to lon/lat degrees.

    At the pole itself (x = y = 0) longitude is genuinely undefined; ``lon_0``
    is returned, which is the conventional choice.
    """
    x = np.asarray(x_m, dtype=float)
    y = np.asarray(y_m, dtype=float)
    rho = np.hypot(x, y)
    c = 2.0 * np.arctan2(rho, 2.0 * body.radius_m * k0)
    lat = np.degrees(c - np.pi / 2.0)
    # atan2(0, 0) is 0, which yields lon_0 at the pole -- the conventional choice.
    lon = wrap_longitude(lon_0 + np.degrees(np.arctan2(x, y)))
    at_pole = rho < _EPS
    lon = np.where(at_pole, wrap_longitude(lon_0), lon)
    if np.ndim(x_m) or np.ndim(y_m):
        return lon, lat
    return float(lon), float(lat)


def point_scale_factor(lat_deg, k0: float = 1.0):
    """Linear scale factor k of the south polar stereographic projection.

    Stereographic projection is conformal, so k is identical along the meridian
    and the parallel::

        k(phi) = 2 k0 / (1 - sin phi)

    k = k0 exactly at the south pole and grows towards the equator.  One
    projected metre corresponds to ``1/k`` metres on the sphere, so a length
    measured in projected coordinates must be divided by k to become a ground
    distance.
    """
    lat = np.asarray(lat_deg, dtype=float)
    k = 2.0 * k0 / (1.0 - np.sin(np.radians(lat)))
    return k if np.ndim(lat_deg) else float(k)


def area_scale_factor(lat_deg, k0: float = 1.0):
    """Areal scale factor, ``k**2`` for a conformal projection.

    A projected area must be divided by this to become true surface area.
    """
    k = point_scale_factor(lat_deg, k0=k0)
    return k * k


# --------------------------------------------------------------------------- #
# Spherical geodesics
# --------------------------------------------------------------------------- #
def great_circle_distance(lon1, lat1, lon2, lat2, body: Body = MOON):
    """Great-circle (planimetric) distance in metres on the sphere.

    Uses the haversine form, which stays accurate for the small separations
    that dominate crater work.
    """
    p1 = np.radians(np.asarray(lat1, float))
    p2 = np.radians(np.asarray(lat2, float))
    dphi = p2 - p1
    dlam = np.radians(longitude_difference(lon2, lon1))
    a = np.sin(dphi / 2.0) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dlam / 2.0) ** 2
    d = 2.0 * body.radius_m * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return d if np.ndim(d) else float(d)


def geodesic_destination(lon_deg, lat_deg, bearing_deg, distance_m, body: Body = MOON):
    """Point at ``distance_m`` along ``bearing_deg`` from a start point.

    Bearing is measured clockwise from north.  Distance is a great-circle arc
    length on the sphere, so this is the correct way to place a crater rim from
    a catalogue centre and diameter -- never by converting degrees of longitude
    to metres with a planar factor.
    """
    phi1 = np.radians(np.asarray(lat_deg, float))
    lam1 = np.radians(np.asarray(lon_deg, float))
    theta = np.radians(np.asarray(bearing_deg, float))
    delta = np.asarray(distance_m, float) / body.radius_m  # angular distance

    sin_phi2 = np.sin(phi1) * np.cos(delta) + np.cos(phi1) * np.sin(delta) * np.cos(theta)
    phi2 = np.arcsin(np.clip(sin_phi2, -1.0, 1.0))
    lam2 = lam1 + np.arctan2(
        np.sin(theta) * np.sin(delta) * np.cos(phi1),
        np.cos(delta) - np.sin(phi1) * sin_phi2,
    )
    lat2 = np.degrees(phi2)
    lon2 = wrap_longitude(np.degrees(lam2))
    scalar = not (np.ndim(lon_deg) or np.ndim(lat_deg)
                  or np.ndim(bearing_deg) or np.ndim(distance_m))
    return (float(lon2), float(lat2)) if scalar else (lon2, lat2)


def crater_rim_samples(lon_deg, lat_deg, diameter_m, n_samples: int = 72,
                       body: Body = MOON):
    """Sample the rim of a circular crater as geodesic points.

    The crater is modelled as a circle of constant *ground* radius
    ``diameter_m / 2`` about its centre, which is what a catalogue
    centre+diameter entry means.  The returned geometry is therefore a
    catalogue-derived approximation, not an observed rim trace.

    Returns
    -------
    (lon, lat) : arrays of length ``n_samples``
    """
    if n_samples < 3:
        raise ValueError("n_samples must be at least 3 to describe a rim")
    if np.any(np.asarray(diameter_m, float) <= 0):
        raise ValueError("diameter_m must be positive")
    bearings = np.linspace(0.0, 360.0, n_samples, endpoint=False)
    return geodesic_destination(
        np.full(n_samples, float(lon_deg)),
        np.full(n_samples, float(lat_deg)),
        bearings,
        np.full(n_samples, float(diameter_m) / 2.0),
        body=body,
    )
