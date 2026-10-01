"""Lunar body constants and coordinate reference systems.

All geometry in this project is defined on the Moon 2000 sphere.  No Earth
EPSG code is ever assigned to lunar data (DECISIONS.md D-006).

References
----------
Moon 2000 / IAU-IAG mean radius 1737400 m, sphere (flattening 0).
Confirmed in this project against the USGS-supplied projection file
``wms_basemaps/Moon/Moon 2000.prj``, read 2026-10-01::

    GEOGCS["GCS_Moon_2000",DATUM["D_Moon_2000",
      SPHEROID["Moon_2000_IAU_IAG",1737400.0,0.0]], ...]
"""
from __future__ import annotations

from dataclasses import dataclass

#: Mean radius of the Moon 2000 sphere, metres.
MOON_RADIUS_M: float = 1737400.0

#: The body is treated as a sphere; flattening is exactly zero.
MOON_FLATTENING: float = 0.0


@dataclass(frozen=True)
class Body:
    """A spherical target body."""

    name: str
    radius_m: float
    flattening: float = 0.0

    def __post_init__(self) -> None:
        if self.radius_m <= 0:
            raise ValueError(f"radius_m must be positive, got {self.radius_m}")
        if self.flattening != 0.0:
            raise NotImplementedError(
                "Only spherical bodies are supported; the Moon 2000 datum is a "
                "sphere. An ellipsoidal body would change every geodesic and "
                "scale-factor formula in geometry.py."
            )

    @property
    def geographic_proj4(self) -> str:
        """PROJ string for the body's geographic (lon/lat) CRS."""
        return (
            f"+proj=longlat +a={self.radius_m:.1f} +b={self.radius_m:.1f} +no_defs"
        )

    def polar_stereographic_proj4(self, south: bool = True, lon_0: float = 0.0) -> str:
        """PROJ string for a polar stereographic CRS on this body.

        The standard parallel is placed at the pole (``lat_ts = +/-90``) so the
        scale factor is exactly 1 at the pole and grows away from it; see
        :func:`crater.geometry.point_scale_factor`.
        """
        lat_0 = -90.0 if south else 90.0
        return (
            f"+proj=stere +lat_0={lat_0:.1f} +lat_ts={lat_0:.1f} "
            f"+lon_0={lon_0:g} +k=1 +x_0=0 +y_0=0 "
            f"+a={self.radius_m:.1f} +b={self.radius_m:.1f} +units=m +no_defs"
        )


#: The project's single body definition.
MOON = Body(name="Moon_2000", radius_m=MOON_RADIUS_M, flattening=MOON_FLATTENING)

#: Latitude convention used throughout: planetocentric.
LATITUDE_CONVENTION = "planetocentric"
#: Longitude convention: positive east, domain [-180, 180).
LONGITUDE_DIRECTION = "east"
LONGITUDE_DOMAIN = (-180.0, 180.0)
