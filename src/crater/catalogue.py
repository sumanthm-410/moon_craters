"""The Robbins (2019) lunar crater catalogue: load, normalise, select, annotate.

Scope
-----
This module turns the archived Robbins lunar crater table into project-native
quantities and nothing more.  It does **not** decide whether the catalogue is
fit for a given purpose; ``reports/robbins_audit.md`` does that, and the answer
for this project's 20-1000 m range is mostly "no".

Provenance of every convention stated below
-------------------------------------------
All of it comes from the delivered archive, not from memory:

* PDS4 label ``data/lunar_crater_database_robbins_2018.xml`` of bundle
  ``urn:nasa:pds:robbins_lunar_crater_database_2018``
  (``<cart:Geodetic_Model>``):

  - ``latitude_type`` = **Planetocentric**
  - ``spheroid_name`` = **Moon_2000**, ``semi_major_radius`` =
    ``semi_minor_radius`` = ``polar_radius`` = **1737400 m**
  - ``longitude_direction`` = **Positive East**
  - ``<cart:Bounding_Coordinates>`` west ``0.000517267`` deg, east ``360`` deg
    -- i.e. the longitude **domain is 0..360**, not -180..180.
  - ``<Table_Delimited>``: 21 comma-delimited fields, CRLF records,
    ``records`` = 1296796, header occupying the first 361 bytes.

  The label gives **no ``unit`` attribute on any field**.  That is a real gap
  in the archive, so the units are stated here together with how they were
  established (see :data:`ROBBINS_FIELDS` and
  :data:`DIAMETER_UNIT_EVIDENCE`), never asserted bare.

* ``document/lunar_crater_database_archive_description.pdf``: "approximately
  complete for all craters larger than about 1-2 km in diameter"; "The exact
  completeness point varies based on location (it is complete to smaller
  diameters in lunar maria ...)"; rims were traced on WAC (70-100 m/pix), TC
  (30 m/pix) and hillshade DTMs (5-60 m/pix) and "Circles and ellipses were fit
  to the rim points", using "Great Circle distances and bearings".

Project conventions applied here
--------------------------------
The project uses east-positive longitude in ``[-180, 180)`` and planetocentric
latitude on the same Moon 2000 sphere (``config/project.yaml``), so the only
coordinate transformation needed is the longitude domain change, which is done
with :func:`crater.geometry.wrap_longitude` and nothing else.  Diameters are
converted km -> m by an exact factor of 1000.

Selection is done by **projecting crater centres** with
:func:`crater.geometry.forward` and testing the projected box.  A degree box is
not equivalent near the pole -- at 86 S one degree of longitude is about 2.1 km
on the ground while one degree of latitude is about 30.3 km, and the ROI's
projected square maps to a curved quadrilateral in lon/lat.
:func:`select_in_degree_box` exists only to demonstrate that difference and is
labelled wrong in its own docstring.

Honesty rules (INTERFACES.md rule 6)
------------------------------------
* A field that the archive leaves empty stays ``NaN`` in a frame and
  :data:`crater.boxes.UNMEASURED` in a record.  It is never filled, imputed or
  back-derived.
* Rim geometry generated from a catalogue centre and diameter is a *circle of
  assumed radius*, not an observed rim.  Every generated annotation carries
  ``source_label_type="catalogue"`` and ``review_status="unreviewed"``
  (``docs/annotation_guide.md``) and
  ``geometry_kind="catalogue_circle_approximation"``.
* A row that cannot be parsed raises :class:`MalformedCatalogueRow`; it is
  never dropped silently and never coerced to a plausible number.
"""
from __future__ import annotations

from dataclasses import dataclass, field as _dc_field
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np
import pandas as pd

from . import geometry as g
from .body import MOON, Body
from .boxes import UNMEASURED, Unmeasured

__all__ = [
    "CatalogueError",
    "MalformedCatalogueRow",
    "SchemaMismatch",
    "FieldSpec",
    "ROBBINS_FIELDS",
    "ROBBINS_FIELD_NAMES",
    "ROBBINS_DTYPES",
    "DIAMETER_COLUMNS",
    "DIAMETER_UNIT_EVIDENCE",
    "DEFAULT_DIAMETER_COLUMN",
    "DEFAULT_LAT_COLUMN",
    "DEFAULT_LON_COLUMN",
    "SOURCE_LONGITUDE_DOMAIN",
    "SOURCE_LATITUDE_TYPE",
    "SOURCE_LONGITUDE_DIRECTION",
    "SOURCE_DIAMETER_UNIT",
    "KM_TO_M",
    "DECLARED_RECORD_COUNT",
    "COMPLETENESS_LIMIT_M",
    "APPROVED_DIAMETER_RANGE_M",
    "SOURCE_LABEL_TYPE",
    "REVIEW_STATUS_UNREVIEWED",
    "GEOMETRY_KIND",
    "load_robbins_csv",
    "normalise_catalogue",
    "equivalent_ellipse_diameter_m",
    "ProjectedBox",
    "project_centres",
    "select_in_projected_box",
    "select_in_degree_box",
    "CatalogueRimAnnotation",
    "catalogue_rim_annotations",
    "rim_annotation_frame",
    "diameter_histogram",
]


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #
class CatalogueError(Exception):
    """Base class for catalogue problems."""


class MalformedCatalogueRow(CatalogueError):
    """A record could not be parsed under the declared schema.

    Carries the offending row numbers and values so the caller can inspect the
    source rather than guess.  Raised instead of coercing to ``NaN``: a value
    that was *present but unparseable* is a data integrity problem, which is
    categorically different from a value the archive left blank.
    """

    def __init__(self, column: str, rows: Sequence[int], values: Sequence[str]) -> None:
        shown = ", ".join(f"row {r}: {v!r}" for r, v in zip(rows[:5], values[:5]))
        more = "" if len(rows) <= 5 else f" (+{len(rows) - 5} more)"
        super().__init__(f"column {column!r} has unparseable values -- {shown}{more}")
        self.column = column
        self.rows = tuple(rows)
        self.values = tuple(values)


class SchemaMismatch(CatalogueError):
    """The file's header does not match the schema declared by the PDS4 label."""


# --------------------------------------------------------------------------- #
# The real schema, transcribed from the delivered PDS4 label
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FieldSpec:
    """One catalogue column exactly as the archive delivers it.

    Attributes
    ----------
    name:
        Column name as it appears in the CSV header (first 361 bytes).
    pds4_data_type:
        ``data_type`` from the label's ``<Field_Delimited>``.
    dtype:
        The numpy/pandas dtype this module parses the column into.
    unit:
        Physical unit, or ``"1"`` for dimensionless, or ``None`` when the
        quantity has no unit (an identifier).  **The label declares no units**;
        see :data:`DIAMETER_UNIT_EVIDENCE` for how these were established.
    unit_declared_by_label:
        ``False`` for every field in this product.  Kept explicit so no caller
        can mistake a transcribed unit for an archived one.
    description:
        What the column holds.  For the diameter family this states the fit
        definition, because the catalogue carries several different ones.
    """

    name: str
    pds4_data_type: str
    dtype: str
    unit: str | None
    description: str
    unit_declared_by_label: bool = False


#: Transcribed one-to-one from ``<Record_Delimited>`` of the delivered label,
#: in label field order.  The ``PTS_RIM_IMG`` note records a label/data
#: disagreement found on inspection and is not silently reconciled.
ROBBINS_FIELDS: tuple[FieldSpec, ...] = (
    FieldSpec("CRATER_ID", "UTF8_String", "string", None,
              "Crater identifier, e.g. '00-1-000000'. Encodes the search "
              "sub-region and a sequence number; opaque to this project."),
    FieldSpec("LAT_CIRC_IMG", "ASCII_Real", "float64", "deg",
              "Planetocentric latitude of the centre of the CIRCLE fitted to "
              "the traced rim points. North positive."),
    FieldSpec("LON_CIRC_IMG", "ASCII_Real", "float64", "deg",
              "East-positive longitude, domain 0..360, of the centre of the "
              "CIRCLE fit."),
    FieldSpec("LAT_ELLI_IMG", "ASCII_Real", "float64", "deg",
              "Planetocentric latitude of the centre of the ELLIPSE fit."),
    FieldSpec("LON_ELLI_IMG", "ASCII_Real", "float64", "deg",
              "East-positive longitude, domain 0..360, of the centre of the "
              "ELLIPSE fit."),
    FieldSpec("DIAM_CIRC_IMG", "ASCII_Real", "float64", "km",
              "DIAMETER of the fitted CIRCLE: the single best-fit circular "
              "diameter. This is the column the Robbins papers use for "
              "size-frequency work and the project default."),
    FieldSpec("DIAM_CIRC_SD_IMG", "ASCII_Real", "float64", "km",
              "Standard deviation of the circle-fit diameter: a fit "
              "uncertainty, NOT a second diameter."),
    FieldSpec("DIAM_ELLI_MAJOR_IMG", "ASCII_Real", "float64", "km",
              "MAJOR AXIS LENGTH of the fitted ELLIPSE (a full axis, i.e. a "
              "diameter, not a semi-axis -- verified against "
              "DIAM_ELLI_ELLIP_IMG = major/minor)."),
    FieldSpec("DIAM_ELLI_MINOR_IMG", "ASCII_Real", "float64", "km",
              "MINOR AXIS LENGTH of the fitted ELLIPSE (a full axis)."),
    FieldSpec("DIAM_ELLI_ECCEN_IMG", "ASCII_Real", "float64", "1",
              "Eccentricity of the fitted ellipse, sqrt(1 - (minor/major)^2). "
              "Dimensionless; despite the DIAM_ prefix this is NOT a length."),
    FieldSpec("DIAM_ELLI_ELLIP_IMG", "ASCII_Real", "float64", "1",
              "Ellipticity of the fitted ellipse, major/minor. Dimensionless; "
              "despite the DIAM_ prefix this is NOT a length."),
    FieldSpec("DIAM_ELLI_ANGLE_IMG", "ASCII_Real", "float64", "deg",
              "Azimuth of the ellipse major axis. An angle, not a length."),
    FieldSpec("LAT_ELLI_SD_IMG", "ASCII_Real", "float64", "deg",
              "Standard deviation of the ellipse-fit centre latitude."),
    FieldSpec("LON_ELLI_SD_IMG", "ASCII_Real", "float64", "deg",
              "Standard deviation of the ellipse-fit centre longitude."),
    FieldSpec("DIAM_ELLI_MAJOR_SD_IMG", "ASCII_Real", "float64", "km",
              "Standard deviation of the ellipse major axis length."),
    FieldSpec("DIAM_ELLI_MINOR_SD_IMG", "ASCII_Real", "float64", "km",
              "Standard deviation of the ellipse minor axis length."),
    FieldSpec("DIAM_ELLI_ANGLE_SD_IMG", "ASCII_Real", "float64", "deg",
              "Standard deviation of the major-axis azimuth."),
    FieldSpec("DIAM_ELLI_ECCEN_SD_IMG", "ASCII_Real", "float64", "1",
              "Standard deviation of the eccentricity."),
    FieldSpec("DIAM_ELLI_ELLIP_SD_IMG", "ASCII_Real", "float64", "1",
              "Standard deviation of the ellipticity."),
    FieldSpec("ARC_IMG", "ASCII_Real", "float64", "1",
              "Fraction of the crater's circumference over which rim points "
              "were actually traced, in (0, 1]. A low value means the "
              "centre and diameter rest on a short arc."),
    FieldSpec("PTS_RIM_IMG", "ASCII_String", "int64", "1",
              "Number of traced rim vertices. The label declares "
              "ASCII_String; the delivered values are integers. Parsed as "
              "int64 and the disagreement is reported, not reconciled."),
)

ROBBINS_FIELD_NAMES: tuple[str, ...] = tuple(f.name for f in ROBBINS_FIELDS)
ROBBINS_DTYPES: Mapping[str, str] = {f.name: f.dtype for f in ROBBINS_FIELDS}
_FIELDS_BY_NAME: Mapping[str, FieldSpec] = {f.name: f for f in ROBBINS_FIELDS}

#: Columns whose values are *lengths*, mapped to their fit definition.  The
#: ``DIAM_`` prefix in this catalogue does not imply a length: eccentricity,
#: ellipticity and azimuth share it.  Anything not listed here must not be
#: unit-converted as a diameter.
DIAMETER_COLUMNS: Mapping[str, str] = {
    "DIAM_CIRC_IMG": "circle_fit_diameter",
    "DIAM_CIRC_SD_IMG": "circle_fit_diameter_standard_deviation",
    "DIAM_ELLI_MAJOR_IMG": "ellipse_fit_major_axis_length",
    "DIAM_ELLI_MINOR_IMG": "ellipse_fit_minor_axis_length",
    "DIAM_ELLI_MAJOR_SD_IMG": "ellipse_fit_major_axis_standard_deviation",
    "DIAM_ELLI_MINOR_SD_IMG": "ellipse_fit_minor_axis_standard_deviation",
}

#: How the kilometre unit was established, since the label declares none.
#: Recorded here so the claim travels with the code.
DIAMETER_UNIT_EVIDENCE: tuple[str, ...] = (
    "min(DIAM_CIRC_IMG) == 1.0 exactly over all 1296796 records, matching the "
    "archive's own 'approximately complete ... larger than about 1-2 km' "
    "statement and the release's companion file name "
    "'Catalog_Moon_Release_20180815_1kmPlus.vrt'.",
    "max(DIAM_CIRC_IMG) == 2491.87 at LAT -52.698, LON 177.587 E, which is "
    "the South Pole-Aitken basin (~2500 km across): consistent only with km.",
    "The crater at LAT -89.6587, LON 129.883 E has DIAM_CIRC_IMG 20.8243, "
    "matching Shackleton (IAU: ~21 km at ~89.6 S, ~129.8 E). The archive "
    "description names Shackleton as the validation case for the fitting.",
)

DEFAULT_DIAMETER_COLUMN = "DIAM_CIRC_IMG"
DEFAULT_LAT_COLUMN = "LAT_CIRC_IMG"
DEFAULT_LON_COLUMN = "LON_CIRC_IMG"

SOURCE_LATITUDE_TYPE = "planetocentric"        # label <cart:latitude_type>
SOURCE_LONGITUDE_DIRECTION = "positive_east"   # label <cart:longitude_direction>
SOURCE_LONGITUDE_DOMAIN = (0.0, 360.0)         # label <cart:Bounding_Coordinates>
SOURCE_DIAMETER_UNIT = "km"                    # see DIAMETER_UNIT_EVIDENCE
SOURCE_BODY_RADIUS_M = 1737400.0               # label <cart:semi_major_radius>

#: ``<records>`` from the delivered label.  Used as an optional assertion, not
#: as a substitute for counting the file.
DECLARED_RECORD_COUNT = 1_296_796

KM_TO_M = 1000.0

#: The archive's own completeness statement, in metres, as a RANGE because the
#: archive states a range and says the exact point varies with location.
#: Treating any single number here as a hard limit would overstate the archive.
COMPLETENESS_LIMIT_M: tuple[float, float] = (1000.0, 2000.0)

#: DECISIONS.md D-003.
APPROVED_DIAMETER_RANGE_M: tuple[float, float] = (20.0, 1000.0)

#: docs/annotation_guide.md label vocabulary.
SOURCE_LABEL_TYPE = "catalogue"
REVIEW_STATUS_UNREVIEWED = "unreviewed"
GEOMETRY_KIND = "catalogue_circle_approximation"


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def _parse_numeric(raw: pd.Series, spec: FieldSpec, *, strict: bool) -> pd.Series:
    """Parse a string column into ``spec.dtype`` without silent coercion.

    Blank (the archive's way of saying "no value") becomes an explicit ``NaN``.
    Anything else that fails to parse raises, because a present-but-unparseable
    value is corruption, not absence.
    """
    stripped = raw.str.strip()
    blank = stripped == ""
    parsed = pd.to_numeric(stripped.where(~blank, None), errors="coerce")
    bad = parsed.isna() & ~blank
    if bad.any():
        idx = list(np.flatnonzero(bad.to_numpy()))
        if strict:
            raise MalformedCatalogueRow(
                spec.name, idx, [stripped.iloc[i] for i in idx]
            )
    if spec.dtype == "int64":
        if parsed.isna().any():
            # An integer field cannot hold "unknown" in a plain int64 column, so
            # use the nullable integer dtype rather than inventing a sentinel.
            return parsed.astype("Int64")
        return parsed.astype("int64")
    return parsed.astype("float64")


def load_robbins_csv(
    path: str | Path,
    *,
    columns: Sequence[str] | None = None,
    nrows: int | None = None,
    strict: bool = True,
    expect_records: int | None = None,
) -> pd.DataFrame:
    """Load the Robbins CSV with the schema the PDS4 label declares.

    Parameters
    ----------
    path:
        The delivered ``lunar_crater_database_robbins_2018.csv``, or a file with
        the same header.
    columns:
        Subset of :data:`ROBBINS_FIELD_NAMES` to keep.  ``None`` keeps all 21.
        Requesting a name not in the declared schema raises
        :class:`SchemaMismatch`.
    nrows:
        Read only the first ``nrows`` records.  Useful for inspection; a
        truncated read is never checked against ``expect_records``.
    strict:
        ``True`` (default) raises :class:`MalformedCatalogueRow` on any
        present-but-unparseable value.  ``False`` turns such a value into an
        explicit ``NaN`` -- still never into a plausible number.
    expect_records:
        If given (e.g. :data:`DECLARED_RECORD_COUNT`), the row count is
        asserted against it and a mismatch raises :class:`SchemaMismatch`.

    Returns
    -------
    pandas.DataFrame
        Columns typed per :data:`ROBBINS_DTYPES`.  Blank source fields are
        ``NaN`` (float columns) or ``pandas.NA`` (the nullable int column);
        nothing is filled.

    Notes
    -----
    The file is read with ``dtype=str`` and ``na_filter=False`` first, so
    pandas never applies its own NaN vocabulary ("NA", "null", "nan", "-1.#IND"
    and friends) to a column of real measurements.  Conversion is then
    explicit, per column, under :func:`_parse_numeric`.
    """
    path = Path(path)
    if columns is not None:
        unknown = [c for c in columns if c not in _FIELDS_BY_NAME]
        if unknown:
            raise SchemaMismatch(
                f"not declared by the PDS4 label: {unknown!r}; "
                f"declared fields are {list(ROBBINS_FIELD_NAMES)!r}"
            )

    header = pd.read_csv(path, nrows=0, dtype=str, na_filter=False)
    got = tuple(str(c).strip() for c in header.columns)
    if got != ROBBINS_FIELD_NAMES:
        missing = [c for c in ROBBINS_FIELD_NAMES if c not in got]
        extra = [c for c in got if c not in ROBBINS_FIELD_NAMES]
        raise SchemaMismatch(
            "header does not match the declared 21-field schema; "
            f"missing={missing!r} unexpected={extra!r}"
        )

    raw = pd.read_csv(
        path,
        dtype=str,
        na_filter=False,          # no silent NaN vocabulary
        keep_default_na=False,
        usecols=list(columns) if columns is not None else None,
        nrows=nrows,
    )
    raw.columns = [str(c).strip() for c in raw.columns]

    out = {}
    for name in raw.columns:
        spec = _FIELDS_BY_NAME[name]
        col = raw[name]
        if spec.dtype == "string":
            stripped = col.str.strip()
            if (stripped == "").any():
                n = int((stripped == "").sum())
                raise MalformedCatalogueRow(
                    name,
                    list(np.flatnonzero((stripped == "").to_numpy())),
                    [""] * n,
                )
            out[name] = stripped.astype("string")
        else:
            out[name] = _parse_numeric(col, spec, strict=strict)
    df = pd.DataFrame(out, columns=list(raw.columns))

    if expect_records is not None and nrows is None and len(df) != expect_records:
        raise SchemaMismatch(
            f"record count {len(df)} does not match the expected {expect_records}"
        )
    df.attrs["catalogue"] = "robbins_2018"
    df.attrs["source_path"] = str(path)
    return df


# --------------------------------------------------------------------------- #
# Normalisation to project conventions
# --------------------------------------------------------------------------- #
def normalise_catalogue(
    df: pd.DataFrame,
    *,
    diameter_column: str = DEFAULT_DIAMETER_COLUMN,
    lat_column: str = DEFAULT_LAT_COLUMN,
    lon_column: str = DEFAULT_LON_COLUMN,
) -> pd.DataFrame:
    """Add project-convention columns: ``lon_deg``, ``lat_deg``, ``diameter_m``.

    * ``lon_deg`` -- source longitude put into ``[-180, 180)`` by
      :func:`crater.geometry.wrap_longitude`.  The source domain is 0..360 east
      positive (label), the project domain is ``[-180, 180)`` east positive, so
      this is a pure domain change with no sign flip and no datum change.
    * ``lat_deg`` -- copied unchanged.  Both source and project are
      planetocentric on the same Moon 2000 sphere (R = 1737400 m in both the
      label and ``config/project.yaml``), so no latitude conversion applies.
      No planetographic conversion is performed, and none is needed: on a
      sphere the two coincide exactly.
    * ``diameter_m`` -- ``diameter_column`` times 1000.  Unknown stays
      ``NaN``.

    ``diameter_column`` must be one of :data:`DIAMETER_COLUMNS`; passing a
    dimensionless column such as ``DIAM_ELLI_ELLIP_IMG`` raises, because
    multiplying an ellipticity by 1000 would produce a confident nonsense.

    The returned frame carries the provenance of the choice in
    ``.attrs['diameter_definition']`` and in the ``diameter_definition``
    column, so a downstream consumer can never lose track of which of the
    catalogue's several diameter definitions it is holding.
    """
    for name, col in (("lat_column", lat_column), ("lon_column", lon_column)):
        if col not in df.columns:
            raise SchemaMismatch(f"{name}={col!r} is not present in the frame")
    if diameter_column not in DIAMETER_COLUMNS:
        raise SchemaMismatch(
            f"{diameter_column!r} is not a length column of this catalogue; "
            f"length columns are {sorted(DIAMETER_COLUMNS)!r}"
        )
    if diameter_column not in df.columns:
        raise SchemaMismatch(f"diameter column {diameter_column!r} not in the frame")

    out = df.copy()
    lon_src = out[lon_column].to_numpy(dtype=float)
    lat_src = out[lat_column].to_numpy(dtype=float)
    if np.any(np.abs(lat_src[np.isfinite(lat_src)]) > 90.0):
        raise MalformedCatalogueRow(
            lat_column,
            list(np.flatnonzero(np.abs(lat_src) > 90.0)),
            [str(v) for v in lat_src[np.abs(lat_src) > 90.0]],
        )

    out["lon_deg"] = g.wrap_longitude(lon_src)
    out["lat_deg"] = lat_src
    # Unknown diameters stay unknown: NaN * 1000 is NaN, and that is correct.
    out["diameter_m"] = out[diameter_column].to_numpy(dtype=float) * KM_TO_M
    out["diameter_definition"] = DIAMETER_COLUMNS[diameter_column]
    out["diameter_distance_kind"] = g.DIAMETER_DISTANCE_KIND
    out.attrs.update(df.attrs)
    out.attrs["diameter_column"] = diameter_column
    out.attrs["diameter_definition"] = DIAMETER_COLUMNS[diameter_column]
    out.attrs["longitude_domain"] = (-180.0, 180.0)
    out.attrs["latitude_type"] = SOURCE_LATITUDE_TYPE
    return out


def equivalent_ellipse_diameter_m(df: pd.DataFrame) -> np.ndarray:
    """``sqrt(major * minor)`` in metres, per ``docs/annotation_guide.md`` §3.

    Returns ``NaN`` wherever either axis is missing -- the 38 records in the
    delivered file with no ellipse fit are reported as unknown, not as their
    circle-fit diameter.
    """
    for col in ("DIAM_ELLI_MAJOR_IMG", "DIAM_ELLI_MINOR_IMG"):
        if col not in df.columns:
            raise SchemaMismatch(f"{col!r} required for the equivalent diameter")
    a = df["DIAM_ELLI_MAJOR_IMG"].to_numpy(dtype=float)
    b = df["DIAM_ELLI_MINOR_IMG"].to_numpy(dtype=float)
    with np.errstate(invalid="ignore"):
        return np.sqrt(a * b) * KM_TO_M


# --------------------------------------------------------------------------- #
# Selection in the projected plane
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ProjectedBox:
    """A bounding box in projected metres, half-open: ``[min, max)`` on both axes.

    Half-open so that abutting boxes partition the plane without double
    counting a centre on a shared edge.  The ROI is passed in as a parameter;
    this class hard-codes no extent and no centre (INTERFACES.md rule 5).
    """

    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def __post_init__(self) -> None:
        for name in ("x_min", "x_max", "y_min", "y_max"):
            v = float(getattr(self, name))
            if not np.isfinite(v):
                raise ValueError(f"{name} must be finite, got {getattr(self, name)!r}")
        if not self.x_max > self.x_min:
            raise ValueError("x_max must exceed x_min")
        if not self.y_max > self.y_min:
            raise ValueError("y_max must exceed y_min")

    @property
    def width_m(self) -> float:
        return float(self.x_max - self.x_min)

    @property
    def height_m(self) -> float:
        return float(self.y_max - self.y_min)

    @property
    def area_projected_m2(self) -> float:
        """Projected area.  NOT true surface area -- divide by ``k**2`` for that."""
        return self.width_m * self.height_m

    def contains(self, x_m, y_m) -> np.ndarray:
        x = np.asarray(x_m, dtype=float)
        y = np.asarray(y_m, dtype=float)
        return (
            (x >= self.x_min) & (x < self.x_max)
            & (y >= self.y_min) & (y < self.y_max)
        )


def project_centres(
    df: pd.DataFrame,
    *,
    body: Body = MOON,
    lon_0: float = 0.0,
    k0: float = 1.0,
) -> pd.DataFrame:
    """Add ``x_m``/``y_m`` from ``lon_deg``/``lat_deg`` via :func:`geometry.forward`."""
    for col in ("lon_deg", "lat_deg"):
        if col not in df.columns:
            raise SchemaMismatch(
                f"{col!r} missing; call normalise_catalogue() first"
            )
    out = df.copy()
    if len(out) == 0:
        out["x_m"] = np.array([], dtype=float)
        out["y_m"] = np.array([], dtype=float)
        out.attrs.update(df.attrs)
        return out
    x, y = g.forward(
        out["lon_deg"].to_numpy(dtype=float),
        out["lat_deg"].to_numpy(dtype=float),
        body=body,
        lon_0=lon_0,
        k0=k0,
    )
    out["x_m"] = np.asarray(x, dtype=float)
    out["y_m"] = np.asarray(y, dtype=float)
    out.attrs.update(df.attrs)
    out.attrs["projection_lon_0"] = float(lon_0)
    out.attrs["projection_k0"] = float(k0)
    out.attrs["projection_body_radius_m"] = float(body.radius_m)
    return out


def select_in_projected_box(
    df: pd.DataFrame,
    box: ProjectedBox,
    *,
    body: Body = MOON,
    lon_0: float = 0.0,
    k0: float = 1.0,
) -> pd.DataFrame:
    """Craters whose projected centre lies in ``box``.  **The correct selector.**

    Centres are projected with :func:`crater.geometry.forward` and tested
    against the projected box.  This is exact for the survey grid, which is
    itself defined in projected metres, and it is the only selector that is
    correct near the pole.

    The returned frame keeps ``x_m``/``y_m`` so a caller can place the crater in
    a tile without reprojecting.  Rows with an unknown centre (``NaN`` longitude
    or latitude) are excluded and counted in
    ``.attrs['excluded_unknown_centre']`` rather than dropped silently.
    """
    projected = project_centres(df, body=body, lon_0=lon_0, k0=k0)
    finite = (
        np.isfinite(projected["x_m"].to_numpy(dtype=float))
        & np.isfinite(projected["y_m"].to_numpy(dtype=float))
    )
    inside = box.contains(projected["x_m"], projected["y_m"]) & finite
    out = projected.loc[inside].copy()
    out.attrs.update(projected.attrs)
    out.attrs["selection"] = "projected_box"
    out.attrs["selection_box"] = (box.x_min, box.x_max, box.y_min, box.y_max)
    out.attrs["excluded_unknown_centre"] = int((~finite).sum())
    return out


def select_in_degree_box(
    df: pd.DataFrame,
    *,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
) -> pd.DataFrame:
    """Craters inside a planar lon/lat rectangle.  **WRONG near the pole.**

    Provided only so that the error can be measured and demonstrated (see
    ``tests/test_catalogue.py`` and ``reports/robbins_audit.md``).  A projected
    square does not map to a lon/lat rectangle: at 86 S the projected ROI's
    corners and edge midpoints have different latitudes, so the bounding
    degree box covers extra ground outside the survey and, for a box narrower
    than the lon span, misses ground inside it.  Never use this to select
    survey craters.

    Longitude comparison is done on the wrapped ``[-180, 180)`` values via
    :func:`crater.geometry.longitude_difference`, so a box spanning the
    antimeridian is handled correctly as a *degree* box -- it is still the
    wrong shape.
    """
    if "lon_deg" not in df.columns or "lat_deg" not in df.columns:
        raise SchemaMismatch("call normalise_catalogue() first")
    lon = df["lon_deg"].to_numpy(dtype=float)
    lat = df["lat_deg"].to_numpy(dtype=float)
    lo = g.wrap_longitude(lon_min)
    span = g.longitude_difference(lon_max, lon_min)
    if span <= 0:
        span = span + 360.0
    offset = g.longitude_difference(lon, lo)
    offset = np.where(offset < 0, offset + 360.0, offset)
    inside = (offset >= 0) & (offset < span) & (lat >= lat_min) & (lat < lat_max)
    out = df.loc[inside].copy()
    out.attrs.update(df.attrs)
    out.attrs["selection"] = "degree_box_INCORRECT_NEAR_POLE"
    return out


# --------------------------------------------------------------------------- #
# Catalogue-derived rim geometry
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class CatalogueRimAnnotation:
    """Rim geometry generated from a catalogue centre and diameter.

    This is **not an observed rim**.  It is a circle of constant ground radius
    about the catalogue centre, sampled geodesically by
    :func:`crater.geometry.crater_rim_samples`.  The true rim is neither
    circular nor necessarily concentric with a circle fitted to a partial arc
    (see ``ARC_IMG``), so the label vocabulary of ``docs/annotation_guide.md``
    marks it accordingly and it must never be promoted to ``human``.

    ``diameter_m`` is :data:`crater.boxes.UNMEASURED` when the catalogue has no
    diameter for the crater; in that case ``rim_lon``/``rim_lat`` are empty and
    ``quality_flag`` is ``"unmeasured"``.  Nothing is guessed.
    """

    crater_id: str
    centre_lon_deg: float
    centre_lat_deg: float
    diameter_m: float | Unmeasured
    rim_lon_deg: np.ndarray
    rim_lat_deg: np.ndarray
    source_label_type: str = SOURCE_LABEL_TYPE
    review_status: str = REVIEW_STATUS_UNREVIEWED
    geometry_kind: str = GEOMETRY_KIND
    diameter_definition: str = DIAMETER_COLUMNS[DEFAULT_DIAMETER_COLUMN]
    diameter_distance_kind: str = g.DIAMETER_DISTANCE_KIND
    rim_arc_fraction: float | Unmeasured = UNMEASURED
    quality_flag: str = "ok"
    notes: tuple[str, ...] = _dc_field(default_factory=tuple)

    @property
    def is_measured(self) -> bool:
        return self.diameter_m is not UNMEASURED


def catalogue_rim_annotations(
    df: pd.DataFrame,
    *,
    n_samples: int = 72,
    body: Body = MOON,
    id_column: str = "CRATER_ID",
) -> list[CatalogueRimAnnotation]:
    """Generate catalogue-derived rim annotations for every row of ``df``.

    ``df`` must already carry ``lon_deg``, ``lat_deg`` and ``diameter_m``
    (i.e. have been through :func:`normalise_catalogue`).

    Every annotation is labelled ``source_label_type="catalogue"`` and
    ``review_status="unreviewed"``.  Rows with an unknown diameter or an
    unknown centre yield an annotation with
    :data:`crater.boxes.UNMEASURED` and no rim samples rather than being
    dropped: the crater is known to exist even where its extent is not.
    """
    for col in ("lon_deg", "lat_deg", "diameter_m"):
        if col not in df.columns:
            raise SchemaMismatch(f"{col!r} missing; call normalise_catalogue() first")
    definition = str(df.attrs.get("diameter_definition",
                                  DIAMETER_COLUMNS[DEFAULT_DIAMETER_COLUMN]))
    ids = (
        df[id_column].astype("string")
        if id_column in df.columns
        else pd.Series([f"row_{i}" for i in range(len(df))], index=df.index, dtype="string")
    )
    arcs = df["ARC_IMG"] if "ARC_IMG" in df.columns else None

    out: list[CatalogueRimAnnotation] = []
    for pos, idx in enumerate(df.index):
        lon = float(df["lon_deg"].iloc[pos])
        lat = float(df["lat_deg"].iloc[pos])
        dia = float(df["diameter_m"].iloc[pos])
        raw_id = ids.iloc[pos]
        crater_id = f"row_{idx}" if pd.isna(raw_id) else str(raw_id)
        arc: float | Unmeasured = UNMEASURED
        if arcs is not None:
            a = arcs.iloc[pos]
            if not pd.isna(a):
                arc = float(a)

        notes: list[str] = []
        if not (np.isfinite(lon) and np.isfinite(lat)):
            out.append(CatalogueRimAnnotation(
                crater_id=crater_id,
                centre_lon_deg=lon, centre_lat_deg=lat,
                diameter_m=UNMEASURED,
                rim_lon_deg=np.empty(0, dtype=float),
                rim_lat_deg=np.empty(0, dtype=float),
                diameter_definition=definition,
                rim_arc_fraction=arc,
                quality_flag="unmeasured",
                notes=("centre unknown in the catalogue",),
            ))
            continue
        if not np.isfinite(dia) or dia <= 0.0:
            out.append(CatalogueRimAnnotation(
                crater_id=crater_id,
                centre_lon_deg=lon, centre_lat_deg=lat,
                diameter_m=UNMEASURED,
                rim_lon_deg=np.empty(0, dtype=float),
                rim_lat_deg=np.empty(0, dtype=float),
                diameter_definition=definition,
                rim_arc_fraction=arc,
                quality_flag="unmeasured",
                notes=("diameter unknown in the catalogue",),
            ))
            continue

        rim_lon, rim_lat = g.crater_rim_samples(
            lon, lat, dia, n_samples=n_samples, body=body
        )
        lo, hi = APPROVED_DIAMETER_RANGE_M
        if dia > hi:
            notes.append("above_range=1")
        if dia < lo:
            notes.append("below_range=1")
        if isinstance(arc, float) and arc < 0.5:
            notes.append("rim_arc_below_half: centre and diameter rest on a short arc")
        out.append(CatalogueRimAnnotation(
            crater_id=crater_id,
            centre_lon_deg=lon, centre_lat_deg=lat,
            diameter_m=dia,
            rim_lon_deg=np.asarray(rim_lon, dtype=float),
            rim_lat_deg=np.asarray(rim_lat, dtype=float),
            diameter_definition=definition,
            rim_arc_fraction=arc,
            quality_flag="ok",
            notes=tuple(notes),
        ))
    return out


def rim_annotation_frame(
    annotations: Iterable[CatalogueRimAnnotation],
) -> pd.DataFrame:
    """Flatten annotations to a table, keeping unknowns explicitly unknown.

    ``diameter_m`` is ``NaN`` and ``diameter_known`` is ``False`` for an
    unmeasured crater.  The pair is deliberate: a consumer that reads only
    ``diameter_m`` sees ``NaN`` (not a number), and one that wants to branch has
    an unambiguous boolean instead of an ``isnan`` guess.
    """
    rows = []
    for a in annotations:
        measured = a.is_measured
        rows.append({
            "crater_id": a.crater_id,
            "centre_lon_deg": a.centre_lon_deg,
            "centre_lat_deg": a.centre_lat_deg,
            "diameter_m": float(a.diameter_m) if measured else np.nan,
            "diameter_known": bool(measured),
            "diameter_definition": a.diameter_definition,
            "diameter_distance_kind": a.diameter_distance_kind,
            "rim_samples": int(a.rim_lon_deg.size),
            "rim_arc_fraction": (
                float(a.rim_arc_fraction)
                if a.rim_arc_fraction is not UNMEASURED else np.nan
            ),
            "rim_arc_fraction_known": a.rim_arc_fraction is not UNMEASURED,
            "source_label_type": a.source_label_type,
            "review_status": a.review_status,
            "geometry_kind": a.geometry_kind,
            "quality_flag": a.quality_flag,
            "notes": ";".join(a.notes),
        })
    return pd.DataFrame(
        rows,
        columns=[
            "crater_id", "centre_lon_deg", "centre_lat_deg", "diameter_m",
            "diameter_known", "diameter_definition", "diameter_distance_kind",
            "rim_samples", "rim_arc_fraction", "rim_arc_fraction_known",
            "source_label_type", "review_status", "geometry_kind",
            "quality_flag", "notes",
        ],
    )


# --------------------------------------------------------------------------- #
# Small audit helper
# --------------------------------------------------------------------------- #
def diameter_histogram(
    diameters_m: Sequence[float] | np.ndarray,
    edges_m: Sequence[float] | np.ndarray,
) -> pd.DataFrame:
    """Count diameters into ``[edge_i, edge_i+1)`` bins, reporting unknowns.

    Unknown (``NaN``) diameters are counted in their own ``unknown`` row rather
    than being dropped, so the bin counts plus the unknown count always equal
    the input length.  A histogram that quietly loses rows is how a label gap
    gets hidden.
    """
    d = np.asarray(diameters_m, dtype=float)
    e = np.asarray(edges_m, dtype=float)
    if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0):
        raise ValueError("edges_m must be strictly increasing with at least 2 values")
    known = np.isfinite(d)
    counts = np.histogram(d[known], bins=e)[0]
    below = int(np.sum(d[known] < e[0]))
    above = int(np.sum(d[known] >= e[-1]))
    rows = [{"bin_low_m": np.nan, "bin_high_m": float(e[0]), "label": "below_first_edge",
             "count": below}]
    for i in range(len(counts)):
        rows.append({
            "bin_low_m": float(e[i]), "bin_high_m": float(e[i + 1]),
            "label": f"[{e[i]:g}, {e[i + 1]:g})", "count": int(counts[i]),
        })
    rows.append({"bin_low_m": float(e[-1]), "bin_high_m": np.nan,
                 "label": "at_or_above_last_edge", "count": above})
    rows.append({"bin_low_m": np.nan, "bin_high_m": np.nan, "label": "unknown",
                 "count": int((~known).sum())})
    return pd.DataFrame(rows)
