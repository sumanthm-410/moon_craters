"""Tests for :mod:`crater.catalogue`.

No network.  Every test runs against a small synthetic CSV written with the
*real* 21-field header transcribed from the delivered PDS4 label, so the parser
is exercised against the actual schema without needing the 238 MB product.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from crater import catalogue as cat
from crater import geometry as g
from crater.boxes import UNMEASURED

# --------------------------------------------------------------------------- #
# Synthetic fixture: the real header, a handful of hand-built records
# --------------------------------------------------------------------------- #
HEADER = ",".join(cat.ROBBINS_FIELD_NAMES)

#: One record template in label field order.  Values are deliberately simple so
#: a conversion error is visible by eye.
_TEMPLATE = {
    "CRATER_ID": "10-1-000001",
    "LAT_CIRC_IMG": "-86.0",
    "LON_CIRC_IMG": "2.9",
    "LAT_ELLI_IMG": "-86.0",
    "LON_ELLI_IMG": "2.9",
    "DIAM_CIRC_IMG": "1.5",
    "DIAM_CIRC_SD_IMG": "0.05",
    "DIAM_ELLI_MAJOR_IMG": "1.6",
    "DIAM_ELLI_MINOR_IMG": "1.4",
    "DIAM_ELLI_ECCEN_IMG": "0.4841",
    "DIAM_ELLI_ELLIP_IMG": "1.142857",
    "DIAM_ELLI_ANGLE_IMG": "30.0",
    "LAT_ELLI_SD_IMG": "0.001",
    "LON_ELLI_SD_IMG": "0.002",
    "DIAM_ELLI_MAJOR_SD_IMG": "0.01",
    "DIAM_ELLI_MINOR_SD_IMG": "0.01",
    "DIAM_ELLI_ANGLE_SD_IMG": "0.5",
    "DIAM_ELLI_ECCEN_SD_IMG": "0.001",
    "DIAM_ELLI_ELLIP_SD_IMG": "0.0005",
    "ARC_IMG": "0.9",
    "PTS_RIM_IMG": "120",
}


def _record(**overrides: str) -> str:
    row = dict(_TEMPLATE)
    row.update(overrides)
    return ",".join(row[name] for name in cat.ROBBINS_FIELD_NAMES)


def write_csv(tmp_path, records, name="synthetic_robbins.csv"):
    path = tmp_path / name
    # The real product uses CRLF; write CRLF so the parser is tested on it.
    body = "\r\n".join([HEADER] + list(records)) + "\r\n"
    path.write_bytes(body.encode("utf-8"))
    return path


@pytest.fixture()
def roi_box():
    """The project's ROI as a projected box (passed in, never hard-coded in src)."""
    return cat.ProjectedBox(x_min=-11000.0, x_max=10000.0,
                            y_min=111000.0, y_max=132000.0)


# --------------------------------------------------------------------------- #
# Schema and loading
# --------------------------------------------------------------------------- #
def test_declared_schema_matches_the_pds4_label():
    assert len(cat.ROBBINS_FIELDS) == 21
    assert cat.ROBBINS_FIELD_NAMES[0] == "CRATER_ID"
    assert cat.ROBBINS_FIELD_NAMES[1:5] == (
        "LAT_CIRC_IMG", "LON_CIRC_IMG", "LAT_ELLI_IMG", "LON_ELLI_IMG",
    )
    assert cat.ROBBINS_FIELD_NAMES[-2:] == ("ARC_IMG", "PTS_RIM_IMG")
    # No field in this product carries a label-declared unit; the module must
    # not pretend otherwise.
    assert all(not f.unit_declared_by_label for f in cat.ROBBINS_FIELDS)


def test_load_assigns_explicit_dtypes(tmp_path):
    path = write_csv(tmp_path, [_record(), _record(CRATER_ID="10-1-000002")])
    df = cat.load_robbins_csv(path)
    assert list(df.columns) == list(cat.ROBBINS_FIELD_NAMES)
    assert df["DIAM_CIRC_IMG"].dtype == np.float64
    assert str(df["PTS_RIM_IMG"].dtype) == "int64"
    assert str(df["CRATER_ID"].dtype) == "string"
    assert len(df) == 2


def test_header_mismatch_is_rejected(tmp_path):
    path = tmp_path / "bad_header.csv"
    path.write_bytes(b"lon,lat,diam\r\n1,2,3\r\n")
    with pytest.raises(cat.SchemaMismatch):
        cat.load_robbins_csv(path)


def test_record_count_assertion(tmp_path):
    path = write_csv(tmp_path, [_record()])
    with pytest.raises(cat.SchemaMismatch):
        cat.load_robbins_csv(path, expect_records=2)
    assert len(cat.load_robbins_csv(path, expect_records=1)) == 1


def test_requesting_an_undeclared_column_raises(tmp_path):
    path = write_csv(tmp_path, [_record()])
    with pytest.raises(cat.SchemaMismatch):
        cat.load_robbins_csv(path, columns=["DIAMETER_KM"])


# --------------------------------------------------------------------------- #
# Malformed rows are rejected, not coerced
# --------------------------------------------------------------------------- #
def test_unparseable_number_raises_rather_than_becoming_nan(tmp_path):
    path = write_csv(tmp_path, [_record(), _record(DIAM_CIRC_IMG="1.2.3")])
    with pytest.raises(cat.MalformedCatalogueRow) as exc:
        cat.load_robbins_csv(path)
    assert exc.value.column == "DIAM_CIRC_IMG"
    assert exc.value.rows == (1,)
    assert exc.value.values == ("1.2.3",)


def test_pandas_nan_vocabulary_is_not_applied_silently(tmp_path):
    """A literal 'NA' in a measurement column is corruption, not a blank."""
    path = write_csv(tmp_path, [_record(DIAM_CIRC_IMG="NA")])
    with pytest.raises(cat.MalformedCatalogueRow):
        cat.load_robbins_csv(path)


def test_non_strict_turns_corruption_into_explicit_unknown(tmp_path):
    path = write_csv(tmp_path, [_record(DIAM_CIRC_IMG="oops")])
    df = cat.load_robbins_csv(path, strict=False)
    assert np.isnan(df["DIAM_CIRC_IMG"].iloc[0])


def test_empty_crater_id_is_malformed(tmp_path):
    path = write_csv(tmp_path, [_record(CRATER_ID="")])
    with pytest.raises(cat.MalformedCatalogueRow):
        cat.load_robbins_csv(path)


def test_latitude_out_of_range_is_malformed(tmp_path):
    path = write_csv(tmp_path, [_record(LAT_CIRC_IMG="-95.0")])
    df = cat.load_robbins_csv(path)
    with pytest.raises(cat.MalformedCatalogueRow):
        cat.normalise_catalogue(df)


# --------------------------------------------------------------------------- #
# Coordinate conventions
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "lon_source, lon_expected",
    [
        ("0.0", 0.0),
        ("2.9", 2.9),
        ("90.0", 90.0),
        ("179.9", 179.9),
        ("180.0", -180.0),      # 180 E is the same meridian as -180
        ("190.0", -170.0),
        ("264.757", -95.243),   # a real-looking 0-360 value
        ("354.341", -5.659),    # the ROI's western edge
        ("360.0", 0.0),
    ],
)
def test_longitude_domain_change_matches_geometry_wrap(tmp_path, lon_source, lon_expected):
    path = write_csv(tmp_path, [_record(LON_CIRC_IMG=lon_source)])
    df = cat.normalise_catalogue(cat.load_robbins_csv(path))
    got = float(df["lon_deg"].iloc[0])
    assert got == pytest.approx(lon_expected, abs=1e-9)
    # and it is exactly what geometry.wrap_longitude does -- no private copy
    assert got == pytest.approx(float(g.wrap_longitude(float(lon_source))), abs=0.0)
    assert -180.0 <= got < 180.0


def test_longitude_round_trip_through_projection(tmp_path):
    """0-360 source -> wrapped -> forward -> inverse returns the same ground point."""
    sources = ["0.0001", "2.9", "97.5", "180.0", "180.0001", "271.3", "359.9999"]
    records = [_record(CRATER_ID=f"10-1-{i:06d}", LON_CIRC_IMG=s, LAT_CIRC_IMG="-86.0")
               for i, s in enumerate(sources)]
    df = cat.normalise_catalogue(cat.load_robbins_csv(write_csv(tmp_path, records)))
    x, y = g.forward(df["lon_deg"].to_numpy(), df["lat_deg"].to_numpy())
    lon_back, lat_back = g.inverse(x, y)
    np.testing.assert_allclose(lon_back, df["lon_deg"].to_numpy(), atol=1e-9)
    np.testing.assert_allclose(lat_back, df["lat_deg"].to_numpy(), atol=1e-9)


def test_crater_near_the_antimeridian(tmp_path):
    """A crater just either side of 180 E must stay a few metres apart, not 360 deg."""
    records = [
        _record(CRATER_ID="A", LON_CIRC_IMG="179.99990", LAT_CIRC_IMG="-86.0"),
        _record(CRATER_ID="B", LON_CIRC_IMG="180.00010", LAT_CIRC_IMG="-86.0"),
    ]
    df = cat.normalise_catalogue(cat.load_robbins_csv(write_csv(tmp_path, records)))
    a, b = df["lon_deg"].to_numpy()
    assert a == pytest.approx(179.9999)
    assert b == pytest.approx(-179.9999)      # wrapped to the negative side
    # The *signed smallest difference* must be the small one, not ~360.
    assert abs(float(g.longitude_difference(b, a))) == pytest.approx(0.0002, rel=1e-9)
    # ... and the ground separation is small: ~2.1 km per degree of longitude
    # at 86 S, so 0.0002 deg is well under a metre.
    d = g.great_circle_distance(a, -86.0, b, -86.0)
    assert d < 1.0
    # Projected coordinates must also be adjacent, not mirrored.
    x, y = g.forward(df["lon_deg"].to_numpy(), df["lat_deg"].to_numpy())
    assert np.hypot(x[0] - x[1], y[0] - y[1]) < 1.0


def test_latitude_is_not_converted(tmp_path):
    """Source and project are both planetocentric on the same sphere."""
    path = write_csv(tmp_path, [_record(LAT_CIRC_IMG="-86.3261")])
    df = cat.normalise_catalogue(cat.load_robbins_csv(path))
    assert float(df["lat_deg"].iloc[0]) == pytest.approx(-86.3261, abs=0.0)
    assert cat.SOURCE_LATITUDE_TYPE == "planetocentric"
    assert cat.SOURCE_BODY_RADIUS_M == 1737400.0


# --------------------------------------------------------------------------- #
# Unit conversion
# --------------------------------------------------------------------------- #
def test_diameter_km_to_m(tmp_path):
    records = [
        _record(CRATER_ID="a", DIAM_CIRC_IMG="1.0"),
        _record(CRATER_ID="b", DIAM_CIRC_IMG="1.5"),
        _record(CRATER_ID="c", DIAM_CIRC_IMG="20.8243"),
        _record(CRATER_ID="d", DIAM_CIRC_IMG="2491.87"),
    ]
    df = cat.normalise_catalogue(cat.load_robbins_csv(write_csv(tmp_path, records)))
    np.testing.assert_allclose(
        df["diameter_m"].to_numpy(), [1000.0, 1500.0, 20824.3, 2491870.0]
    )
    assert cat.SOURCE_DIAMETER_UNIT == "km"
    assert cat.KM_TO_M == 1000.0


def test_diameter_definition_is_carried(tmp_path):
    df = cat.load_robbins_csv(write_csv(tmp_path, [_record()]))
    circ = cat.normalise_catalogue(df, diameter_column="DIAM_CIRC_IMG")
    major = cat.normalise_catalogue(df, diameter_column="DIAM_ELLI_MAJOR_IMG")
    assert circ["diameter_definition"].iloc[0] == "circle_fit_diameter"
    assert major["diameter_definition"].iloc[0] == "ellipse_fit_major_axis_length"
    assert float(circ["diameter_m"].iloc[0]) == pytest.approx(1500.0)
    assert float(major["diameter_m"].iloc[0]) == pytest.approx(1600.0)
    assert circ.attrs["diameter_column"] == "DIAM_CIRC_IMG"


def test_dimensionless_column_cannot_be_used_as_a_diameter(tmp_path):
    df = cat.load_robbins_csv(write_csv(tmp_path, [_record()]))
    for bad in ("DIAM_ELLI_ECCEN_IMG", "DIAM_ELLI_ELLIP_IMG", "DIAM_ELLI_ANGLE_IMG"):
        with pytest.raises(cat.SchemaMismatch):
            cat.normalise_catalogue(df, diameter_column=bad)


def test_equivalent_ellipse_diameter(tmp_path):
    path = write_csv(tmp_path, [
        _record(CRATER_ID="a", DIAM_ELLI_MAJOR_IMG="1.6", DIAM_ELLI_MINOR_IMG="1.4"),
        _record(CRATER_ID="b", DIAM_ELLI_MAJOR_IMG="", DIAM_ELLI_MINOR_IMG=""),
    ])
    df = cat.load_robbins_csv(path)
    eq = cat.equivalent_ellipse_diameter_m(df)
    assert eq[0] == pytest.approx(np.sqrt(1.6 * 1.4) * 1000.0)
    assert np.isnan(eq[1])          # no ellipse fit -> stays unknown


# --------------------------------------------------------------------------- #
# Unknowns stay unknown
# --------------------------------------------------------------------------- #
def test_blank_field_is_explicit_nan_not_filled(tmp_path):
    path = write_csv(tmp_path, [_record(DIAM_ELLI_MAJOR_IMG="", DIAM_ELLI_MINOR_IMG="")])
    df = cat.load_robbins_csv(path)
    assert np.isnan(df["DIAM_ELLI_MAJOR_IMG"].iloc[0])
    assert np.isnan(df["DIAM_ELLI_MINOR_IMG"].iloc[0])
    # the circle fit is still present and must not have been touched
    assert float(df["DIAM_CIRC_IMG"].iloc[0]) == pytest.approx(1.5)


def test_blank_integer_field_uses_nullable_dtype(tmp_path):
    path = write_csv(tmp_path, [_record(PTS_RIM_IMG="")])
    df = cat.load_robbins_csv(path)
    assert str(df["PTS_RIM_IMG"].dtype) == "Int64"
    assert pd.isna(df["PTS_RIM_IMG"].iloc[0])


def test_unknown_diameter_yields_unmeasured_annotation_not_a_guess(tmp_path):
    path = write_csv(tmp_path, [
        _record(CRATER_ID="known", DIAM_CIRC_IMG="0.4"),
        _record(CRATER_ID="blank", DIAM_CIRC_IMG=""),
    ])
    df = cat.normalise_catalogue(cat.load_robbins_csv(path))
    anns = cat.catalogue_rim_annotations(df, n_samples=8)
    assert len(anns) == 2
    known, blank = anns
    assert known.diameter_m == pytest.approx(400.0)
    assert known.rim_lon_deg.size == 8
    assert blank.diameter_m is UNMEASURED
    assert blank.rim_lon_deg.size == 0
    assert blank.quality_flag == "unmeasured"
    assert "diameter unknown in the catalogue" in blank.notes
    frame = cat.rim_annotation_frame(anns)
    assert bool(frame["diameter_known"].iloc[0]) is True
    assert bool(frame["diameter_known"].iloc[1]) is False
    assert np.isnan(frame["diameter_m"].iloc[1])


def test_histogram_never_loses_rows():
    d = np.array([10.0, 50.0, 500.0, 1500.0, np.nan])
    h = cat.diameter_histogram(d, [20.0, 1000.0])
    assert int(h["count"].sum()) == len(d)
    assert int(h.loc[h.label == "below_first_edge", "count"].iloc[0]) == 1
    assert int(h.loc[h.label == "unknown", "count"].iloc[0]) == 1
    assert int(h.loc[h.label == "at_or_above_last_edge", "count"].iloc[0]) == 1


# --------------------------------------------------------------------------- #
# Annotation labelling (docs/annotation_guide.md)
# --------------------------------------------------------------------------- #
def test_annotations_are_labelled_catalogue_and_unreviewed(tmp_path):
    df = cat.normalise_catalogue(cat.load_robbins_csv(write_csv(tmp_path, [_record()])))
    (a,) = cat.catalogue_rim_annotations(df, n_samples=16)
    assert a.source_label_type == "catalogue"
    assert a.review_status == "unreviewed"
    assert a.geometry_kind == "catalogue_circle_approximation"
    assert a.diameter_distance_kind == g.DIAMETER_DISTANCE_KIND
    assert a.rim_arc_fraction == pytest.approx(0.9)
    frame = cat.rim_annotation_frame([a])
    assert frame["source_label_type"].iloc[0] == "catalogue"
    assert frame["review_status"].iloc[0] == "unreviewed"


def test_rim_samples_lie_at_the_catalogue_radius(tmp_path):
    """Generated rim points are a constant *ground* radius from the centre."""
    path = write_csv(tmp_path, [_record(DIAM_CIRC_IMG="0.5",
                                        LAT_CIRC_IMG="-86.0", LON_CIRC_IMG="2.9")])
    df = cat.normalise_catalogue(cat.load_robbins_csv(path))
    (a,) = cat.catalogue_rim_annotations(df, n_samples=36)
    d = g.great_circle_distance(a.centre_lon_deg, a.centre_lat_deg,
                                a.rim_lon_deg, a.rim_lat_deg)
    np.testing.assert_allclose(d, 250.0, rtol=1e-9)


def test_above_range_crater_is_flagged(tmp_path):
    path = write_csv(tmp_path, [_record(CRATER_ID="big", DIAM_CIRC_IMG="2.0")])
    df = cat.normalise_catalogue(cat.load_robbins_csv(path))
    (a,) = cat.catalogue_rim_annotations(df, n_samples=8)
    assert "above_range=1" in a.notes


# --------------------------------------------------------------------------- #
# Projected-box selection vs a naive degree box
# --------------------------------------------------------------------------- #
def test_projected_box_half_open_edges(roi_box):
    """Edge semantics are asserted on exact coordinates, not via a round trip.

    ``[min, max)`` on both axes, so abutting boxes partition the plane and a
    centre on a shared edge is counted exactly once.
    """
    assert bool(roi_box.contains(roi_box.x_min, roi_box.y_min)) is True
    assert bool(roi_box.contains(roi_box.x_max, 121500.0)) is False
    assert bool(roi_box.contains(0.0, roi_box.y_max)) is False
    assert bool(roi_box.contains(roi_box.x_min - 1e-9, 121500.0)) is False
    assert bool(roi_box.contains(np.nan, 121500.0)) is False


def test_projected_box_selection(tmp_path, roi_box):
    """Points chosen by their projected coordinates land where expected.

    The targets stay a metre clear of the box edges: a crater centre is placed
    by inverting a projected coordinate and is then re-projected by the
    selector, and that float round trip is only good to a small fraction of a
    metre.  Exact edge behaviour is covered by
    :func:`test_projected_box_half_open_edges` instead.
    """
    # Build craters by inverting target projected coordinates, so the expected
    # answer is known independently of the selector.
    targets = [
        (0.0, 121500.0, True),          # ROI centre-ish
        (-10999.0, 111001.0, True),     # just inside the SW corner
        (9999.0, 131999.0, True),       # just inside the NE corner
        (10001.0, 121500.0, False),     # just outside x_max
        (0.0, 132001.0, False),         # just outside y_max
        (-10999.0, 131999.0, True),     # just inside the NW corner
        (0.0, 150000.0, False),         # north of the ROI
        (40000.0, 121500.0, False),     # east of the ROI
    ]
    records = []
    for i, (x, y, _) in enumerate(targets):
        lon, lat = g.inverse(x, y)
        records.append(_record(CRATER_ID=f"t{i}",
                               LON_CIRC_IMG=f"{lon % 360.0:.12f}",
                               LAT_CIRC_IMG=f"{lat:.12f}"))
    df = cat.normalise_catalogue(cat.load_robbins_csv(write_csv(tmp_path, records)))
    sel = cat.select_in_projected_box(df, roi_box)
    expected = {f"t{i}" for i, (_, _, keep) in enumerate(targets) if keep}
    assert set(sel["CRATER_ID"].tolist()) == expected
    # x_m / y_m are returned and agree with the targets
    for cid, (x, y, keep) in zip([f"t{i}" for i in range(len(targets))], targets):
        if not keep:
            continue
        row = sel[sel["CRATER_ID"] == cid].iloc[0]
        assert float(row["x_m"]) == pytest.approx(x, abs=1e-6)
        assert float(row["y_m"]) == pytest.approx(y, abs=1e-6)


def test_projected_box_and_degree_box_disagree_near_the_pole(tmp_path, roi_box):
    """The headline correctness test: a degree box is the wrong shape at 86 S.

    The ROI's projected corners span lon -5.659..5.148 and lat
    -86.326..-85.634.  A crater placed at the lon/lat *corner* of that
    bounding degree box is well outside the projected square, because the
    square's corners are its extreme longitudes and its extreme latitudes
    occur at different places.
    """
    corners = [(roi_box.x_min, roi_box.y_min), (roi_box.x_max, roi_box.y_min),
               (roi_box.x_min, roi_box.y_max), (roi_box.x_max, roi_box.y_max)]
    lons, lats = [], []
    for x, y in corners:
        lo, la = g.inverse(x, y)
        lons.append(lo)
        lats.append(la)
    lon_min, lon_max = min(lons), max(lons)
    lat_min, lat_max = min(lats), max(lats)

    # A point at (lon_min, lat_max): inside the degree box, outside the square.
    lon_lat_corner = (lon_min, lat_max)
    xc, yc = g.forward(*lon_lat_corner)
    assert not bool(roi_box.contains(xc, yc))

    records = [
        _record(CRATER_ID="square_and_degrees",
                LON_CIRC_IMG="0.0", LAT_CIRC_IMG=f"{g.inverse(0.0, 121500.0)[1]:.12f}"),
        _record(CRATER_ID="degrees_only",
                LON_CIRC_IMG=f"{lon_lat_corner[0] % 360.0:.12f}",
                LAT_CIRC_IMG=f"{lon_lat_corner[1]:.12f}"),
    ]
    df = cat.normalise_catalogue(cat.load_robbins_csv(write_csv(tmp_path, records)))

    projected = cat.select_in_projected_box(df, roi_box)
    degrees = cat.select_in_degree_box(
        df, lon_min=lon_min, lon_max=lon_max,
        lat_min=lat_min, lat_max=lat_max + 1e-9,
    )
    assert set(projected["CRATER_ID"]) == {"square_and_degrees"}
    assert set(degrees["CRATER_ID"]) == {"square_and_degrees", "degrees_only"}
    assert len(degrees) > len(projected)
    assert degrees.attrs["selection"] == "degree_box_INCORRECT_NEAR_POLE"
    assert projected.attrs["selection"] == "projected_box"


def test_degree_box_also_misses_ground_inside_the_square(tmp_path, roi_box):
    """The error is not one-sided: a *narrower* degree box drops real ROI craters.

    Taking the degree box from the ROI's edge midpoints instead of its corners
    excludes craters that genuinely lie inside the projected ROI.
    """
    # Latitude at the middle of the ROI's northern edge is the box's
    # northernmost *edge* latitude, but the NE/NW corners reach further north.
    _, lat_edge_mid = g.inverse(0.0, roi_box.y_max)
    lon_corner, lat_corner = g.inverse(roi_box.x_max - 1.0, roi_box.y_max - 1.0)
    assert lat_corner > lat_edge_mid     # the corner is north of the edge midpoint

    records = [_record(CRATER_ID="corner_crater",
                       LON_CIRC_IMG=f"{lon_corner % 360.0:.12f}",
                       LAT_CIRC_IMG=f"{lat_corner:.12f}")]
    df = cat.normalise_catalogue(cat.load_robbins_csv(write_csv(tmp_path, records)))

    assert len(cat.select_in_projected_box(df, roi_box)) == 1
    narrow = cat.select_in_degree_box(
        df, lon_min=-5.659, lon_max=5.148,
        lat_min=-86.326, lat_max=lat_edge_mid,
    )
    assert len(narrow) == 0


def test_projected_box_rejects_degenerate_extent():
    with pytest.raises(ValueError):
        cat.ProjectedBox(0.0, 0.0, 0.0, 1.0)
    with pytest.raises(ValueError):
        cat.ProjectedBox(0.0, 1.0, 1.0, 0.0)
    with pytest.raises(ValueError):
        cat.ProjectedBox(float("nan"), 1.0, 0.0, 1.0)


def test_projected_box_area_is_labelled_projected(roi_box):
    assert roi_box.area_projected_m2 == pytest.approx(21000.0 * 21000.0)
    # and the true surface area is smaller by k**2 at the ROI latitude
    k2 = g.area_scale_factor(-86.0)
    assert roi_box.area_projected_m2 / k2 < roi_box.area_projected_m2


def test_selection_on_an_empty_frame_is_empty_not_an_error(tmp_path, roi_box):
    df = cat.normalise_catalogue(cat.load_robbins_csv(write_csv(tmp_path, [_record()])))
    empty = df.iloc[0:0]
    sel = cat.select_in_projected_box(empty, roi_box)
    assert len(sel) == 0


def test_select_requires_normalised_frame(tmp_path, roi_box):
    df = cat.load_robbins_csv(write_csv(tmp_path, [_record()]))
    with pytest.raises(cat.SchemaMismatch):
        cat.select_in_projected_box(df, roi_box)


# --------------------------------------------------------------------------- #
# Completeness constants are a range, not a single invented number
# --------------------------------------------------------------------------- #
def test_completeness_limit_is_a_range_from_the_archive():
    lo, hi = cat.COMPLETENESS_LIMIT_M
    assert (lo, hi) == (1000.0, 2000.0)
    approved_lo, approved_hi = cat.APPROVED_DIAMETER_RANGE_M
    # The whole approved range sits at or below the optimistic completeness
    # limit: the catalogue cannot be ground truth for this project.
    assert approved_hi <= lo
