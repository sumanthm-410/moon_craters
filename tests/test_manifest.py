"""Tests for the provenance manifest.

The centre of gravity here is the unknown sentinel.  A manifest that turns "the
archive did not tell us the incidence angle" into ``0.0`` is worse than useless:
0 deg incidence is a real, physically meaningful value (sun at the zenith), and
at the ROI latitude of ~86 S it is one that cannot occur.  So the round-trip
tests below assert that unknown stays unknown *and* that a genuine zero stays a
genuine zero, in the same file, in the same column.
"""
from __future__ import annotations

import csv
import math

import pytest

from crater import manifest as mf

PRESIGNED = (
    "https://asc-pds-services.s3.us-west-2.amazonaws.com/Moon/x.tif"
    "?X-Amz-Signature=5d41402abc4b2a76b9719d911017c592&versionId=4"
)


def make_entry(**overrides) -> mf.ManifestEntry:
    """A plausible, valid, fully-specified row; override what a test is about."""
    row = dict(
        product_id="LRO_WAC_GLOBAL_303PPD_V3",
        observation_id="UNKNOWN",
        processing_level="mosaic_rdr",
        source_url="https://asc-pds-services.s3.us-west-2.amazonaws.com/Moon/wac.tif",
        metadata_url="https://asc-pds-services.s3.us-west-2.amazonaws.com/Moon/wac.xml",
        acquisition_time="UNKNOWN",
        nominal_resolution_m=100.0,
        incidence_deg=78.4,
        emission_deg=3.1,
        phase_deg=81.2,
        footprint_reference="global",
        file_size_bytes=123456789,
        checksum="sha256:" + "ab" * 32,
        local_path="data/raw/wac.tif",
        download_status="validated",
        roi_overlap=1.0,
        quality_notes="ok: exists, not_error_document, size, checksum, raster_opens",
    )
    row.update(overrides)
    return mf.ManifestEntry(**row)


# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
def test_columns_are_exactly_the_agreed_seventeen():
    assert mf.MANIFEST_COLUMNS == (
        "product_id", "observation_id", "processing_level", "source_url",
        "metadata_url", "acquisition_time", "nominal_resolution_m", "incidence_deg",
        "emission_deg", "phase_deg", "footprint_reference", "file_size_bytes",
        "checksum", "local_path", "download_status", "roi_overlap", "quality_notes",
    )
    assert len(mf.MANIFEST_COLUMNS) == len(set(mf.MANIFEST_COLUMNS)) == 17


def test_header_is_written_in_schema_order(tmp_path):
    path = mf.write_manifest([make_entry()], tmp_path / "manifest.csv")
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert header == ",".join(mf.MANIFEST_COLUMNS)


def test_reading_a_manifest_with_a_missing_column_raises(tmp_path):
    path = tmp_path / "manifest.csv"
    columns = [c for c in mf.MANIFEST_COLUMNS if c != "checksum"]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerow(["x"] * len(columns))
    with pytest.raises(mf.ManifestSchemaError, match="checksum"):
        mf.read_manifest(path)


def test_reading_a_manifest_with_an_extra_column_raises(tmp_path):
    path = tmp_path / "manifest.csv"
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(list(mf.MANIFEST_COLUMNS) + ["sneaky_extra"])
        writer.writerow(["x"] * (len(mf.MANIFEST_COLUMNS) + 1))
    with pytest.raises(mf.ManifestSchemaError, match="sneaky_extra"):
        mf.read_manifest(path)


def test_reading_a_reordered_header_raises(tmp_path):
    path = tmp_path / "manifest.csv"
    reordered = list(mf.MANIFEST_COLUMNS)
    reordered[0], reordered[1] = reordered[1], reordered[0]
    with open(path, "w", newline="", encoding="utf-8") as handle:
        csv.writer(handle).writerow(reordered)
    with pytest.raises(mf.ManifestSchemaError, match="wrong order"):
        mf.read_manifest(path)


def test_empty_manifest_round_trips_as_header_only(tmp_path):
    path = mf.write_manifest([], tmp_path / "manifest.csv")
    assert path.read_text(encoding="utf-8").strip() == ",".join(mf.MANIFEST_COLUMNS)
    assert mf.read_manifest(path) == []
    assert mf.summarise_manifest(path).n_entries == 0


def test_from_row_rejects_unknown_columns():
    with pytest.raises(mf.ManifestSchemaError, match="sneaky"):
        mf.ManifestEntry.from_row({"product_id": "x", "sneaky": "1"})


# --------------------------------------------------------------------------- #
# The unknown sentinel
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("value", [None, "", "   ", "UNKNOWN", "unknown", "nan",
                                   "NaN", "None", "null", "N/A", "-", float("nan")])
def test_is_unknown_recognises_every_spelling_of_nothing(value):
    assert mf.is_unknown(value)


@pytest.mark.parametrize("value", [0, 0.0, "0", "0.0", "-0.0", 1e-12, "false", "0 deg"])
def test_is_unknown_never_claims_a_real_value_is_unknown(value):
    assert not mf.is_unknown(value)


@pytest.mark.parametrize("value,expected", [
    (None, "UNKNOWN"), ("", "UNKNOWN"), (float("nan"), "UNKNOWN"),
    (0, "0"), (0.0, "0.0"), (-0.0, "-0.0"), (0.1, "0.1"), (1e-9, "1e-09"),
    (123456789, "123456789"), (100.0, "100.0"), ("  text  ", "text"),
])
def test_format_cell(value, expected):
    assert mf.format_cell(value) == expected


def test_format_cell_refuses_infinity():
    with pytest.raises(mf.ManifestValueError, match="non-finite"):
        mf.format_cell(float("inf"), column="incidence_deg")


def test_as_float_returns_none_for_unknown_and_never_coerces_junk():
    assert mf.as_float(mf.UNKNOWN) is None
    assert mf.as_float("") is None
    assert mf.as_float("nan") is None          # NOT float('nan')
    assert mf.as_float("0.0") == 0.0
    assert mf.as_float("0") == 0.0
    with pytest.raises(mf.ManifestValueError, match="neither a number"):
        mf.as_float("about 40 deg", column="incidence_deg")


def test_as_int_rejects_fractions():
    assert mf.as_int("12") == 12
    assert mf.as_int(mf.UNKNOWN) is None
    with pytest.raises(mf.ManifestValueError, match="whole number"):
        mf.as_int("12.5", column="file_size_bytes")


def test_unknown_round_trips_distinctly_from_a_real_zero(tmp_path):
    """The central requirement: UNKNOWN and 0.0 must never collapse together."""
    unknown_row = make_entry(
        product_id="UNKNOWN_GEOMETRY",
        incidence_deg=None,            # the archive did not say
        emission_deg=float("nan"),     # a NaN that must not become a number
        phase_deg="",                   # an empty cell upstream
        nominal_resolution_m=mf.UNKNOWN,
        file_size_bytes=None,
        roi_overlap=None,
        download_status="pending",
        quality_notes="geometry not published by the archive",
    )
    zero_row = make_entry(
        product_id="GENUINE_ZEROS",
        incidence_deg=0.0,             # a real measured zero
        emission_deg=0,
        phase_deg=0.0,
        nominal_resolution_m=0.0,
        file_size_bytes=0,
        roi_overlap=0.0,
        download_status="pending",
        quality_notes="all-zero geometry, deliberately",
    )

    path = mf.write_manifest([unknown_row, zero_row], tmp_path / "manifest.csv")
    text = path.read_text(encoding="utf-8")
    assert "UNKNOWN" in text
    assert ",,"  not in text, "an unknown must never be written as an empty cell"
    assert "nan" not in text.lower().replace("unknown", "")

    back = mf.read_manifest(path)
    assert back == [unknown_row, zero_row]
    unknown_back, zero_back = back

    for column in ("incidence_deg", "emission_deg", "phase_deg",
                   "nominal_resolution_m", "file_size_bytes", "roi_overlap"):
        assert unknown_back.cell(column) == mf.UNKNOWN, column
        assert unknown_back.number(column) is None, column
        assert not unknown_back.known(column), column

        assert zero_back.cell(column) in ("0", "0.0"), column
        assert zero_back.number(column) == 0.0, column
        assert zero_back.known(column), column

    # And the distinction is still there in the typed accessors.
    assert unknown_back.incidence is None and zero_back.incidence == 0.0
    assert unknown_back.size_bytes is None and zero_back.size_bytes == 0
    # The failure this guards against: None and 0.0 comparing equal.
    assert unknown_back.incidence != zero_back.incidence


def test_round_trip_preserves_float_text_exactly(tmp_path):
    entry = make_entry(nominal_resolution_m=0.1, incidence_deg=1e-9,
                       emission_deg=87.65432109876543, phase_deg=180.0)
    path = mf.write_manifest([entry], tmp_path / "manifest.csv")
    back = mf.read_manifest(path)[0]
    assert back == entry
    assert back.resolution_m == 0.1
    assert back.number("incidence_deg") == 1e-9
    assert back.number("emission_deg") == 87.65432109876543


def test_round_trip_preserves_awkward_text(tmp_path):
    notes = 'rejected: HTML error page, "403 Forbidden", comma, and\na newline'
    entry = make_entry(
        download_status="rejected", quality_notes=notes,
        local_path="data/raw/with space/M1 LE.IMG",
    )
    path = mf.write_manifest([entry], tmp_path / "manifest.csv")
    back = mf.read_manifest(path)[0]
    assert back.quality_notes == notes
    assert back.local_path == "data/raw/with space/M1 LE.IMG"
    assert back == entry


def test_unknown_is_not_coerced_by_the_pandas_view(tmp_path):
    """A frame view must not reintroduce NaN -- that is the 0.0 bug's entry point."""
    entry = make_entry(incidence_deg=None, emission_deg=0.0, download_status="pending")
    path = mf.write_manifest([entry, make_entry(product_id="OTHER")],
                             tmp_path / "manifest.csv")
    frame = mf.read_manifest_frame(path)
    assert tuple(frame.columns) == mf.MANIFEST_COLUMNS
    assert frame["incidence_deg"][0] == mf.UNKNOWN
    assert frame["emission_deg"][0] == "0.0"
    assert not frame.isna().any().any(), "no cell may read back as NaN"
    # Every cell is text, so no silent arithmetic on a sentinel is possible.
    assert all(isinstance(v, str) for v in frame["incidence_deg"])


def test_entry_normalises_on_construction():
    entry = mf.ManifestEntry(product_id="P", file_size_bytes=12, incidence_deg=None)
    assert entry.file_size_bytes == "12"
    assert entry.incidence_deg == mf.UNKNOWN
    assert entry.download_status == "pending"  # the default, not UNKNOWN
    assert mf.ManifestEntry(download_status=None).download_status == "pending"


def test_with_values_returns_a_normalised_copy():
    entry = make_entry()
    updated = entry.with_values(download_status="rejected", quality_notes="checksum mismatch")
    assert updated.download_status == "rejected"
    assert entry.download_status == "validated"  # frozen, untouched
    with pytest.raises(KeyError, match="not manifest columns"):
        entry.with_values(nope=1)


def test_urls_are_stored_redacted(tmp_path):
    entry = make_entry(source_url=PRESIGNED, metadata_url=PRESIGNED)
    assert "5d41402abc4b2a76b9719d911017c592" not in entry.source_url
    assert "REDACTED" in entry.source_url
    assert "versionId=4" in entry.source_url
    path = mf.write_manifest([entry], tmp_path / "manifest.csv")
    assert "5d41402abc4b2a76b9719d911017c592" not in path.read_text(encoding="utf-8")


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_a_complete_manifest_validates(tmp_path):
    path = mf.write_manifest(
        [make_entry(), make_entry(product_id="OTHER", download_status="pending",
                                  quality_notes=mf.UNKNOWN)],
        tmp_path / "manifest.csv",
    )
    report = mf.validate_manifest(path)
    assert report.ok
    assert report.n_entries == 2
    assert not report.errors


def test_status_outside_the_vocabulary_is_an_error():
    report = mf.validate_manifest([make_entry(download_status="downloaded-ish")])
    assert not report.ok
    assert any(i.column == "download_status" for i in report.errors)
    assert "vocabulary" in report.errors[0].message


@pytest.mark.parametrize("status", mf.DOWNLOAD_STATUS_VOCABULARY)
def test_every_vocabulary_status_is_accepted(status):
    report = mf.validate_manifest([make_entry(download_status=status)])
    assert not [i for i in report.errors if i.column == "download_status"]


@pytest.mark.parametrize("status", sorted(mf.REJECTED_STATUSES))
def test_a_rejected_entry_must_carry_a_reason(status):
    report = mf.validate_manifest(
        [make_entry(download_status=status, quality_notes=None)]
    )
    assert not report.ok
    assert any(i.column == "quality_notes" and "must record a reason" in i.message
               for i in report.errors)


@pytest.mark.parametrize("status", sorted(mf.ACCEPTED_STATUSES))
def test_an_accepted_entry_must_carry_a_reason_too(status):
    report = mf.validate_manifest(
        [make_entry(download_status=status, quality_notes=mf.UNKNOWN)]
    )
    assert not report.ok
    assert any("accepted" in i.message for i in report.errors)


def test_a_pending_entry_needs_no_reason():
    report = mf.validate_manifest([make_entry(download_status="pending",
                                              quality_notes=None)])
    assert report.ok


def test_accepted_entry_without_local_path_is_an_error():
    report = mf.validate_manifest([make_entry(local_path=None)])
    assert not report.ok
    assert any(i.column == "local_path" for i in report.errors)


def test_accepted_entry_without_checksum_is_a_warning_not_an_error():
    report = mf.validate_manifest([make_entry(checksum=None)])
    assert report.ok
    assert any(i.column == "checksum" and i.severity == mf.SEVERITY_WARNING
               for i in report.warnings)


def test_duplicate_and_missing_product_ids_are_errors():
    report = mf.validate_manifest([make_entry(), make_entry(), make_entry(product_id=None)])
    columns = [i.column for i in report.errors]
    assert columns.count("product_id") == 2
    assert any("duplicate" in i.message for i in report.errors)


def test_non_numeric_text_in_a_numeric_column_is_an_error():
    report = mf.validate_manifest([make_entry(incidence_deg="about 78")])
    assert not report.ok
    assert any(i.column == "incidence_deg" for i in report.errors)


@pytest.mark.parametrize("value", [-0.5, 1.5, 2.0])
def test_roi_overlap_outside_zero_to_one_is_an_error(value):
    report = mf.validate_manifest([make_entry(roi_overlap=value)])
    assert not report.ok
    assert any(i.column == "roi_overlap" and "fraction" in i.message for i in report.errors)


def test_roi_overlap_may_be_unknown_while_the_roi_is_deferred():
    """D-002 leaves the ROI centre unset, so overlap is legitimately unknown."""
    report = mf.validate_manifest([make_entry(roi_overlap=None)])
    assert report.ok


@pytest.mark.parametrize("column,value", [
    ("incidence_deg", 190.0), ("emission_deg", -1.0), ("phase_deg", 360.0),
    ("file_size_bytes", -1), ("nominal_resolution_m", -0.5),
])
def test_physically_impossible_numbers_are_errors(column, value):
    report = mf.validate_manifest([make_entry(**{column: value})])
    assert not report.ok
    assert any(i.column == column for i in report.errors)


def test_validation_report_is_readable():
    report = mf.validate_manifest([make_entry(download_status="rejected",
                                              quality_notes=None)])
    text = report.report()
    assert "quality_notes" in text and "error" in text
    assert str(report.errors[0]).startswith("[error] row 1")


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
def test_summary_counts_by_status_and_processing_level(tmp_path):
    entries = [
        make_entry(product_id="A", download_status="validated", processing_level="edr",
                   file_size_bytes=100),
        make_entry(product_id="B", download_status="validated", processing_level="cdr",
                   file_size_bytes=200),
        make_entry(product_id="C", download_status="rejected", processing_level="edr",
                   quality_notes="HTML error page", file_size_bytes=300),
        make_entry(product_id="D", download_status="pending", processing_level="edr",
                   quality_notes=None, file_size_bytes=None),
    ]
    summary = mf.summarise_manifest(entries)
    assert summary.n_entries == 4
    assert summary.by_status == {"validated": 2, "rejected": 1, "pending": 1}
    assert summary.by_processing_level == {"edr": 3, "cdr": 1}
    assert summary.by_status_and_level[("validated", "edr")] == 1
    assert summary.n_accepted == 2
    assert summary.n_rejected == 1


def test_summary_never_invents_a_total_size(tmp_path):
    """With an unknown size present, no total exists; the summary must say so."""
    entries = [
        make_entry(product_id="A", file_size_bytes=1000),
        make_entry(product_id="B", file_size_bytes=None, download_status="pending",
                   quality_notes=None),
    ]
    summary = mf.summarise_manifest(entries)
    assert summary.known_bytes == 1000
    assert summary.rows_with_unknown_size == 1
    assert not hasattr(summary, "total_bytes")
    text = summary.describe()
    assert "no total size can be stated" in text
    assert "1 row(s) have UNKNOWN size" in text


def test_summary_counts_unknown_cells_per_column():
    summary = mf.summarise_manifest([make_entry(acquisition_time=None,
                                                observation_id=None)])
    assert summary.unknown_cells_by_column["acquisition_time"] == 1
    assert summary.unknown_cells_by_column["observation_id"] == 1
    assert "incidence_deg" not in summary.unknown_cells_by_column


def test_summary_reports_a_total_when_every_size_is_known():
    summary = mf.summarise_manifest([make_entry(file_size_bytes=7),
                                     make_entry(product_id="B", file_size_bytes=5)])
    assert summary.known_bytes == 12
    assert summary.rows_with_unknown_size == 0
    assert "total bytes: 12" in summary.describe()


# --------------------------------------------------------------------------- #
# Interoperability with crater.download
# --------------------------------------------------------------------------- #
def test_entry_from_download_records_only_what_is_known(tmp_path):
    from crater import download as dl

    path = tmp_path / "p.tif"
    path.write_bytes(b"\x1f\x8b" + b"\x00" * 32)  # gzip magic: not an error page
    validation = dl.validate_raster_file(path, expect_raster=False)
    outcome = dl.DownloadOutcome(
        path=path, url=dl.redact_url(PRESIGNED), status=dl.STATUS_DOWNLOADED,
        bytes_on_disk=path.stat().st_size, bytes_transferred=path.stat().st_size,
        resumed=False, attempts=(), validation=validation, server_supports_range=True,
    )
    entry = mf.entry_from_download("PROD_1", outcome)
    assert entry.download_status == "validated"
    assert entry.size_bytes == path.stat().st_size
    assert entry.local_path == str(path)
    assert "5d41402abc4b2a76b9719d911017c592" not in entry.source_url
    # Nothing was supplied about the observation, so nothing is invented.
    assert entry.observation_id == mf.UNKNOWN
    assert entry.incidence_deg == mf.UNKNOWN
    assert entry.acquisition_time == mf.UNKNOWN
    assert mf.validate_manifest([entry]).ok


def test_entry_from_download_marks_a_failed_validation_as_rejected(tmp_path):
    from crater import download as dl

    path = tmp_path / "p.IMG"
    path.write_bytes(b"<html><head><title>403 Forbidden</title></head></html>")
    validation = dl.validate_raster_file(path)
    outcome = dl.DownloadOutcome(
        path=path, url="https://pds.lroc.asu.edu/EDR/p.IMG", status="downloaded",
        bytes_on_disk=path.stat().st_size, bytes_transferred=0, resumed=False,
        attempts=(), validation=validation, server_supports_range=None,
    )
    entry = mf.entry_from_download("PROD_2", outcome)
    assert entry.download_status == "rejected"
    assert "not_error_document failed" in entry.quality_notes
    assert mf.validate_manifest([entry]).ok  # a rejection with a reason is valid


def test_entry_from_download_without_validation_is_not_claimed_validated(tmp_path):
    from crater import download as dl

    outcome = dl.DownloadOutcome(
        path=tmp_path / "p.tif", url="https://example.invalid/p.tif",
        status=dl.STATUS_DOWNLOADED, bytes_on_disk=10, bytes_transferred=10,
        resumed=False, attempts=(), validation=None, server_supports_range=None,
        note="validation was switched off",
    )
    entry = mf.entry_from_download("PROD_3", outcome)
    assert entry.download_status == "downloaded"
    assert entry.download_status != "validated"
    assert entry.quality_notes == "validation was switched off"


# --------------------------------------------------------------------------- #
# Atomic write
# --------------------------------------------------------------------------- #
def test_write_is_atomic_and_leaves_no_temp_file(tmp_path):
    target = tmp_path / "manifest.csv"
    mf.write_manifest([make_entry()], target)
    mf.write_manifest([make_entry(product_id="SECOND")], target)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["manifest.csv"]
    assert mf.read_manifest(target)[0].product_id == "SECOND"


def test_write_creates_missing_parent_directories(tmp_path):
    target = tmp_path / "data" / "raw" / "manifest.csv"
    mf.write_manifest([make_entry()], target)
    assert target.is_file()


def test_manifest_filename_default():
    assert mf.MANIFEST_FILENAME == "manifest.csv"
