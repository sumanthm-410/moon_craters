"""The provenance manifest: one row per archive product, unknowns kept unknown.

Every product that enters this project is recorded in ``manifest.csv`` with the
columns in :data:`MANIFEST_COLUMNS`, in that exact order.  The manifest is the
audit trail that lets a reader of the final size-frequency distribution trace a
measured crater back to a specific observation, its illumination geometry and
its calibration level.

Why everything is stored as text
-------------------------------
The manifest's hardest requirement is that *unknown must stay unknown*.  The
obvious implementation -- a ``pandas`` frame with float columns -- fails it
silently in three different ways:

1. an empty cell is read as ``NaN`` and then formatted/aggregated into ``0.0``
   by a downstream ``fillna(0)`` or ``sum()``;
2. a column holding both numbers and a sentinel becomes ``object`` dtype, and
   ``astype(float)`` on it turns ``"UNKNOWN"`` into an exception or, worse,
   ``errors="coerce"`` turns it into ``NaN`` and then into a number;
3. ``NaN`` is indistinguishable, once written back out, from "the archive said
   zero" -- and an incidence angle of 0 deg is a real, physically meaningful
   value that must never be confused with "we do not know it".

So each cell is held as the exact text that will be written to the file, with
:data:`UNKNOWN` as the one and only sentinel, and numbers are obtained through
the explicit accessors :func:`as_float` / :func:`as_int`, which return ``None``
for unknown.  A caller then *cannot* accidentally average an unknown into a
number: it has to decide what ``None`` means.

Round-tripping is therefore byte-exact: ``read_manifest(write_manifest(x)) == x``.
"""
from __future__ import annotations

import csv
import math
import os
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .download import redact_url

# --------------------------------------------------------------------------- #
# Schema
# --------------------------------------------------------------------------- #
#: The manifest's columns, in file order.  Frozen: downstream owners read this.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "product_id",
    "observation_id",
    "processing_level",
    "source_url",
    "metadata_url",
    "acquisition_time",
    "nominal_resolution_m",
    "incidence_deg",
    "emission_deg",
    "phase_deg",
    "footprint_reference",
    "file_size_bytes",
    "checksum",
    "local_path",
    "download_status",
    "roi_overlap",
    "quality_notes",
)

#: The explicit sentinel for "this metadata is not known".  Never "", never NaN.
UNKNOWN = "UNKNOWN"

#: Spellings that mean unknown on input.  ``"nan"``/``"none"`` are included
#: because a careless ``str(float('nan'))`` or ``str(None)`` upstream must land
#: as unknown rather than as a number or a literal string.
_UNKNOWN_SPELLINGS: frozenset[str] = frozenset(
    {"", "unknown", "nan", "none", "null", "n/a", "na", "-", "?"}
)

#: Columns that must hold a real number or :data:`UNKNOWN`.
NUMERIC_COLUMNS: tuple[str, ...] = (
    "nominal_resolution_m", "incidence_deg", "emission_deg", "phase_deg",
    "file_size_bytes", "roi_overlap",
)

#: Columns that must hold a whole number or :data:`UNKNOWN`.
INTEGER_COLUMNS: tuple[str, ...] = ("file_size_bytes",)

#: URL columns.  Written redacted, so a presigned archive URL never lands in a
#: file that will be committed or shared.
URL_COLUMNS: tuple[str, ...] = ("source_url", "metadata_url")

#: The fixed vocabulary for ``download_status``.
DOWNLOAD_STATUS_VOCABULARY: tuple[str, ...] = (
    "pending",      # row created, nothing attempted yet
    "queued",       # selected for download
    "downloading",  # in flight (a crash leaves this behind; resume handles it)
    "downloaded",   # bytes on disk, not yet content-validated
    "validated",    # bytes on disk and content validation passed
    "rejected",     # content arrived but failed validation -- reason required
    "failed",       # transfer could not be completed -- reason required
    "unavailable",  # host denied / product does not exist -- reason required
    "skipped",      # deliberately not fetched -- reason required
)

#: Statuses that assert we hold a usable file.
ACCEPTED_STATUSES: frozenset[str] = frozenset({"downloaded", "validated"})

#: Terminal negative statuses.
REJECTED_STATUSES: frozenset[str] = frozenset({"rejected", "failed", "unavailable", "skipped"})

#: Statuses that must carry a human-readable reason in ``quality_notes``.
#: Accepted rows are included deliberately: "why do we trust this file" is as
#: much a part of the audit trail as "why did we drop that one".
REASON_REQUIRED_STATUSES: frozenset[str] = ACCEPTED_STATUSES | REJECTED_STATUSES

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


class ManifestError(Exception):
    """Base class for manifest problems."""


class ManifestSchemaError(ManifestError):
    """The file's columns are not exactly :data:`MANIFEST_COLUMNS`."""


class ManifestValueError(ManifestError):
    """A cell holds something that is neither a number nor the sentinel."""


# --------------------------------------------------------------------------- #
# Cell formatting and typed access
# --------------------------------------------------------------------------- #
def is_unknown(value: Any) -> bool:
    """True when ``value`` means "not known".

    ``None``, an empty/whitespace string, any spelling in
    :data:`_UNKNOWN_SPELLINGS`, and a float ``NaN`` all mean unknown.  A real
    ``0``, ``0.0`` or ``"0.0"`` does **not**.
    """
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str):
        return value.strip().lower() in _UNKNOWN_SPELLINGS
    return False


def format_cell(value: Any, *, column: str | None = None) -> str:
    """Canonical text for one manifest cell.

    Unknowns become :data:`UNKNOWN`.  Floats use ``repr``, which is the
    shortest text that round-trips exactly, so ``0.1`` comes back as ``0.1``
    and not as ``0.1000000000000000055``.  Infinities are rejected rather than
    written as a token that would parse back as a number.
    """
    if is_unknown(value):
        return UNKNOWN
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if math.isinf(value):
            raise ManifestValueError(
                f"refusing to write a non-finite value for {column or 'cell'}: {value!r}"
            )
        return repr(value)
    if isinstance(value, int):
        return str(value)
    text = str(value).strip()
    if column in URL_COLUMNS:
        # Never store a credential, even one the caller passed in innocently.
        text = redact_url(text)
    if "\r" in text:
        # csv would write it, but a bare CR inside a quoted field is read back
        # differently by different tools; normalise to \n so round-trip is exact.
        text = text.replace("\r\n", "\n").replace("\r", "\n")
    return text if text else UNKNOWN


def as_float(value: Any, *, column: str | None = None) -> float | None:
    """Parse a cell as a float, returning ``None`` for unknown.

    Raises :class:`ManifestValueError` on text that is neither.  There is
    deliberately no ``errors="coerce"`` path: silently turning junk into
    ``NaN`` (and later into ``0.0``) is the failure mode this module exists to
    prevent.
    """
    if is_unknown(value):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        parsed = float(str(value).strip())
    except (TypeError, ValueError) as exc:
        raise ManifestValueError(
            f"{column or 'cell'}: {value!r} is neither a number nor {UNKNOWN}"
        ) from exc
    if math.isnan(parsed) or math.isinf(parsed):
        raise ManifestValueError(
            f"{column or 'cell'}: {value!r} parses to a non-finite number; "
            f"write {UNKNOWN} instead"
        )
    return parsed


def as_int(value: Any, *, column: str | None = None) -> int | None:
    """Parse a cell as an integer, returning ``None`` for unknown."""
    parsed = as_float(value, column=column)
    if parsed is None:
        return None
    if parsed != int(parsed):
        raise ManifestValueError(
            f"{column or 'cell'}: {value!r} is not a whole number"
        )
    return int(parsed)


# --------------------------------------------------------------------------- #
# Entries
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ManifestEntry:
    """One manifest row.

    Every field is the *text* that will appear in the file; the constructor
    normalises whatever it is given (``None``, ``float``, ``int``, ``str``,
    ``NaN``) through :func:`format_cell`.  Use :meth:`number` or the named
    accessors to get numbers back.
    """

    product_id: str = UNKNOWN
    observation_id: str = UNKNOWN
    processing_level: str = UNKNOWN
    source_url: str = UNKNOWN
    metadata_url: str = UNKNOWN
    acquisition_time: str = UNKNOWN
    nominal_resolution_m: str = UNKNOWN
    incidence_deg: str = UNKNOWN
    emission_deg: str = UNKNOWN
    phase_deg: str = UNKNOWN
    footprint_reference: str = UNKNOWN
    file_size_bytes: str = UNKNOWN
    checksum: str = UNKNOWN
    local_path: str = UNKNOWN
    download_status: str = "pending"
    roi_overlap: str = UNKNOWN
    quality_notes: str = UNKNOWN

    def __post_init__(self) -> None:
        for column in MANIFEST_COLUMNS:
            object.__setattr__(
                self, column, format_cell(getattr(self, column), column=column)
            )
        if self.download_status == UNKNOWN:
            object.__setattr__(self, "download_status", "pending")

    # -- access ---------------------------------------------------------- #
    def cell(self, column: str) -> str:
        if column not in MANIFEST_COLUMNS:
            raise KeyError(f"{column!r} is not a manifest column")
        return getattr(self, column)

    def known(self, column: str) -> bool:
        """True when ``column`` holds a real value rather than the sentinel."""
        return not is_unknown(self.cell(column))

    def number(self, column: str) -> float | None:
        """Numeric view of ``column``: ``None`` when unknown."""
        if column in INTEGER_COLUMNS:
            value = as_int(self.cell(column), column=column)
            return None if value is None else float(value)
        return as_float(self.cell(column), column=column)

    @property
    def size_bytes(self) -> int | None:
        return as_int(self.file_size_bytes, column="file_size_bytes")

    @property
    def resolution_m(self) -> float | None:
        return as_float(self.nominal_resolution_m, column="nominal_resolution_m")

    @property
    def incidence(self) -> float | None:
        return as_float(self.incidence_deg, column="incidence_deg")

    @property
    def emission(self) -> float | None:
        return as_float(self.emission_deg, column="emission_deg")

    @property
    def phase(self) -> float | None:
        return as_float(self.phase_deg, column="phase_deg")

    @property
    def roi_overlap_fraction(self) -> float | None:
        return as_float(self.roi_overlap, column="roi_overlap")

    def to_row(self) -> dict[str, str]:
        """The row as an ordered mapping of column -> exact text."""
        return {column: getattr(self, column) for column in MANIFEST_COLUMNS}

    def with_values(self, **updates: Any) -> "ManifestEntry":
        """A copy with ``updates`` applied (and re-normalised)."""
        unknown_columns = set(updates) - set(MANIFEST_COLUMNS)
        if unknown_columns:
            raise KeyError(f"not manifest columns: {sorted(unknown_columns)}")
        row = self.to_row()
        row.update(updates)
        return ManifestEntry(**row)

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "ManifestEntry":
        """Build an entry from a mapping, requiring exactly the known columns.

        Extra keys are an error: a manifest with a stray column is a schema
        drift that would be silently dropped otherwise.
        """
        extra = set(row) - set(MANIFEST_COLUMNS)
        if extra:
            raise ManifestSchemaError(f"unexpected manifest columns: {sorted(extra)}")
        return cls(**{column: row.get(column, UNKNOWN) for column in MANIFEST_COLUMNS})


def entry_from_download(
    product_id: str,
    outcome: Any,
    *,
    observation_id: Any = None,
    processing_level: Any = None,
    metadata_url: Any = None,
    acquisition_time: Any = None,
    nominal_resolution_m: Any = None,
    incidence_deg: Any = None,
    emission_deg: Any = None,
    phase_deg: Any = None,
    footprint_reference: Any = None,
    checksum: Any = None,
    roi_overlap: Any = None,
    checksum_algorithm: str = "sha256",
) -> ManifestEntry:
    """Build a row from a :class:`crater.download.DownloadOutcome`.

    Only facts the outcome actually carries are recorded; everything the caller
    did not supply stays :data:`UNKNOWN`.  ``download_status`` is ``validated``
    only when a validation result exists and passed -- a download with
    validation switched off records ``downloaded``, not ``validated``.
    """
    validation = getattr(outcome, "validation", None)
    if validation is None:
        status = "downloaded"
        notes = getattr(outcome, "note", None) or "downloaded without content validation"
    elif validation.ok:
        status = "validated"
        notes = validation.reason()
    else:
        status = "rejected"
        notes = validation.reason()
    return ManifestEntry(
        product_id=product_id,
        observation_id=observation_id,
        processing_level=processing_level,
        source_url=getattr(outcome, "url", None),
        metadata_url=metadata_url,
        acquisition_time=acquisition_time,
        nominal_resolution_m=nominal_resolution_m,
        incidence_deg=incidence_deg,
        emission_deg=emission_deg,
        phase_deg=phase_deg,
        footprint_reference=footprint_reference,
        file_size_bytes=getattr(outcome, "bytes_on_disk", None),
        checksum=checksum,
        local_path=str(getattr(outcome, "path", "")) or None,
        download_status=status,
        roi_overlap=roi_overlap,
        quality_notes=notes,
    )


# --------------------------------------------------------------------------- #
# Read / write
# --------------------------------------------------------------------------- #
#: Default manifest file name, relative to a run's data directory.
MANIFEST_FILENAME = "manifest.csv"


def write_manifest(entries: Iterable[ManifestEntry], path: str | os.PathLike[str]) -> Path:
    """Write ``entries`` to ``path`` atomically, header first.

    The write goes to a sibling temp file which is fsynced and renamed, so a
    crash never leaves a half-written manifest -- the audit trail is either the
    old one or the new one.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(MANIFEST_COLUMNS), lineterminator="\n",
            extrasaction="raise",
        )
        writer.writeheader()
        for entry in entries:
            writer.writerow(entry.to_row())
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, target)
    return target


def read_manifest(path: str | os.PathLike[str]) -> list[ManifestEntry]:
    """Read a manifest.

    Nothing is type-coerced on the way in: cells arrive as the text that was
    written, so ``UNKNOWN`` stays ``UNKNOWN`` and ``0.0`` stays ``0.0``.

    Raises
    ------
    ManifestSchemaError
        If the header is not exactly :data:`MANIFEST_COLUMNS` in order, or a
        row has the wrong number of fields.
    """
    target = Path(path)
    with open(target, "r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        try:
            header = next(reader)
        except StopIteration:
            raise ManifestSchemaError(f"{target} is empty; expected a header row") from None
        if tuple(header) != MANIFEST_COLUMNS:
            missing = [c for c in MANIFEST_COLUMNS if c not in header]
            extra = [c for c in header if c not in MANIFEST_COLUMNS]
            detail = (
                f"missing={missing} unexpected={extra}"
                if (missing or extra)
                else f"same columns in the wrong order: {tuple(header)}"
            )
            raise ManifestSchemaError(
                f"{target} header is not the manifest schema; {detail}"
            )
        entries: list[ManifestEntry] = []
        for line_number, row in enumerate(reader, start=2):
            if not row:
                continue  # a trailing blank line is not a row
            if len(row) != len(MANIFEST_COLUMNS):
                raise ManifestSchemaError(
                    f"{target}:{line_number} has {len(row)} fields, "
                    f"expected {len(MANIFEST_COLUMNS)}"
                )
            entries.append(ManifestEntry.from_row(dict(zip(MANIFEST_COLUMNS, row))))
    return entries


def read_manifest_frame(path: str | os.PathLike[str]):
    """Read a manifest into a ``pandas`` frame of **strings**.

    Provided for downstream owners who want a frame.  Every column is read as
    text with NA detection switched off, so ``UNKNOWN`` survives and no cell is
    ever turned into ``NaN`` (and from there into a number).  Convert
    explicitly with :func:`as_float` when you need numbers.
    """
    import pandas as pd

    frame = pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
    )
    if tuple(frame.columns) != MANIFEST_COLUMNS:
        raise ManifestSchemaError(
            f"{path} header is not the manifest schema: {tuple(frame.columns)}"
        )
    return frame


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ManifestIssue:
    """One problem found in a manifest."""

    row: int | None
    product_id: str
    column: str
    severity: str
    message: str

    def __str__(self) -> str:
        where = "header" if self.row is None else f"row {self.row}"
        return f"[{self.severity}] {where} {self.product_id}.{self.column}: {self.message}"


@dataclass(frozen=True)
class ManifestValidation:
    """Outcome of :func:`validate_manifest`."""

    issues: tuple[ManifestIssue, ...]
    n_entries: int

    @property
    def ok(self) -> bool:
        return not self.errors

    @property
    def errors(self) -> tuple[ManifestIssue, ...]:
        return tuple(i for i in self.issues if i.severity == SEVERITY_ERROR)

    @property
    def warnings(self) -> tuple[ManifestIssue, ...]:
        return tuple(i for i in self.issues if i.severity == SEVERITY_WARNING)

    def report(self) -> str:
        if not self.issues:
            return f"manifest valid: {self.n_entries} entries, no issues"
        return "\n".join(str(issue) for issue in self.issues)


def validate_manifest(
    source: Iterable[ManifestEntry] | str | os.PathLike[str],
) -> ManifestValidation:
    """Validate a manifest, by path or as already-read entries.

    Errors (make the manifest invalid):

    * the file's columns are not exactly :data:`MANIFEST_COLUMNS` (raised as
      :class:`ManifestSchemaError` by :func:`read_manifest` before we get here);
    * ``download_status`` outside :data:`DOWNLOAD_STATUS_VOCABULARY`;
    * a terminal row (accepted *or* rejected) with no reason in ``quality_notes``;
    * an accepted row with no ``local_path``;
    * a numeric column holding text that is neither a number nor ``UNKNOWN``;
    * ``roi_overlap`` outside ``[0, 1]``;
    * a missing or duplicated ``product_id``.

    Warnings (recorded, but the manifest is still usable):

    * an accepted row with no checksum -- the archive may genuinely not publish
      one, which is a provenance limitation worth surfacing;
    * an unknown ``source_url`` on a row that claims a local file.
    """
    if isinstance(source, (str, os.PathLike)):
        entries = read_manifest(source)
    else:
        entries = list(source)

    issues: list[ManifestIssue] = []
    seen: dict[str, int] = {}

    for index, entry in enumerate(entries, start=1):
        pid = entry.product_id

        if is_unknown(pid):
            issues.append(ManifestIssue(index, pid, "product_id", SEVERITY_ERROR,
                                        "product_id is required and must not be UNKNOWN"))
        elif pid in seen:
            issues.append(ManifestIssue(index, pid, "product_id", SEVERITY_ERROR,
                                        f"duplicate product_id, first seen at row {seen[pid]}"))
        else:
            seen[pid] = index

        status = entry.download_status
        if status not in DOWNLOAD_STATUS_VOCABULARY:
            issues.append(ManifestIssue(
                index, pid, "download_status", SEVERITY_ERROR,
                f"{status!r} is not in the vocabulary {DOWNLOAD_STATUS_VOCABULARY}",
            ))

        if status in REASON_REQUIRED_STATUSES and not entry.known("quality_notes"):
            kind = "accepted" if status in ACCEPTED_STATUSES else "rejected"
            issues.append(ManifestIssue(
                index, pid, "quality_notes", SEVERITY_ERROR,
                f"an {kind} entry (status {status!r}) must record a reason",
            ))

        if status in ACCEPTED_STATUSES:
            if not entry.known("local_path"):
                issues.append(ManifestIssue(
                    index, pid, "local_path", SEVERITY_ERROR,
                    f"status {status!r} claims a file but local_path is UNKNOWN",
                ))
            if not entry.known("checksum"):
                issues.append(ManifestIssue(
                    index, pid, "checksum", SEVERITY_WARNING,
                    "accepted without a checksum; integrity rests on size and "
                    "raster-readability only",
                ))
            if not entry.known("source_url"):
                issues.append(ManifestIssue(
                    index, pid, "source_url", SEVERITY_WARNING,
                    "accepted entry has no source URL; provenance is incomplete",
                ))

        for column in NUMERIC_COLUMNS:
            try:
                value = entry.number(column)
            except ManifestValueError as exc:
                issues.append(ManifestIssue(index, pid, column, SEVERITY_ERROR, str(exc)))
                continue
            if value is None:
                continue
            if column == "roi_overlap" and not 0.0 <= value <= 1.0:
                issues.append(ManifestIssue(
                    index, pid, column, SEVERITY_ERROR,
                    f"roi_overlap is a fraction in [0, 1], got {value!r}",
                ))
            if column in ("file_size_bytes", "nominal_resolution_m") and value < 0:
                issues.append(ManifestIssue(
                    index, pid, column, SEVERITY_ERROR, f"must not be negative, got {value!r}",
                ))
            if column in ("incidence_deg", "emission_deg", "phase_deg") and not 0.0 <= value <= 180.0:
                issues.append(ManifestIssue(
                    index, pid, column, SEVERITY_ERROR,
                    f"illumination angle outside [0, 180] deg: {value!r}",
                ))

    return ManifestValidation(issues=tuple(issues), n_entries=len(entries))


# --------------------------------------------------------------------------- #
# Summary
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ManifestSummary:
    """Counts by status and processing level, plus an honest byte total.

    ``known_bytes`` sums only the rows whose ``file_size_bytes`` is known, and
    ``rows_with_unknown_size`` says how many were left out.  There is no single
    "total size" field, because with unknown rows present no such number exists
    and reporting one would be a fabrication (INTERFACES.md rule 6).
    """

    n_entries: int
    by_status: dict[str, int]
    by_processing_level: dict[str, int]
    by_status_and_level: dict[tuple[str, str], int]
    n_accepted: int
    n_rejected: int
    known_bytes: int
    rows_with_unknown_size: int
    unknown_cells_by_column: dict[str, int]

    def describe(self) -> str:
        """Multi-line human summary, safe to print in a stage log."""
        lines = [f"{self.n_entries} manifest entries "
                 f"({self.n_accepted} accepted, {self.n_rejected} rejected)"]
        lines.append(
            "  by status: "
            + (", ".join(f"{k}={v}" for k, v in sorted(self.by_status.items())) or "none")
        )
        lines.append(
            "  by processing level: "
            + (", ".join(f"{k}={v}" for k, v in sorted(self.by_processing_level.items()))
               or "none")
        )
        if self.rows_with_unknown_size:
            lines.append(
                f"  known bytes: {self.known_bytes} over "
                f"{self.n_entries - self.rows_with_unknown_size} rows; "
                f"{self.rows_with_unknown_size} row(s) have UNKNOWN size, so no "
                f"total size can be stated"
            )
        else:
            lines.append(f"  total bytes: {self.known_bytes}")
        return "\n".join(lines)


def summarise_manifest(
    source: Iterable[ManifestEntry] | str | os.PathLike[str],
) -> ManifestSummary:
    """Counts by ``download_status`` and ``processing_level``."""
    entries = (
        read_manifest(source) if isinstance(source, (str, os.PathLike)) else list(source)
    )
    by_status: Counter[str] = Counter()
    by_level: Counter[str] = Counter()
    by_both: Counter[tuple[str, str]] = Counter()
    unknown_cells: Counter[str] = Counter()
    known_bytes = 0
    unknown_size = 0

    for entry in entries:
        by_status[entry.download_status] += 1
        by_level[entry.processing_level] += 1
        by_both[(entry.download_status, entry.processing_level)] += 1
        try:
            size = entry.size_bytes
        except ManifestValueError:
            size = None
        if size is None:
            unknown_size += 1
        else:
            known_bytes += size
        for column in MANIFEST_COLUMNS:
            if is_unknown(entry.cell(column)):
                unknown_cells[column] += 1

    return ManifestSummary(
        n_entries=len(entries),
        by_status=dict(by_status),
        by_processing_level=dict(by_level),
        by_status_and_level=dict(by_both),
        n_accepted=sum(v for k, v in by_status.items() if k in ACCEPTED_STATUSES),
        n_rejected=sum(v for k, v in by_status.items() if k in REJECTED_STATUSES),
        known_bytes=known_bytes,
        rows_with_unknown_size=unknown_size,
        unknown_cells_by_column=dict(unknown_cells),
    )


__all__ = [
    "ACCEPTED_STATUSES", "DOWNLOAD_STATUS_VOCABULARY", "INTEGER_COLUMNS",
    "MANIFEST_COLUMNS", "MANIFEST_FILENAME", "ManifestEntry", "ManifestError",
    "ManifestIssue", "ManifestSchemaError", "ManifestSummary", "ManifestValidation",
    "ManifestValueError", "NUMERIC_COLUMNS", "REASON_REQUIRED_STATUSES",
    "REJECTED_STATUSES", "SEVERITY_ERROR", "SEVERITY_WARNING", "UNKNOWN",
    "URL_COLUMNS", "as_float", "as_int", "entry_from_download", "format_cell",
    "is_unknown", "read_manifest", "read_manifest_frame", "summarise_manifest",
    "validate_manifest", "write_manifest",
]
