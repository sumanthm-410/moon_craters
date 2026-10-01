"""Resumable archive downloads with content validation and credential redaction.

This module is the only place in the project that touches the network, and it is
written so that every decision it makes can be tested without a network.  The
HTTP layer is injected (``session``), as are the clock and the sleep function,
so the retry/backoff/resume logic is exercised against in-memory fakes in
``tests/test_download.py``.

Design rules (non-negotiable, see INTERFACES.md rule 6)
------------------------------------------------------
* **Never fabricate a result.**  A download that cannot be completed raises;
  a file that cannot be validated is rejected and *removed*.  No function here
  invents a byte count, a checksum or a URL.
* **The final path is only ever written by an atomic rename of a validated
  file.**  Bytes land in a sibling ``.part`` staging file, are validated there,
  and only then are ``os.replace``-d into place.  A failed or interrupted
  download therefore cannot leave a corrupt file at the final path, and a
  consumer that sees the final path can trust it.
* **Source bytes are preserved byte-for-byte.**  Staging files are opened in
  binary append/read-write mode; nothing is decoded, recompressed or
  line-ending translated on the way through.
* **Size is not validation.**  A 403 HTML error page or an S3 ``<Error>``
  document saved under an ``.IMG`` name has a perfectly plausible size.  The
  magic-byte / leading-content sniff is a *required* check, not an optional one.
* **Credentials are never logged.**  Archive URLs are frequently presigned
  (``?X-Amz-Signature=...``, ``?token=...``).  Every URL that reaches a log
  line, an exception message or a manifest goes through :func:`redact_url`.

Permanent vs transient
----------------------
A 403 or 404 is a *permanent* answer: retrying it wastes time and, for a
policy-denied host, produces identical denials forever.  It raises
:class:`PermanentDownloadError` on the first attempt, with no retry.  A 503, a
429, a read timeout or a truncated body is *transient* and is retried with
bounded exponential backoff, resuming from the partial file where the server
supports it.
"""
from __future__ import annotations

import hashlib
import os
import re
import urllib.parse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

# --------------------------------------------------------------------------- #
# Status classification
# --------------------------------------------------------------------------- #
#: Outcomes that will not change if we ask again: authorisation, policy,
#: addressing and method errors.  Never retried.
PERMANENT_STATUS: frozenset[int] = frozenset(
    {400, 401, 402, 403, 404, 405, 406, 410, 414, 451, 501, 505}
)

#: Outcomes that plausibly change on a later attempt.
TRANSIENT_STATUS: frozenset[int] = frozenset({408, 425, 429, 500, 502, 503, 504, 507, 509})

#: Successful responses this module knows how to consume.
_OK_STATUS: frozenset[int] = frozenset({200, 206})

CLASSIFICATION_OK = "ok"
CLASSIFICATION_PERMANENT = "permanent"
CLASSIFICATION_TRANSIENT = "transient"


class DownloadError(Exception):
    """Base class for download failures.  Messages are always redacted."""


class PermanentDownloadError(DownloadError):
    """The request will not succeed on a retry (403, 404, policy denial)."""


class TransientDownloadError(DownloadError):
    """The request may succeed later (timeout, 503, truncated body)."""


class ValidationFailed(DownloadError):
    """A downloaded file failed content validation and was discarded."""

    def __init__(self, message: str, result: "ValidationResult") -> None:
        super().__init__(message)
        self.result = result


def classify_http_status(status: int) -> str:
    """Classify an HTTP status as ``ok``, ``permanent`` or ``transient``.

    Unknown statuses are classified conservatively: anything else in 4xx is
    permanent (the client asked for something wrong), anything else in 5xx is
    transient (the server is unwell).
    """
    status = int(status)
    if status in _OK_STATUS:
        return CLASSIFICATION_OK
    if status in PERMANENT_STATUS:
        return CLASSIFICATION_PERMANENT
    if status in TRANSIENT_STATUS:
        return CLASSIFICATION_TRANSIENT
    if 200 <= status < 300:
        # A 2xx we do not model (204, 304-adjacent proxies).  Treat as permanent
        # rather than silently accepting a body we did not expect.
        return CLASSIFICATION_PERMANENT
    if 400 <= status < 500:
        return CLASSIFICATION_PERMANENT
    if 500 <= status < 600:
        return CLASSIFICATION_TRANSIENT
    return CLASSIFICATION_PERMANENT


#: Exceptions from an injected HTTP layer that are treated as transient.
#: ``requests`` exceptions all derive from ``OSError`` via ``IOError``, so this
#: covers connection resets, DNS hiccups and read timeouts without importing
#: ``requests`` here.
TRANSIENT_EXCEPTIONS: tuple[type[BaseException], ...] = (
    TransientDownloadError,
    TimeoutError,
    ConnectionError,
    OSError,
)


# --------------------------------------------------------------------------- #
# Credential redaction
# --------------------------------------------------------------------------- #
#: Query-parameter names whose *values* must never be shown.  Matching is
#: case-insensitive and on substrings, so ``X-Amz-Security-Token`` is caught by
#: ``token`` and ``X-Goog-Signature`` by ``signature``.
_SECRET_QUERY_HINTS: tuple[str, ...] = (
    "signature", "token", "credential", "secret", "password", "passwd",
    "apikey", "api_key", "access_key", "accesskey", "awsaccesskeyid",
    "auth", "sig", "key_id", "keyid", "sas", "session",
)

#: Replacement text.  Deliberately not the same length as any real secret.
REDACTED = "REDACTED"

# Free-text form, for secrets pasted into a message without a full URL, e.g.
# a server error that echoes "Signature=abcd1234".
_SECRET_TEXT_RE = re.compile(
    r"(?i)\b("
    + "|".join(re.escape(h) for h in _SECRET_QUERY_HINTS)
    + r")([A-Za-z0-9_\-]*)\s*([=:])\s*([^\s&\"'<>,;)]+)"
)

_URL_RE = re.compile(r"(?i)\b[a-z][a-z0-9+.\-]*://[^\s\"'<>]+")


def _is_secret_key(key: str) -> bool:
    low = key.lower()
    return any(hint in low for hint in _SECRET_QUERY_HINTS)


def redact_url(url: str) -> str:
    """Return ``url`` with credential-bearing parts replaced by ``REDACTED``.

    Host and path are preserved, because they are what makes an error message
    diagnosable ("which product, which archive").  Removed are: userinfo in the
    netloc, and the values of query parameters whose names look like a token,
    signature or key.  Non-secret parameters (``version``, ``format``) are kept
    so a presigned URL stays recognisable.
    """
    if not url:
        return url
    try:
        parts = urllib.parse.urlsplit(url)
    except ValueError:  # pragma: no cover - urlsplit is extremely permissive
        return REDACTED
    if not parts.scheme and not parts.netloc:
        # Not a URL at all; fall back to the free-text rule so a bare
        # "token=..." fragment is still redacted.
        return _SECRET_TEXT_RE.sub(lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}", url)

    netloc = parts.netloc
    if "@" in netloc:
        netloc = f"{REDACTED}@{netloc.rsplit('@', 1)[1]}"

    query = parts.query
    if query:
        pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
        if pairs:
            query = urllib.parse.urlencode(
                [(k, REDACTED if _is_secret_key(k) else v) for k, v in pairs]
            )
        else:
            # Opaque query with no key=value structure; it may itself be a
            # token, so drop it entirely rather than guess.
            query = REDACTED
    fragment = REDACTED if parts.fragment and _is_secret_key(parts.fragment) else parts.fragment
    return urllib.parse.urlunsplit((parts.scheme, netloc, parts.path, query, fragment))


def redact_text(text: Any) -> str:
    """Redact every URL and every ``secret=value`` fragment inside ``text``."""
    out = str(text)
    out = _URL_RE.sub(lambda m: redact_url(m.group(0)), out)
    out = _SECRET_TEXT_RE.sub(
        lambda m: f"{m.group(1)}{m.group(2)}{m.group(3)}{REDACTED}", out
    )
    return out


def redact_headers(headers: Mapping[str, str]) -> dict[str, str]:
    """Redact ``Authorization``-style request headers for logging."""
    out: dict[str, str] = {}
    for key, value in headers.items():
        if _is_secret_key(key) or key.lower() in {"authorization", "cookie", "proxy-authorization"}:
            out[key] = REDACTED
        else:
            out[key] = redact_text(value)
    return out


# --------------------------------------------------------------------------- #
# Content sniffing
# --------------------------------------------------------------------------- #
#: Number of leading bytes used for the magic-byte / error-document sniff.
SNIFF_BYTES = 4096

KIND_HTML = "html"
KIND_XML = "xml"
KIND_TIFF = "tiff"
KIND_BIGTIFF = "bigtiff"
KIND_JPEG2000 = "jpeg2000"
KIND_PNG = "png"
KIND_GZIP = "gzip"
KIND_ZIP = "zip"
KIND_PDS_LABEL = "pds_label"
KIND_VICAR = "vicar"
KIND_ISIS_CUBE = "isis_cube"
KIND_JSON = "json"
KIND_EMPTY = "empty"
KIND_UNKNOWN_BINARY = "unknown_binary"
KIND_UNKNOWN_TEXT = "unknown_text"

#: Kinds that are never a planetary data product -- they are what a proxy,
#: an authentication wall or an object store returns *instead* of the product.
ERROR_DOCUMENT_KINDS: frozenset[str] = frozenset({KIND_HTML, KIND_XML, KIND_JSON})

_HTML_MARKERS = (b"<!doctype html", b"<html", b"<head", b"<body", b"<!-- ")
_XML_MARKERS = (b"<?xml", b"<error>", b"<error ")

#: Phrases that identify an error document even when wrapped in markup we did
#: not anticipate.  Lower-cased comparison.
_ERROR_PHRASES = (
    b"access denied", b"accessdenied", b"forbidden", b"not found",
    b"nosuchkey", b"signaturedoesnotmatch", b"invalidaccesskeyid",
    b"expiredtoken", b"authentication required", b"403", b"404",
    b"proxy error", b"request blocked",
)


def sniff_payload_kind(head: bytes) -> str:
    """Classify a payload from its leading bytes.

    This is the core of the "HTML page saved as .IMG" defence.  It looks at
    magic bytes for the formats this project actually consumes, and at leading
    textual content for markup, rather than trusting the file name or the
    ``Content-Type`` header (a misconfigured archive mirror sets both wrongly).
    """
    if not head:
        return KIND_EMPTY
    if head[:4] in (b"II*\x00", b"MM\x00*"):
        return KIND_TIFF
    if head[:4] in (b"II+\x00", b"MM\x00+"):
        return KIND_BIGTIFF
    if head[:4] == b"\x89PNG":
        return KIND_PNG
    if head[:2] == b"\x1f\x8b":
        return KIND_GZIP
    if head[:4] in (b"PK\x03\x04", b"PK\x05\x06"):
        return KIND_ZIP
    if head[:4] == b"\xff\x4f\xff\x51" or head[4:8] == b"jP  " or head[4:8] == b"ftyp":
        return KIND_JPEG2000

    stripped = head.lstrip(b" \t\r\n\x00")
    low = stripped[:512].lower()
    if any(low.startswith(marker) for marker in _HTML_MARKERS):
        return KIND_HTML
    if any(low.startswith(marker) for marker in _XML_MARKERS):
        # An S3/GCS error body is XML; so is a PDS4 label.  Distinguish by
        # looking for a PDS4 root element before calling it an error document.
        if b"product_observational" in low or b"pds4" in low or b"pds/v1" in low:
            return KIND_XML if b"<error" in low else KIND_PDS_LABEL
        return KIND_XML
    if stripped[:1] in (b"{", b"["):
        return KIND_JSON
    if stripped.startswith(b"PDS_VERSION_ID") or stripped.startswith(b"CCSD"):
        return KIND_PDS_LABEL
    if stripped.startswith(b"NJPL1I00"):
        return KIND_PDS_LABEL
    if stripped.startswith(b"LBLSIZE"):
        return KIND_VICAR
    if re.match(rb"\s*Object\s*=\s*IsisCube", stripped[:64], re.IGNORECASE):
        return KIND_ISIS_CUBE

    # Heuristic text/binary split: a planetary raster's first block is almost
    # never pure printable ASCII.
    printable = sum(1 for b in head[:512] if 9 <= b <= 13 or 32 <= b <= 126)
    if printable == len(head[:512]):
        return KIND_UNKNOWN_TEXT
    return KIND_UNKNOWN_BINARY


def looks_like_error_document(head: bytes) -> tuple[bool, str]:
    """Return ``(is_error_document, reason)`` for a payload's leading bytes.

    ``reason`` is always a human-readable sentence, suitable for a manifest
    ``quality_notes`` field.
    """
    kind = sniff_payload_kind(head)
    low = head[:SNIFF_BYTES].lower()
    phrases = [p.decode() for p in _ERROR_PHRASES if p in low]
    if kind in ERROR_DOCUMENT_KINDS:
        detail = f"; contains {phrases[:3]}" if phrases else ""
        return True, f"payload sniffs as {kind}, not a data product{detail}"
    if kind == KIND_EMPTY:
        return True, "payload is empty"
    if kind == KIND_UNKNOWN_TEXT and phrases:
        return True, f"payload is plain text containing error phrases {phrases[:3]}"
    return False, f"payload sniffs as {kind}"


def read_head(path: str | os.PathLike[str], n: int = SNIFF_BYTES) -> bytes:
    """Read up to ``n`` leading bytes of ``path`` in binary mode."""
    with open(path, "rb") as handle:
        return handle.read(n)


# --------------------------------------------------------------------------- #
# Checksums
# --------------------------------------------------------------------------- #
#: Hash algorithms an archive may advertise.  ``md5`` is included because PDS
#: and S3 commonly publish only an MD5; it is used for integrity, never for
#: security.
SUPPORTED_CHECKSUM_ALGORITHMS: tuple[str, ...] = ("md5", "sha1", "sha256", "sha512")

_CHECKSUM_CHUNK = 1 << 20


def file_checksum(path: str | os.PathLike[str], algorithm: str = "sha256") -> str:
    """Streamed hex digest of a file.  Raises ``ValueError`` on unknown algorithm."""
    algo = algorithm.lower().replace("-", "")
    if algo not in SUPPORTED_CHECKSUM_ALGORITHMS:
        raise ValueError(
            f"unsupported checksum algorithm {algorithm!r}; "
            f"supported: {SUPPORTED_CHECKSUM_ALGORITHMS}"
        )
    digest = hashlib.new(algo)
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHECKSUM_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
CHECK_EXISTS = "exists"
CHECK_NOT_ERROR_DOCUMENT = "not_error_document"
CHECK_SIZE = "size"
CHECK_CHECKSUM = "checksum"
CHECK_RASTER = "raster_opens"

STATUS_PASS = "pass"
STATUS_FAIL = "fail"
STATUS_SKIP = "skip"

#: File suffixes for which the rasterio open check is run by default.
RASTER_SUFFIXES: frozenset[str] = frozenset(
    {".tif", ".tiff", ".img", ".jp2", ".cub", ".vrt", ".lbl"}
)


@dataclass(frozen=True)
class CheckResult:
    """One validation check: its name, outcome and the reason for it."""

    name: str
    status: str
    reason: str

    @property
    def passed(self) -> bool:
        return self.status == STATUS_PASS

    @property
    def failed(self) -> bool:
        return self.status == STATUS_FAIL

    def __str__(self) -> str:  # pragma: no cover - convenience only
        return f"{self.name}: {self.status} ({self.reason})"


@dataclass(frozen=True)
class ValidationResult:
    """Structured per-check outcome of validating a downloaded file.

    ``ok`` is true only when no check failed.  A *skipped* check never makes a
    file valid by itself: :attr:`checks_performed` records what was actually
    verified, so a caller (or the manifest) can state the strength of the
    evidence instead of implying a full validation happened.
    """

    path: Path
    size_bytes: int
    checks: tuple[CheckResult, ...]

    @property
    def ok(self) -> bool:
        return not any(check.failed for check in self.checks)

    @property
    def failures(self) -> tuple[CheckResult, ...]:
        return tuple(check for check in self.checks if check.failed)

    @property
    def checks_performed(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if check.status != STATUS_SKIP)

    @property
    def skipped(self) -> tuple[str, ...]:
        return tuple(check.name for check in self.checks if check.status == STATUS_SKIP)

    def get(self, name: str) -> CheckResult:
        for check in self.checks:
            if check.name == name:
                return check
        raise KeyError(name)

    def reason(self) -> str:
        """One-line summary suitable for a manifest ``quality_notes`` cell."""
        if self.ok:
            return "ok: " + ", ".join(self.checks_performed)
        return "; ".join(f"{c.name} failed: {c.reason}" for c in self.failures)


def validate_raster_file(
    path: str | os.PathLike[str],
    *,
    expected_size_bytes: int | None = None,
    checksum: str | None = None,
    checksum_algorithm: str = "sha256",
    expect_raster: bool | None = None,
    min_width: int = 1,
    min_height: int = 1,
    min_count: int = 1,
) -> ValidationResult:
    """Validate a downloaded product and report every check individually.

    Checks, in order:

    ``exists``
        The path exists and is a regular file.  Recorded separately so a
        missing file does not masquerade as a content failure.
    ``not_error_document``
        **Required.**  Magic-byte and leading-content sniff.  An HTML page or
        an S3 ``<Error>`` XML document saved under an ``.IMG``/``.tif`` name
        fails here, which size- and even checksum-free pipelines miss entirely.
    ``size``
        Byte size equals ``expected_size_bytes`` when the archive supplied one.
        Skipped (never guessed) when it did not.
    ``checksum``
        Hex digest equals ``checksum`` when the archive supplied one.
    ``raster_opens``
        ``rasterio.open`` succeeds and the dataset reports sane
        ``width``/``height``/``count``.  This is the check that distinguishes
        "the bytes arrived" from "the product is readable".  ``expect_raster``
        defaults to a decision based on the file suffix
        (:data:`RASTER_SUFFIXES`); pass ``False`` for a tarball, and also for a
        raw PDS EDR whose label is a separate file -- GDAL cannot open the
        image without it, so requiring it here would reject a perfectly good
        download.  Pass ``True`` to require readability regardless of suffix.
    """
    file_path = Path(path)
    checks: list[CheckResult] = []

    if not file_path.is_file():
        checks.append(CheckResult(CHECK_EXISTS, STATUS_FAIL, f"not a regular file: {file_path}"))
        for name in (CHECK_NOT_ERROR_DOCUMENT, CHECK_SIZE, CHECK_CHECKSUM, CHECK_RASTER):
            checks.append(CheckResult(name, STATUS_SKIP, "file missing"))
        return ValidationResult(path=file_path, size_bytes=0, checks=tuple(checks))

    size = file_path.stat().st_size
    checks.append(CheckResult(CHECK_EXISTS, STATUS_PASS, f"regular file, {size} bytes"))

    head = read_head(file_path)
    is_error, sniff_reason = looks_like_error_document(head)
    checks.append(
        CheckResult(
            CHECK_NOT_ERROR_DOCUMENT,
            STATUS_FAIL if is_error else STATUS_PASS,
            redact_text(sniff_reason),
        )
    )

    if expected_size_bytes is None:
        checks.append(CheckResult(CHECK_SIZE, STATUS_SKIP, "archive supplied no expected size"))
    elif size == int(expected_size_bytes):
        checks.append(CheckResult(CHECK_SIZE, STATUS_PASS, f"{size} bytes as expected"))
    else:
        checks.append(
            CheckResult(
                CHECK_SIZE,
                STATUS_FAIL,
                f"expected {int(expected_size_bytes)} bytes, got {size}",
            )
        )

    if checksum is None:
        checks.append(
            CheckResult(CHECK_CHECKSUM, STATUS_SKIP, "archive supplied no checksum")
        )
    else:
        try:
            actual = file_checksum(file_path, checksum_algorithm)
        except ValueError as exc:
            checks.append(CheckResult(CHECK_CHECKSUM, STATUS_FAIL, str(exc)))
        else:
            expected = str(checksum).strip().lower()
            if actual == expected:
                checks.append(
                    CheckResult(
                        CHECK_CHECKSUM, STATUS_PASS, f"{checksum_algorithm}={actual}"
                    )
                )
            else:
                checks.append(
                    CheckResult(
                        CHECK_CHECKSUM,
                        STATUS_FAIL,
                        f"{checksum_algorithm} mismatch: expected {expected}, got {actual}",
                    )
                )

    want_raster = (
        file_path.suffix.lower() in RASTER_SUFFIXES if expect_raster is None else bool(expect_raster)
    )
    if not want_raster:
        checks.append(
            CheckResult(CHECK_RASTER, STATUS_SKIP, "not expected to be a raster product")
        )
    elif is_error:
        # Opening an HTML page with GDAL only produces a second, less specific
        # error; the sniff already said what is wrong.
        checks.append(
            CheckResult(CHECK_RASTER, STATUS_SKIP, "skipped: payload is an error document")
        )
    else:
        checks.append(_raster_check(file_path, min_width, min_height, min_count))

    return ValidationResult(path=file_path, size_bytes=size, checks=tuple(checks))


def _raster_check(path: Path, min_width: int, min_height: int, min_count: int) -> CheckResult:
    try:
        import rasterio
    except Exception as exc:  # pragma: no cover - rasterio is a hard dependency
        return CheckResult(CHECK_RASTER, STATUS_FAIL, f"rasterio unavailable: {exc}")
    try:
        with rasterio.open(path) as dataset:
            width, height, count = dataset.width, dataset.height, dataset.count
            dtypes = tuple(dataset.dtypes)
    except Exception as exc:
        return CheckResult(
            CHECK_RASTER, STATUS_FAIL, f"rasterio could not open the file: {type(exc).__name__}"
        )
    if width < min_width or height < min_height or count < min_count:
        return CheckResult(
            CHECK_RASTER,
            STATUS_FAIL,
            f"implausible raster shape width={width} height={height} count={count}",
        )
    return CheckResult(
        CHECK_RASTER,
        STATUS_PASS,
        f"opens as raster: {width}x{height}x{count}, dtypes={dtypes}",
    )


# --------------------------------------------------------------------------- #
# Retry policy and transfer bookkeeping
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class RetryPolicy:
    """Bounded retry with exponential backoff and explicit timeouts.

    ``max_attempts`` is a hard ceiling on *requests made*, not on retries after
    the first, so ``max_attempts=1`` means "try once, never retry".  Backoff is
    deterministic by default (``jitter=0``) so tests can assert the exact sleep
    sequence; production callers should set a small jitter to avoid
    synchronising against other clients.
    """

    max_attempts: int = 4
    backoff_initial_s: float = 1.0
    backoff_factor: float = 2.0
    backoff_max_s: float = 60.0
    jitter: float = 0.0
    connect_timeout_s: float = 10.0
    read_timeout_s: float = 60.0

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        if self.backoff_initial_s < 0 or self.backoff_max_s < 0:
            raise ValueError("backoff times must be non-negative")
        if self.backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0 for exponential backoff")
        if not 0.0 <= self.jitter <= 1.0:
            raise ValueError("jitter must be a fraction in [0, 1]")

    @property
    def timeout(self) -> tuple[float, float]:
        """``(connect, read)`` timeout pair, the shape ``requests`` expects."""
        return (self.connect_timeout_s, self.read_timeout_s)

    def backoff_for(self, attempt: int, rand: Callable[[], float] | None = None) -> float:
        """Delay in seconds before the attempt *after* ``attempt`` (1-based)."""
        delay = self.backoff_initial_s * (self.backoff_factor ** max(0, attempt - 1))
        delay = min(delay, self.backoff_max_s)
        if self.jitter and rand is not None:
            delay *= 1.0 + self.jitter * (2.0 * rand() - 1.0)
        return max(0.0, delay)


@dataclass(frozen=True)
class AttemptRecord:
    """What one HTTP attempt did.  Kept for the manifest and for tests."""

    attempt: int
    status_code: int | None
    classification: str
    range_requested: str | None
    range_honoured: bool
    bytes_received: int
    note: str


@dataclass(frozen=True)
class DownloadOutcome:
    """Result of :func:`fetch_to_path`.

    ``url`` is stored already redacted: this object may legitimately be written
    into a manifest or a log, so it must never carry a signature.
    """

    path: Path
    url: str
    status: str
    bytes_on_disk: int
    bytes_transferred: int
    resumed: bool
    attempts: tuple[AttemptRecord, ...]
    validation: ValidationResult | None
    server_supports_range: bool | None
    note: str = ""

    @property
    def ok(self) -> bool:
        return self.status in {"downloaded", "already_present"}


STATUS_DOWNLOADED = "downloaded"
STATUS_ALREADY_PRESENT = "already_present"
STATUS_WOULD_DOWNLOAD = "would_download"


# --------------------------------------------------------------------------- #
# The fetcher
# --------------------------------------------------------------------------- #
def staging_path_for(dest: str | os.PathLike[str]) -> Path:
    """Deterministic staging path for ``dest``.

    Deterministic (not a random temp name) so a resumed run in a *new process*
    finds the partial file from the previous one.  It is a sibling of ``dest``
    so the final ``os.replace`` is same-filesystem and therefore atomic.
    """
    dest = Path(dest)
    return dest.with_name(dest.name + ".part")


def _default_session() -> Any:
    """A ``requests.Session``, imported lazily so tests never need it."""
    import requests  # local import: offline tests must not pay for this

    return requests.Session()


def _log(logger: Callable[[str], None] | None, message: str) -> None:
    if logger is not None:
        logger(redact_text(message))


def _parse_content_range_total(value: str | None) -> int | None:
    """Total size from a ``Content-Range: bytes 100-199/1234`` header."""
    if not value:
        return None
    match = re.search(r"/\s*(\d+)\s*$", value)
    return int(match.group(1)) if match else None


def _header(headers: Mapping[str, Any] | None, name: str) -> str | None:
    """Case-insensitive header lookup that tolerates a plain ``dict`` fake."""
    if not headers:
        return None
    try:
        value = headers.get(name)  # requests' CaseInsensitiveDict handles this
    except AttributeError:  # pragma: no cover
        value = None
    if value is not None:
        return str(value)
    low = name.lower()
    for key, val in headers.items():
        if str(key).lower() == low:
            return None if val is None else str(val)
    return None


def fetch_to_path(
    url: str,
    dest: str | os.PathLike[str],
    *,
    expected_size_bytes: int | None = None,
    checksum: str | None = None,
    checksum_algorithm: str = "sha256",
    expect_raster: bool | None = None,
    policy: RetryPolicy | None = None,
    session: Any | None = None,
    headers: Mapping[str, str] | None = None,
    allow_resume: bool = True,
    validate: bool = True,
    chunk_size: int = 1 << 20,
    dry_run: bool = False,
    sleep: Callable[[float], None] | None = None,
    rand: Callable[[], float] | None = None,
    logger: Callable[[str], None] | None = None,
    progress: Callable[[int, int | None], None] | None = None,
    overwrite: bool = False,
) -> DownloadOutcome:
    """Download ``url`` to ``dest``, resumably, and validate before publishing.

    The returned :class:`DownloadOutcome` carries a redacted URL, the per-attempt
    history and the :class:`ValidationResult`.

    Raises
    ------
    PermanentDownloadError
        On a 403/404-class response, on the first attempt, with no retry.
    TransientDownloadError
        When every one of ``policy.max_attempts`` attempts failed transiently.
        The staging file is **kept** so a later call can resume it.
    ValidationFailed
        When the bytes arrived but failed validation.  The staging file is
        *removed*, because corrupt content must not seed a later resume, and
        nothing is written to ``dest``.
    """
    policy = policy or RetryPolicy()
    sleep = sleep or _default_sleep
    dest_path = Path(dest)
    safe_url = redact_url(url)
    staging = staging_path_for(dest_path)
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    # A product already in place and still valid is the common resume case and
    # must not cost a request.
    if dest_path.exists() and not overwrite:
        existing = (
            validate_raster_file(
                dest_path,
                expected_size_bytes=expected_size_bytes,
                checksum=checksum,
                checksum_algorithm=checksum_algorithm,
                expect_raster=expect_raster,
            )
            if validate
            else None
        )
        if existing is None or existing.ok:
            _log(logger, f"already present and valid, skipping: {dest_path}")
            return DownloadOutcome(
                path=dest_path,
                url=safe_url,
                status=STATUS_ALREADY_PRESENT,
                bytes_on_disk=dest_path.stat().st_size,
                bytes_transferred=0,
                resumed=False,
                attempts=(),
                validation=existing,
                server_supports_range=None,
                note="destination already present and validated",
            )
        _log(
            logger,
            f"destination exists but is invalid ({existing.reason()}); re-downloading",
        )
        dest_path.unlink()

    if dry_run:
        partial = staging.stat().st_size if staging.exists() else 0
        return DownloadOutcome(
            path=dest_path,
            url=safe_url,
            status=STATUS_WOULD_DOWNLOAD,
            bytes_on_disk=0,
            bytes_transferred=0,
            resumed=partial > 0,
            attempts=(),
            validation=None,
            server_supports_range=None,
            note=(
                f"dry run: would fetch {safe_url} to {dest_path}"
                + (f", resuming from {partial} bytes" if partial else "")
            ),
        )

    session = session if session is not None else _default_session()
    base_headers = dict(headers or {})
    records: list[AttemptRecord] = []
    transferred = 0
    resumed_any = False
    supports_range: bool | None = None
    last_transient: str = ""

    for attempt in range(1, policy.max_attempts + 1):
        offset = staging.stat().st_size if (allow_resume and staging.exists()) else 0
        if offset and supports_range is False:
            # The server already told us it ignores Range; do not pretend.
            _log(logger, "server does not support Range; restarting from zero")
            staging.unlink()
            offset = 0

        request_headers = dict(base_headers)
        range_header: str | None = None
        if offset > 0:
            range_header = f"bytes={offset}-"
            request_headers["Range"] = range_header

        _log(
            logger,
            f"attempt {attempt}/{policy.max_attempts} GET {safe_url} "
            f"headers={redact_headers(request_headers)}",
        )
        try:
            response = session.get(
                url,
                headers=request_headers,
                stream=True,
                timeout=policy.timeout,
            )
        except TRANSIENT_EXCEPTIONS as exc:
            last_transient = f"{type(exc).__name__}: {redact_text(exc)}"
            records.append(
                AttemptRecord(
                    attempt, None, CLASSIFICATION_TRANSIENT, range_header, False, 0,
                    last_transient,
                )
            )
            _log(logger, f"attempt {attempt} failed transiently: {last_transient}")
            if attempt < policy.max_attempts:
                sleep(policy.backoff_for(attempt, rand))
            continue

        try:
            status_code = int(response.status_code)
            classification = classify_http_status(status_code)
            resp_headers = getattr(response, "headers", {}) or {}
            accept_ranges = (_header(resp_headers, "Accept-Ranges") or "").strip().lower()

            if status_code == 416 and offset > 0:
                # The partial is at least as long as the resource: it is stale
                # or the resource changed.  Restart cleanly rather than guess.
                records.append(
                    AttemptRecord(
                        attempt, status_code, CLASSIFICATION_TRANSIENT, range_header,
                        False, 0, "416: partial file is stale, restarting from zero",
                    )
                )
                _log(logger, "416 Range Not Satisfiable; discarding partial file")
                staging.unlink(missing_ok=True)
                continue

            if classification == CLASSIFICATION_PERMANENT:
                note = f"HTTP {status_code} (permanent)"
                records.append(
                    AttemptRecord(
                        attempt, status_code, classification, range_header, False, 0, note
                    )
                )
                _log(logger, f"permanent failure, not retrying: {note} for {safe_url}")
                raise PermanentDownloadError(
                    f"{note} for {safe_url}: this will not succeed on a retry "
                    f"(host policy, authorisation or a wrong product id)"
                )
            if classification == CLASSIFICATION_TRANSIENT:
                last_transient = f"HTTP {status_code} (transient)"
                records.append(
                    AttemptRecord(
                        attempt, status_code, classification, range_header, False, 0,
                        last_transient,
                    )
                )
                _log(logger, f"attempt {attempt}: {last_transient} for {safe_url}")
                if attempt < policy.max_attempts:
                    sleep(policy.backoff_for(attempt, rand))
                continue

            # --- 200 or 206 ------------------------------------------------- #
            honoured = status_code == 206
            if offset > 0:
                if honoured:
                    supports_range = True
                    resumed_any = True
                else:
                    # Asked for a range, got the whole entity: the server does
                    # not support Range.  Appending here would corrupt the file
                    # by duplicating the first ``offset`` bytes -- restart.
                    supports_range = False
                    _log(
                        logger,
                        f"server ignored Range (HTTP {status_code}, "
                        f"Accept-Ranges={accept_ranges or 'absent'}); "
                        f"discarding {offset} partial bytes and restarting",
                    )
                    staging.unlink(missing_ok=True)
                    offset = 0
            elif accept_ranges:
                supports_range = accept_ranges != "none"

            total = _parse_content_range_total(_header(resp_headers, "Content-Range"))
            if total is None:
                length = _header(resp_headers, "Content-Length")
                if length is not None and str(length).strip().isdigit():
                    total = offset + int(str(length).strip())
            if total is None and expected_size_bytes is not None:
                total = int(expected_size_bytes)

            mode = "ab" if offset > 0 else "wb"
            received = 0
            staging.parent.mkdir(parents=True, exist_ok=True)
            try:
                with open(staging, mode) as handle:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if not chunk:
                            continue
                        handle.write(chunk)
                        received += len(chunk)
                        if progress is not None:
                            progress(offset + received, total)
                    handle.flush()
                    os.fsync(handle.fileno())
            except TRANSIENT_EXCEPTIONS as exc:
                transferred += received
                last_transient = (
                    f"stream interrupted after {received} bytes: "
                    f"{type(exc).__name__}: {redact_text(exc)}"
                )
                records.append(
                    AttemptRecord(
                        attempt, status_code, CLASSIFICATION_TRANSIENT, range_header,
                        honoured, received, last_transient,
                    )
                )
                _log(logger, f"attempt {attempt}: {last_transient}")
                if attempt < policy.max_attempts:
                    sleep(policy.backoff_for(attempt, rand))
                continue

            transferred += received
            on_disk = staging.stat().st_size
            if total is not None and on_disk < total:
                last_transient = (
                    f"truncated body: {on_disk} of {total} bytes "
                    f"(server closed the connection early)"
                )
                records.append(
                    AttemptRecord(
                        attempt, status_code, CLASSIFICATION_TRANSIENT, range_header,
                        honoured, received, last_transient,
                    )
                )
                _log(logger, f"attempt {attempt}: {last_transient}")
                if attempt < policy.max_attempts:
                    sleep(policy.backoff_for(attempt, rand))
                continue
            if total is not None and on_disk > total:
                # More bytes than the resource has: a resumed append against a
                # changed resource.  Never publish this; restart clean.
                last_transient = f"overlong body: {on_disk} bytes for a {total}-byte resource"
                records.append(
                    AttemptRecord(
                        attempt, status_code, CLASSIFICATION_TRANSIENT, range_header,
                        honoured, received, last_transient,
                    )
                )
                _log(logger, f"attempt {attempt}: {last_transient}; discarding partial")
                staging.unlink(missing_ok=True)
                if attempt < policy.max_attempts:
                    sleep(policy.backoff_for(attempt, rand))
                continue

            records.append(
                AttemptRecord(
                    attempt, status_code, CLASSIFICATION_OK, range_header, honoured,
                    received, f"received {received} bytes, {on_disk} on disk",
                )
            )
            break
        finally:
            close = getattr(response, "close", None)
            if callable(close):
                close()
    else:
        raise TransientDownloadError(
            f"giving up on {safe_url} after {policy.max_attempts} attempt(s); "
            f"last failure: {last_transient or 'unknown'}. "
            f"Partial file kept at {staging} for a later resume."
        )

    # --- validate in staging, then publish atomically ---------------------- #
    result = (
        validate_raster_file(
            staging,
            expected_size_bytes=expected_size_bytes,
            checksum=checksum,
            checksum_algorithm=checksum_algorithm,
            expect_raster=expect_raster if expect_raster is not None
            else (dest_path.suffix.lower() in RASTER_SUFFIXES),
        )
        if validate
        else None
    )
    if result is not None and not result.ok:
        # Discard the bytes: they are not a partial transfer, they are wrong
        # content, and keeping them would make the next resume append to junk.
        staging.unlink(missing_ok=True)
        _log(logger, f"rejected download of {safe_url}: {result.reason()}")
        raise ValidationFailed(
            f"validation failed for {safe_url} -> {dest_path}: {result.reason()}; "
            f"nothing was written to the final path",
            result,
        )

    os.replace(staging, dest_path)
    _fsync_dir(dest_path.parent)
    size = dest_path.stat().st_size
    _log(logger, f"published {dest_path} ({size} bytes) from {safe_url}")
    return DownloadOutcome(
        path=dest_path,
        url=safe_url,
        status=STATUS_DOWNLOADED,
        bytes_on_disk=size,
        bytes_transferred=transferred,
        resumed=resumed_any,
        attempts=tuple(records),
        validation=result,
        server_supports_range=supports_range,
        note="validated in staging and renamed into place",
    )


def _default_sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


def _fsync_dir(directory: Path) -> None:
    """fsync a directory so a rename survives a crash.  Best effort."""
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:  # pragma: no cover - platform dependent
        return
    try:
        os.fsync(fd)
    except OSError:  # pragma: no cover - some filesystems refuse
        pass
    finally:
        os.close(fd)


__all__ = [
    "AttemptRecord", "CHECK_CHECKSUM", "CHECK_EXISTS", "CHECK_NOT_ERROR_DOCUMENT",
    "CHECK_RASTER", "CHECK_SIZE", "CLASSIFICATION_OK", "CLASSIFICATION_PERMANENT",
    "CLASSIFICATION_TRANSIENT", "CheckResult", "DownloadError", "DownloadOutcome",
    "ERROR_DOCUMENT_KINDS", "PERMANENT_STATUS", "PermanentDownloadError",
    "REDACTED", "RASTER_SUFFIXES", "RetryPolicy", "STATUS_ALREADY_PRESENT",
    "STATUS_DOWNLOADED", "STATUS_FAIL", "STATUS_PASS", "STATUS_SKIP",
    "STATUS_WOULD_DOWNLOAD", "SUPPORTED_CHECKSUM_ALGORITHMS", "TRANSIENT_STATUS",
    "TransientDownloadError", "ValidationFailed", "ValidationResult",
    "classify_http_status", "fetch_to_path", "file_checksum",
    "looks_like_error_document", "read_head", "redact_headers", "redact_text",
    "redact_url", "sniff_payload_kind", "staging_path_for", "validate_raster_file",
]
