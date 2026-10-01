"""Tests for resumable archive downloads, content validation and redaction.

Every test here is **offline**.  The HTTP layer is a scripted in-memory fake
(:class:`FakeSession`) that records the request headers it was given, so the
Range header, the attempt count and the backoff schedule are asserted directly
rather than inferred.  No test opens a socket; the planetary data hosts this
module will eventually talk to are denied by policy in this environment
(DECISIONS.md D-001), and a unit test must not depend on them in any case.

The scenarios that matter most are the ones a size-only validator passes and a
real archive fails: an HTML 403 page saved under an ``.IMG`` name, a server that
silently ignores ``Range``, and a checksum mismatch that must leave *nothing*
at the final path.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin

from crater import download as dl
from crater.body import MOON

# --------------------------------------------------------------------------- #
# Fixtures: payloads
# --------------------------------------------------------------------------- #
HTML_403 = (
    b"<!DOCTYPE html>\n<html><head><title>403 Forbidden</title></head>\n"
    b"<body><h1>Access denied</h1><p>Request blocked by policy.</p></body></html>\n"
)

S3_ERROR_XML = (
    b'<?xml version="1.0" encoding="UTF-8"?>\n'
    b"<Error><Code>AccessDenied</Code><Message>Access Denied</Message>"
    b"<RequestId>ABC123</RequestId></Error>"
)

#: A URL shaped like a real presigned archive URL.  The secret parts must never
#: appear in a message, a log line or a manifest.
SIGNED_URL = (
    "https://asc-pds-services.s3.us-west-2.amazonaws.com/wms_basemaps/Moon/x.tif"
    "?X-Amz-Algorithm=AWS4-HMAC-SHA256"
    "&X-Amz-Credential=AKIAIOSFODNN7EXAMPLE%2F20261001%2Fus-west-2%2Fs3%2Faws4_request"
    "&X-Amz-Signature=5d41402abc4b2a76b9719d911017c592"
    "&X-Amz-Security-Token=FQoGZXIvYXdzSECRETTOKEN"
    "&versionId=7"
)

SECRETS_IN_SIGNED_URL = (
    "5d41402abc4b2a76b9719d911017c592",
    "AKIAIOSFODNN7EXAMPLE",
    "FQoGZXIvYXdzSECRETTOKEN",
)


@pytest.fixture(scope="module")
def geotiff_bytes(tmp_path_factory) -> bytes:
    """A small, genuinely readable lunar GeoTIFF, as bytes.

    Built with the project's south polar stereographic CRS on the Moon 2000
    sphere (never an Earth EPSG code, DECISIONS.md D-006), so the raster check
    is exercised against a product shaped like the real thing.
    """
    path = tmp_path_factory.mktemp("payload") / "source.tif"
    rng = np.random.default_rng(20261001)
    data = rng.integers(0, 255, size=(32, 48), dtype="uint8")
    with rasterio.open(
        path, "w", driver="GTiff", width=48, height=32, count=1, dtype="uint8",
        crs=rasterio.crs.CRS.from_proj4(MOON.polar_stereographic_proj4()),
        transform=from_origin(-2400.0, -1600.0, 100.0, 100.0),
    ) as dataset:
        dataset.write(data, 1)
    return path.read_bytes()


# --------------------------------------------------------------------------- #
# Fixtures: a scripted, offline HTTP layer
# --------------------------------------------------------------------------- #
class FakeNetworkError(OSError):
    """A transient network failure, shaped like a ``requests`` error (an OSError)."""


@dataclass
class Plan:
    """One scripted response.

    ``supports_range`` false means the server ignores a ``Range`` header and
    answers 200 with the whole entity -- the behaviour that silently corrupts a
    naive resumable downloader.
    """

    status: int = 200
    supports_range: bool = True
    advertise_accept_ranges: bool = True
    truncate_at: int | None = None   # stop the stream early, keep Content-Length
    raise_after: int | None = None   # raise FakeNetworkError mid-stream
    body: bytes | None = None        # override the served content
    headers: dict[str, str] = field(default_factory=dict)


class FakeResponse:
    def __init__(self, status_code: int, headers: dict[str, str], body: bytes,
                 truncate_at: int | None = None, raise_after: int | None = None) -> None:
        self.status_code = status_code
        self.headers = headers
        self._body = body
        self._truncate_at = truncate_at
        self._raise_after = raise_after
        self.closed = False

    def iter_content(self, chunk_size: int = 1024):
        """Yield the body, honouring an early cut or a mid-stream failure.

        Chunks are clipped to the cut point so a large ``chunk_size`` cannot
        accidentally deliver the whole body before the failure fires.
        """
        limit = len(self._body) if self._truncate_at is None else self._truncate_at
        if self._raise_after is not None:
            limit = min(limit, self._raise_after)
        sent = 0
        while sent < limit:
            step = min(max(1, chunk_size), limit - sent)
            yield self._body[sent:sent + step]
            sent += step
        if self._raise_after is not None:
            raise FakeNetworkError("connection reset by peer")

    def close(self) -> None:
        self.closed = True


class FakeSession:
    """An offline ``requests.Session`` stand-in with a scripted response list.

    ``plans`` are consumed in order; the last one repeats once exhausted, so a
    test only scripts the responses it cares about.
    """

    def __init__(self, content: bytes = b"", plans: list[Plan] | None = None) -> None:
        self.content = content
        self.plans = list(plans or [Plan()])
        self.requests: list[dict] = []

    def _plan_for(self, index: int) -> Plan:
        return self.plans[min(index, len(self.plans) - 1)]

    def get(self, url, headers=None, stream=False, timeout=None):
        index = len(self.requests)
        self.requests.append(
            {"url": url, "headers": dict(headers or {}), "stream": stream, "timeout": timeout}
        )
        plan = self._plan_for(index)
        body = self.content if plan.body is None else plan.body
        if plan.status not in (200, 206):
            return FakeResponse(plan.status, {"Content-Length": "0"}, b"")

        requested = (headers or {}).get("Range")
        start = 0
        if requested and plan.supports_range:
            start = int(str(requested).split("=", 1)[1].split("-", 1)[0])
            if start >= len(body):
                return FakeResponse(416, {}, b"")
            slice_ = body[start:]
            resp_headers = {
                "Content-Length": str(len(slice_)),
                "Content-Range": f"bytes {start}-{len(body) - 1}/{len(body)}",
                "Accept-Ranges": "bytes",
            }
            resp_headers.update(plan.headers)
            return FakeResponse(206, resp_headers, slice_, plan.truncate_at, plan.raise_after)

        resp_headers = {"Content-Length": str(len(body))}
        if plan.advertise_accept_ranges:
            resp_headers["Accept-Ranges"] = "bytes" if plan.supports_range else "none"
        resp_headers.update(plan.headers)
        return FakeResponse(200, resp_headers, body, plan.truncate_at, plan.raise_after)


class ExplodingSession:
    """A session that fails the test if it is used at all."""

    def get(self, *args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("the network must not be touched in this scenario")


@pytest.fixture
def recorded_sleeps():
    """A sleep function that records its delays instead of waiting."""
    delays: list[float] = []
    return delays, delays.append


@pytest.fixture
def log_lines():
    lines: list[str] = []
    return lines, lines.append


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


# --------------------------------------------------------------------------- #
# Status classification
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("status,expected", [
    (200, dl.CLASSIFICATION_OK), (206, dl.CLASSIFICATION_OK),
    (403, dl.CLASSIFICATION_PERMANENT), (404, dl.CLASSIFICATION_PERMANENT),
    (401, dl.CLASSIFICATION_PERMANENT), (451, dl.CLASSIFICATION_PERMANENT),
    (418, dl.CLASSIFICATION_PERMANENT),
    (429, dl.CLASSIFICATION_TRANSIENT), (500, dl.CLASSIFICATION_TRANSIENT),
    (503, dl.CLASSIFICATION_TRANSIENT), (504, dl.CLASSIFICATION_TRANSIENT),
    (599, dl.CLASSIFICATION_TRANSIENT),
])
def test_status_classification(status, expected):
    assert dl.classify_http_status(status) == expected


def test_permanent_and_transient_sets_are_disjoint():
    assert not (dl.PERMANENT_STATUS & dl.TRANSIENT_STATUS)


# --------------------------------------------------------------------------- #
# Redaction
# --------------------------------------------------------------------------- #
def test_redact_url_keeps_host_and_path_drops_secrets():
    safe = dl.redact_url(SIGNED_URL)
    for secret in SECRETS_IN_SIGNED_URL:
        assert secret not in safe
    # The diagnosable parts survive: you can still tell which product failed.
    assert "asc-pds-services.s3.us-west-2.amazonaws.com" in safe
    assert "/wms_basemaps/Moon/x.tif" in safe
    assert "versionId=7" in safe
    assert safe.count(dl.REDACTED) >= 3


def test_redact_url_strips_userinfo():
    assert dl.redact_url("https://pilot:hunter2@pds.lroc.asu.edu/x.IMG") == (
        f"https://{dl.REDACTED}@pds.lroc.asu.edu/x.IMG"
    )
    assert "hunter2" not in dl.redact_url("https://pilot:hunter2@pds.lroc.asu.edu/x.IMG")


def test_redact_text_handles_bare_secret_fragments():
    text = "server said Signature=5d41402abc and token: FQoGZXIvYXdz"
    out = dl.redact_text(text)
    assert "5d41402abc" not in out
    assert "FQoGZXIvYXdz" not in out
    assert out.count(dl.REDACTED) == 2


def test_redact_headers_hides_authorization():
    out = dl.redact_headers({"Authorization": "Bearer abc.def", "Range": "bytes=10-"})
    assert out["Authorization"] == dl.REDACTED
    assert out["Range"] == "bytes=10-"


def test_token_in_url_is_redacted_from_a_permanent_error(tmp_path, log_lines):
    """A 403 on a presigned URL must not leak the signature into the message."""
    lines, logger = log_lines
    session = FakeSession(content=b"", plans=[Plan(status=403)])
    with pytest.raises(dl.PermanentDownloadError) as excinfo:
        dl.fetch_to_path(
            SIGNED_URL, tmp_path / "x.tif", session=session,
            policy=dl.RetryPolicy(max_attempts=3), logger=logger,
            sleep=lambda _s: None,
        )
    message = str(excinfo.value)
    for secret in SECRETS_IN_SIGNED_URL:
        assert secret not in message, "a credential leaked into the exception"
        assert all(secret not in line for line in lines), "a credential leaked into the log"
    assert dl.REDACTED in message
    assert "403" in message
    # And the log says something useful about which product failed.
    assert any("/wms_basemaps/Moon/x.tif" in line for line in lines)


def test_outcome_url_is_stored_redacted(tmp_path, geotiff_bytes):
    """The outcome goes into the manifest, so its URL must already be safe."""
    session = FakeSession(content=geotiff_bytes)
    outcome = dl.fetch_to_path(SIGNED_URL, tmp_path / "ok.tif", session=session)
    for secret in SECRETS_IN_SIGNED_URL:
        assert secret not in outcome.url
    assert dl.REDACTED in outcome.url


# --------------------------------------------------------------------------- #
# Content sniffing
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("head,expected", [
    (b"II*\x00rest-of-a-little-endian-tiff", dl.KIND_TIFF),
    (b"MM\x00*big-endian-tiff", dl.KIND_TIFF),
    (b"II+\x00bigtiff", dl.KIND_BIGTIFF),
    (b"\x1f\x8b\x08\x00gzip", dl.KIND_GZIP),
    (b"PK\x03\x04zip", dl.KIND_ZIP),
    (HTML_403, dl.KIND_HTML),
    (b"   \n<html><body>oops</body></html>", dl.KIND_HTML),
    (S3_ERROR_XML, dl.KIND_XML),
    (b'{"error": "forbidden"}', dl.KIND_JSON),
    (b"PDS_VERSION_ID = PDS3\r\nRECORD_TYPE = FIXED_LENGTH\r\n", dl.KIND_PDS_LABEL),
    (b"LBLSIZE=2048  FORMAT='BYTE'", dl.KIND_VICAR),
    (b"Object = IsisCube\n  Object = Core\n", dl.KIND_ISIS_CUBE),
    (b"", dl.KIND_EMPTY),
    (b"\x00\x01\x02\x03\xff\xfe", dl.KIND_UNKNOWN_BINARY),
])
def test_sniff_payload_kind(head, expected):
    assert dl.sniff_payload_kind(head) == expected


@pytest.mark.parametrize("payload", [HTML_403, S3_ERROR_XML, b'{"message":"Access Denied"}', b""])
def test_error_documents_are_recognised(payload):
    is_error, reason = dl.looks_like_error_document(payload)
    assert is_error
    assert reason  # always explains itself


def test_real_raster_head_is_not_an_error_document(geotiff_bytes):
    is_error, reason = dl.looks_like_error_document(geotiff_bytes[:4096])
    assert not is_error
    assert "tiff" in reason


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #
def test_html_error_page_saved_as_img_is_rejected(tmp_path):
    """The required check: a 403 page under a .IMG name is not a product."""
    path = tmp_path / "M1096364197LE.IMG"
    path.write_bytes(HTML_403)
    result = dl.validate_raster_file(path)
    assert not result.ok
    failed = result.get(dl.CHECK_NOT_ERROR_DOCUMENT)
    assert failed.failed
    assert "html" in failed.reason
    # Size alone would have passed it: the page is a perfectly plausible length.
    assert result.size_bytes == len(HTML_403) > 0


def test_validation_reports_each_check_individually(tmp_path, geotiff_bytes):
    path = tmp_path / "good.tif"
    path.write_bytes(geotiff_bytes)
    result = dl.validate_raster_file(
        path, expected_size_bytes=len(geotiff_bytes), checksum=sha256_of(geotiff_bytes),
    )
    assert result.ok
    assert {c.name for c in result.checks} == {
        dl.CHECK_EXISTS, dl.CHECK_NOT_ERROR_DOCUMENT, dl.CHECK_SIZE,
        dl.CHECK_CHECKSUM, dl.CHECK_RASTER,
    }
    assert all(c.passed for c in result.checks)
    assert "48x32x1" in result.get(dl.CHECK_RASTER).reason


def test_missing_checksum_and_size_are_skipped_not_invented(tmp_path, geotiff_bytes):
    path = tmp_path / "good.tif"
    path.write_bytes(geotiff_bytes)
    result = dl.validate_raster_file(path)
    assert result.ok
    assert set(result.skipped) == {dl.CHECK_SIZE, dl.CHECK_CHECKSUM}
    assert dl.CHECK_SIZE not in result.checks_performed
    # The reason states that the archive supplied nothing, rather than guessing.
    assert "supplied no" in result.get(dl.CHECK_SIZE).reason


def test_non_raster_binary_under_tif_fails_the_raster_check(tmp_path):
    path = tmp_path / "truncated.tif"
    path.write_bytes(b"\x00\x01not-a-raster" * 64)
    result = dl.validate_raster_file(path)
    assert not result.ok
    assert result.get(dl.CHECK_NOT_ERROR_DOCUMENT).passed  # it is binary, not a page
    assert result.get(dl.CHECK_RASTER).failed
    assert "could not open" in result.get(dl.CHECK_RASTER).reason


def test_size_mismatch_fails(tmp_path, geotiff_bytes):
    path = tmp_path / "good.tif"
    path.write_bytes(geotiff_bytes)
    result = dl.validate_raster_file(path, expected_size_bytes=len(geotiff_bytes) + 1)
    assert not result.ok
    assert result.get(dl.CHECK_SIZE).failed


def test_missing_file_fails_exists_and_skips_the_rest(tmp_path):
    result = dl.validate_raster_file(tmp_path / "absent.tif")
    assert not result.ok
    assert result.get(dl.CHECK_EXISTS).failed
    assert len(result.skipped) == 4


def test_raster_check_skipped_for_non_raster_products(tmp_path):
    path = tmp_path / "bundle.tar.gz"
    path.write_bytes(b"\x1f\x8b\x08\x00" + b"\x00" * 128)
    result = dl.validate_raster_file(path)
    assert result.ok
    assert dl.CHECK_RASTER in result.skipped


def test_file_checksum_rejects_unknown_algorithm(tmp_path):
    path = tmp_path / "x.bin"
    path.write_bytes(b"abc")
    assert dl.file_checksum(path, "md5") == hashlib.md5(b"abc").hexdigest()
    with pytest.raises(ValueError, match="unsupported checksum algorithm"):
        dl.file_checksum(path, "crc32")


# --------------------------------------------------------------------------- #
# Happy path and atomicity
# --------------------------------------------------------------------------- #
def test_download_preserves_bytes_and_publishes_atomically(tmp_path, geotiff_bytes, log_lines):
    lines, logger = log_lines
    dest = tmp_path / "products" / "LRO_WAC.tif"
    session = FakeSession(content=geotiff_bytes)
    outcome = dl.fetch_to_path(
        "https://example.invalid/LRO_WAC.tif", dest, session=session,
        expected_size_bytes=len(geotiff_bytes), checksum=sha256_of(geotiff_bytes),
        logger=logger,
    )
    assert outcome.status == dl.STATUS_DOWNLOADED
    assert outcome.ok and outcome.validation.ok
    # Byte-for-byte preservation, not merely the right length.
    assert dest.read_bytes() == geotiff_bytes
    assert sha256_of(dest.read_bytes()) == sha256_of(geotiff_bytes)
    assert not dl.staging_path_for(dest).exists()
    assert outcome.bytes_transferred == len(geotiff_bytes)
    assert not outcome.resumed
    assert outcome.server_supports_range is True
    with rasterio.open(dest) as dataset:  # the published file really opens
        assert (dataset.width, dataset.height, dataset.count) == (48, 32, 1)


def test_timeouts_are_passed_as_a_connect_read_pair(tmp_path, geotiff_bytes):
    session = FakeSession(content=geotiff_bytes)
    policy = dl.RetryPolicy(connect_timeout_s=3.5, read_timeout_s=17.0)
    dl.fetch_to_path("https://example.invalid/a.tif", tmp_path / "a.tif",
                     session=session, policy=policy)
    assert session.requests[0]["timeout"] == (3.5, 17.0)
    assert session.requests[0]["stream"] is True


def test_already_present_valid_file_costs_no_request(tmp_path, geotiff_bytes):
    dest = tmp_path / "a.tif"
    dest.write_bytes(geotiff_bytes)
    outcome = dl.fetch_to_path(
        "https://example.invalid/a.tif", dest, session=ExplodingSession(),
        checksum=sha256_of(geotiff_bytes),
    )
    assert outcome.status == dl.STATUS_ALREADY_PRESENT
    assert outcome.bytes_transferred == 0


def test_dry_run_reports_without_touching_the_network(tmp_path):
    dest = tmp_path / "a.tif"
    dl.staging_path_for(dest).write_bytes(b"0" * 17)
    outcome = dl.fetch_to_path(
        "https://example.invalid/a.tif", dest, session=ExplodingSession(), dry_run=True,
    )
    assert outcome.status == dl.STATUS_WOULD_DOWNLOAD
    assert outcome.resumed is True
    assert "17 bytes" in outcome.note
    assert not dest.exists()


# --------------------------------------------------------------------------- #
# Resumption
# --------------------------------------------------------------------------- #
def test_resume_from_partial_file_sends_range_and_reassembles_exactly(
    tmp_path, geotiff_bytes
):
    """The core resume case: a partial file on disk, a Range request, exact bytes."""
    dest = tmp_path / "resumed.tif"
    offset = 400
    assert offset < len(geotiff_bytes)
    dl.staging_path_for(dest).write_bytes(geotiff_bytes[:offset])

    session = FakeSession(content=geotiff_bytes)
    outcome = dl.fetch_to_path(
        "https://example.invalid/resumed.tif", dest, session=session,
        checksum=sha256_of(geotiff_bytes), expected_size_bytes=len(geotiff_bytes),
    )

    assert len(session.requests) == 1
    assert session.requests[0]["headers"]["Range"] == f"bytes={offset}-"
    assert outcome.resumed is True
    assert outcome.bytes_transferred == len(geotiff_bytes) - offset
    assert dest.read_bytes() == geotiff_bytes
    assert outcome.attempts[0].status_code == 206
    assert outcome.attempts[0].range_honoured is True


def test_truncated_stream_is_retried_and_resumed(tmp_path, geotiff_bytes, recorded_sleeps):
    """A server that closes early is transient: resume, do not restart."""
    delays, sleep = recorded_sleeps
    dest = tmp_path / "t.tif"
    cut = 300
    session = FakeSession(content=geotiff_bytes, plans=[Plan(truncate_at=cut), Plan()])
    outcome = dl.fetch_to_path(
        "https://example.invalid/t.tif", dest, session=session,
        checksum=sha256_of(geotiff_bytes), sleep=sleep,
        policy=dl.RetryPolicy(max_attempts=3),
    )
    assert len(session.requests) == 2
    assert "Range" not in session.requests[0]["headers"]
    assert session.requests[1]["headers"]["Range"] == f"bytes={cut}-"
    assert dest.read_bytes() == geotiff_bytes
    assert outcome.resumed is True
    assert outcome.attempts[0].classification == dl.CLASSIFICATION_TRANSIENT
    assert "truncated" in outcome.attempts[0].note
    assert delays == [1.0]


def test_mid_stream_connection_reset_is_resumed(tmp_path, geotiff_bytes):
    dest = tmp_path / "reset.tif"
    session = FakeSession(
        content=geotiff_bytes, plans=[Plan(raise_after=200), Plan()],
    )
    outcome = dl.fetch_to_path(
        "https://example.invalid/reset.tif", dest, session=session,
        sleep=lambda _s: None, policy=dl.RetryPolicy(max_attempts=3),
        checksum=sha256_of(geotiff_bytes),
    )
    assert dest.read_bytes() == geotiff_bytes
    assert outcome.resumed is True
    assert session.requests[1]["headers"]["Range"].startswith("bytes=")


def test_server_without_range_support_restarts_cleanly(tmp_path, geotiff_bytes):
    """A server that ignores Range must trigger a restart, never an append.

    The partial file here is deliberately *wrong* bytes, so an implementation
    that appended the full body to it would produce a longer, corrupt file and
    this test would catch it.
    """
    dest = tmp_path / "norange.tif"
    staging = dl.staging_path_for(dest)
    staging.write_bytes(b"X" * 250)

    session = FakeSession(
        content=geotiff_bytes,
        plans=[Plan(supports_range=False, advertise_accept_ranges=True)],
    )
    outcome = dl.fetch_to_path(
        "https://example.invalid/norange.tif", dest, session=session,
        checksum=sha256_of(geotiff_bytes), expected_size_bytes=len(geotiff_bytes),
    )

    assert session.requests[0]["headers"]["Range"] == "bytes=250-"
    assert outcome.server_supports_range is False
    assert outcome.resumed is False
    assert dest.stat().st_size == len(geotiff_bytes)
    assert dest.read_bytes() == geotiff_bytes          # not 250 + len bytes
    assert b"XXXX" not in dest.read_bytes()[:300]
    assert not staging.exists()


def test_stale_partial_longer_than_resource_restarts_on_416(tmp_path, geotiff_bytes):
    dest = tmp_path / "stale.tif"
    dl.staging_path_for(dest).write_bytes(b"Z" * (len(geotiff_bytes) + 10))
    session = FakeSession(content=geotiff_bytes, plans=[Plan()])
    outcome = dl.fetch_to_path(
        "https://example.invalid/stale.tif", dest, session=session,
        sleep=lambda _s: None, policy=dl.RetryPolicy(max_attempts=3),
        checksum=sha256_of(geotiff_bytes),
    )
    # First request asks for the stale range and is told 416; second starts over.
    assert session.requests[0]["headers"]["Range"].startswith("bytes=")
    assert "Range" not in session.requests[1]["headers"]
    assert dest.read_bytes() == geotiff_bytes
    assert outcome.attempts[0].status_code == 416


def test_allow_resume_false_discards_the_partial(tmp_path, geotiff_bytes):
    dest = tmp_path / "fresh.tif"
    dl.staging_path_for(dest).write_bytes(geotiff_bytes[:100])
    session = FakeSession(content=geotiff_bytes)
    dl.fetch_to_path(
        "https://example.invalid/fresh.tif", dest, session=session, allow_resume=False,
        checksum=sha256_of(geotiff_bytes),
    )
    assert "Range" not in session.requests[0]["headers"]
    assert dest.read_bytes() == geotiff_bytes


# --------------------------------------------------------------------------- #
# Retry policy
# --------------------------------------------------------------------------- #
def test_403_is_not_retried(tmp_path, log_lines):
    lines, logger = log_lines
    session = FakeSession(plans=[Plan(status=403)])
    with pytest.raises(dl.PermanentDownloadError, match="permanent"):
        dl.fetch_to_path(
            "https://pds.lroc.asu.edu/EDR/M1.IMG", tmp_path / "M1.IMG", session=session,
            policy=dl.RetryPolicy(max_attempts=5), sleep=lambda _s: None, logger=logger,
        )
    assert len(session.requests) == 1, "a 403 must not be retried as if transient"
    assert any("not retrying" in line for line in lines)


def test_404_is_not_retried(tmp_path):
    session = FakeSession(plans=[Plan(status=404)])
    with pytest.raises(dl.PermanentDownloadError):
        dl.fetch_to_path(
            "https://example.invalid/missing.IMG", tmp_path / "missing.IMG",
            session=session, policy=dl.RetryPolicy(max_attempts=4), sleep=lambda _s: None,
        )
    assert len(session.requests) == 1


def test_503_is_retried_with_exponential_backoff_then_gives_up(tmp_path, recorded_sleeps):
    delays, sleep = recorded_sleeps
    session = FakeSession(plans=[Plan(status=503)])
    policy = dl.RetryPolicy(max_attempts=4, backoff_initial_s=1.0, backoff_factor=2.0)
    with pytest.raises(dl.TransientDownloadError) as excinfo:
        dl.fetch_to_path(
            "https://example.invalid/a.tif", tmp_path / "a.tif", session=session,
            policy=policy, sleep=sleep,
        )
    assert len(session.requests) == 4
    assert delays == [1.0, 2.0, 4.0], "no sleep after the final attempt"
    assert "after 4 attempt(s)" in str(excinfo.value)
    assert not (tmp_path / "a.tif").exists()


def test_backoff_is_capped(tmp_path):
    policy = dl.RetryPolicy(backoff_initial_s=10.0, backoff_factor=10.0, backoff_max_s=30.0)
    assert [policy.backoff_for(n) for n in (1, 2, 3, 4)] == [10.0, 30.0, 30.0, 30.0]


def test_connection_error_is_transient_and_bounded(tmp_path, recorded_sleeps):
    delays, sleep = recorded_sleeps

    class RefusingSession:
        def __init__(self) -> None:
            self.calls = 0

        def get(self, *args, **kwargs):
            self.calls += 1
            raise FakeNetworkError("connect timed out")

    session = RefusingSession()
    with pytest.raises(dl.TransientDownloadError):
        dl.fetch_to_path(
            "https://example.invalid/a.tif", tmp_path / "a.tif", session=session,
            policy=dl.RetryPolicy(max_attempts=2), sleep=sleep,
        )
    assert session.calls == 2
    assert delays == [1.0]


def test_retry_policy_rejects_nonsense():
    with pytest.raises(ValueError, match="max_attempts"):
        dl.RetryPolicy(max_attempts=0)
    with pytest.raises(ValueError, match="backoff_factor"):
        dl.RetryPolicy(backoff_factor=0.5)
    with pytest.raises(ValueError, match="jitter"):
        dl.RetryPolicy(jitter=2.0)


# --------------------------------------------------------------------------- #
# Rejection leaves nothing behind
# --------------------------------------------------------------------------- #
def test_html_error_page_served_under_img_name_is_rejected(tmp_path, log_lines):
    """End to end: the archive returns 200 with a login/error page."""
    lines, logger = log_lines
    dest = tmp_path / "M1096364197LE.IMG"
    session = FakeSession(content=HTML_403)
    with pytest.raises(dl.ValidationFailed) as excinfo:
        dl.fetch_to_path(
            "https://pds.lroc.asu.edu/EDR/M1096364197LE.IMG", dest, session=session,
            logger=logger,
        )
    result = excinfo.value.result
    assert result.get(dl.CHECK_NOT_ERROR_DOCUMENT).failed
    assert not dest.exists(), "a rejected payload must never reach the final path"
    assert not dl.staging_path_for(dest).exists(), "corrupt staging must not seed a resume"
    assert "nothing was written to the final path" in str(excinfo.value)
    assert any("rejected download" in line for line in lines)


def test_s3_error_xml_under_tif_name_is_rejected(tmp_path):
    dest = tmp_path / "mosaic.tif"
    session = FakeSession(content=S3_ERROR_XML)
    with pytest.raises(dl.ValidationFailed):
        dl.fetch_to_path("https://example.invalid/mosaic.tif", dest, session=session)
    assert not dest.exists()


def test_checksum_mismatch_is_rejected_and_leaves_no_final_file(tmp_path, geotiff_bytes):
    """The archive's checksum disagrees: publish nothing, keep nothing."""
    dest = tmp_path / "corrupt.tif"
    session = FakeSession(content=geotiff_bytes)
    wrong = sha256_of(geotiff_bytes + b"!")
    with pytest.raises(dl.ValidationFailed) as excinfo:
        dl.fetch_to_path(
            "https://example.invalid/corrupt.tif", dest, session=session, checksum=wrong,
        )
    result = excinfo.value.result
    assert result.get(dl.CHECK_CHECKSUM).failed
    assert "mismatch" in result.get(dl.CHECK_CHECKSUM).reason
    assert not dest.exists()
    assert not dl.staging_path_for(dest).exists()
    # The bytes themselves were fine as a raster; only the checksum disagreed,
    # which is exactly the case a readability-only validator would wave through.
    assert result.get(dl.CHECK_RASTER).passed


def test_existing_invalid_destination_is_replaced(tmp_path, geotiff_bytes):
    dest = tmp_path / "x.tif"
    dest.write_bytes(HTML_403)  # a previous run published junk
    session = FakeSession(content=geotiff_bytes)
    outcome = dl.fetch_to_path(
        "https://example.invalid/x.tif", dest, session=session,
        checksum=sha256_of(geotiff_bytes),
    )
    assert outcome.status == dl.STATUS_DOWNLOADED
    assert dest.read_bytes() == geotiff_bytes


def test_failed_transfer_keeps_the_partial_for_a_later_resume(tmp_path, geotiff_bytes):
    """Giving up must leave the partial bytes, so a later run can finish them."""
    dest = tmp_path / "later.tif"
    cut = 500
    session = FakeSession(content=geotiff_bytes, plans=[Plan(truncate_at=cut)])
    with pytest.raises(dl.TransientDownloadError, match="Partial file kept"):
        dl.fetch_to_path(
            "https://example.invalid/later.tif", dest, session=session,
            policy=dl.RetryPolicy(max_attempts=1), sleep=lambda _s: None,
        )
    assert len(session.requests) == 1
    assert not dest.exists()
    staging = dl.staging_path_for(dest)
    assert staging.exists() and staging.stat().st_size == cut
    assert staging.read_bytes() == geotiff_bytes[:cut]

    # A later call resumes from exactly those bytes and completes.
    session2 = FakeSession(content=geotiff_bytes)
    outcome = dl.fetch_to_path(
        "https://example.invalid/later.tif", dest, session=session2,
        checksum=sha256_of(geotiff_bytes),
    )
    assert session2.requests[0]["headers"]["Range"] == f"bytes={cut}-"
    assert dest.read_bytes() == geotiff_bytes
    assert outcome.resumed is True
