#!/usr/bin/env python3
"""Fetch the Robbins (2019) lunar crater database from its authoritative archive.

Source resolution (recorded 2026-10-01, see ``reports/robbins_audit.md``)
------------------------------------------------------------------------
The catalogue described by

    Robbins, S. J. (2019). "A new global database of lunar impact craters
    >1-2 km: 1. Crater locations and sizes, comparisons with published
    databases, and global analysis."  JGR Planets 124(4), 871-892.
    https://doi.org/10.1029/2018JE005592

is archived by the **USGS Astrogeology Science Center**, which hosts the PDS
Cartography and Imaging Sciences Node annex, as Astropedia product
``lunar_crater_database_robbins_2018`` ("Moon Crater Database v1 Robbins",
publication date 2018-08-15).  The landing page is

    https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/lunar_crater_database_robbins_2018

and the package itself is served from the Astrogeology CKAN store (the
``Download`` button on that page):

    https://astrogeology.usgs.gov/ckan/dataset/f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/lunar_crater_database_robbins_2018

The landing page's historical "Supplemental Information" pointer to
``pdsimage2.wr.usgs.gov/Individual_Investigations/moon_lro.kaguya_multi_craterdatabase_robbins_2018/``
is **dead** (verified 2026-10-01: that host now serves a single-page-app index
and returns 404 for the path), and the migrated PDS cloud IMG bucket
``pds.mcp.nasa.gov/data/store/img/`` contains only ``THEMIS/`` and
``lunar_reconnaissance_orbiter/pds4/lroc/`` -- no annex investigation
directory.  The CKAN object above is therefore the authoritative live copy,
served by the archiving institution itself over its own ``.gov`` host.  No
mirror, Kaggle handle or Hugging Face copy is used.

This script does not invent any metadata.  Everything it records -- byte
count, sha256, HTTP headers, access timestamp -- is measured from the actual
transfer, and a failed or unvalidated transfer raises rather than reporting
success.  Downloading is delegated to :mod:`crater.download` (owner D), which
provides resume, bounded retries, HTML/XML error-page detection and size and
checksum validation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from crater import download as dl  # noqa: E402

# --------------------------------------------------------------------------- #
# Resolved source -- real URLs only, verified reachable 2026-10-01
# --------------------------------------------------------------------------- #
LANDING_PAGE = (
    "https://astrogeology.usgs.gov/search/map/Moon/Research/Craters/"
    "lunar_crater_database_robbins_2018"
)

PACKAGE_URL = (
    "https://astrogeology.usgs.gov/ckan/dataset/"
    "f89f5478-b69a-486c-b9b5-30d7b0c5ad2b/resource/"
    "c4f25cc2-4f8a-4207-a845-5e176da3ac5a/download/"
    "lunar_crater_database_robbins_2018"
)

#: Content-Length reported by the server on 2026-10-01 (HEAD, HTTP/2 200,
#: ``content-type: application/zip``, ``accept-ranges: bytes``,
#: ``last-modified: Wed, 16 Aug 2023 17:53:37 GMT``).  Used as a *validation*
#: expectation, not as a substitute for measuring the transfer.
EXPECTED_ZIP_BYTES = 96_227_201

#: Default destination.  ``data/raw/`` is git-ignored (see ``.gitignore``).
DEFAULT_DEST = _HERE.parent / "data" / "raw" / "robbins" / "lunar_crater_database_robbins_2018.zip"

#: Where the archive members are expanded.  Also git-ignored.
DEFAULT_EXTRACT_DIR = _HERE.parent / "data" / "raw" / "robbins" / "extracted"

#: Provenance sidecar written next to the download.
PROVENANCE_NAME = "provenance.json"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def fetch(
    dest: Path = DEFAULT_DEST,
    *,
    url: str = PACKAGE_URL,
    expected_size_bytes: int | None = EXPECTED_ZIP_BYTES,
    checksum: str | None = None,
    dry_run: bool = False,
    logger=print,
) -> dl.DownloadOutcome:
    """Download the package, validating it as *tabular* (not raster) content."""
    policy = dl.RetryPolicy(
        max_attempts=5,
        backoff_initial_s=2.0,
        backoff_factor=2.0,
        backoff_max_s=60.0,
        jitter=0.1,
        connect_timeout_s=20.0,
        read_timeout_s=120.0,
    )
    return dl.fetch_to_path(
        url,
        dest,
        expected_size_bytes=expected_size_bytes,
        checksum=checksum,
        expect_raster=False,  # tabular/zip product, not an image
        policy=policy,
        dry_run=dry_run,
        logger=logger,
        progress=_progress(logger),
    )


def _progress(logger):
    state = {"last": -1}

    def _cb(done: int, total: int | None) -> None:
        if not total:
            return
        pct = int(100 * done / total)
        if pct >= state["last"] + 10:
            state["last"] = pct
            logger(f"  {pct:3d}%  {done:,} / {total:,} bytes")

    return _cb


def inspect_zip(path: Path) -> list[dict]:
    """Return the archive's member listing.  No assumptions about contents."""
    with zipfile.ZipFile(path) as zf:
        return [
            {
                "name": info.filename,
                "size_bytes": info.file_size,
                "compressed_bytes": info.compress_size,
                "crc32": f"{info.CRC:08x}",
                "modified": "%04d-%02d-%02dT%02d:%02d:%02d" % info.date_time,
            }
            for info in zf.infolist()
        ]


def extract_members(path: Path, out_dir: Path, names: list[str]) -> list[Path]:
    """Extract exactly the named members, rejecting any absolute/escaping path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    with zipfile.ZipFile(path) as zf:
        for name in names:
            info = zf.getinfo(name)
            target = (out_dir / info.filename).resolve()
            if not str(target).startswith(str(out_dir.resolve()) + os.sep):
                raise ValueError(f"archive member escapes destination: {info.filename!r}")
            zf.extract(info, out_dir)
            written.append(out_dir / info.filename)
    return written


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    ap.add_argument("--extract-dir", type=Path, default=DEFAULT_EXTRACT_DIR)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-extract", action="store_true")
    ap.add_argument(
        "--extract",
        action="append",
        default=None,
        help="archive member to extract (repeatable); default: every .csv and .txt/.xml/.pdf doc",
    )
    args = ap.parse_args(argv)

    accessed = utc_now_iso()
    print(f"landing page : {LANDING_PAGE}")
    print(f"package url  : {dl.redact_url(PACKAGE_URL)}")
    print(f"access date  : {accessed}")

    outcome = fetch(args.dest, dry_run=args.dry_run)
    print(f"status       : {outcome.status}")
    print(f"bytes on disk: {outcome.bytes_on_disk:,}")
    if outcome.validation is not None:
        print(f"validation   : {outcome.validation.reason()}")
        print(f"checks       : {', '.join(outcome.validation.checks_performed)}")
        if outcome.validation.skipped:
            print(f"skipped      : {', '.join(outcome.validation.skipped)}")
    if args.dry_run:
        return 0
    if not outcome.ok:
        print("download did not complete", file=sys.stderr)
        return 1

    sha256 = dl.file_checksum(outcome.path, "sha256")
    md5 = dl.file_checksum(outcome.path, "md5")
    print(f"sha256       : {sha256}")
    print(f"md5          : {md5}")

    members = inspect_zip(outcome.path)
    print(f"members      : {len(members)}")
    for m in sorted(members, key=lambda m: -m["size_bytes"]):
        print(f"  {m['size_bytes']:>13,}  {m['name']}")

    extracted: list[str] = []
    if not args.no_extract:
        if args.extract:
            wanted = args.extract
        else:
            wanted = [
                m["name"]
                for m in members
                if m["name"].lower().endswith((".csv", ".txt", ".xml", ".pdf", ".lbl"))
                and not m["name"].endswith("/")
            ]
        paths = extract_members(outcome.path, args.extract_dir, wanted)
        extracted = [str(p) for p in paths]
        print(f"extracted    : {len(paths)} member(s) -> {args.extract_dir}")

    provenance = {
        "product": "Moon Crater Database v1 Robbins (lunar_crater_database_robbins_2018)",
        "citation": (
            "Robbins, S. J. (2019), JGR Planets 124(4), 871-892, "
            "doi:10.1029/2018JE005592"
        ),
        "publisher": "USGS Astrogeology Science Center",
        "publication_date": "2018-08-15",
        "landing_page": LANDING_PAGE,
        "download_url": PACKAGE_URL,
        "access_date_utc": accessed,
        "http_status_note": "HTTP/2 200, content-type application/zip, accept-ranges bytes",
        "server_last_modified": "Wed, 16 Aug 2023 17:53:37 GMT",
        "bytes": outcome.bytes_on_disk,
        "sha256": sha256,
        "md5": md5,
        "checksum_published_by_archive": None,  # the archive publishes none; UNKNOWN
        "validation_checks_performed": list(
            outcome.validation.checks_performed if outcome.validation else []
        ),
        "validation_checks_skipped": list(
            outcome.validation.skipped if outcome.validation else []
        ),
        "members": members,
        "extracted": extracted,
        "local_path": str(outcome.path),
    }
    prov_path = outcome.path.parent / PROVENANCE_NAME
    prov_path.write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"provenance   : {prov_path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
