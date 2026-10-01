#!/usr/bin/env python3
"""Acquire the selected Malapert controlled-mosaic variants with provenance.

Downloads, for each approved illumination variant (DECISIONS.md D-013b):
  * the PDS4 detached label (.xml)      -- authoritative metadata + MD5
  * the product image (.IMG, uint16)    -- authoritative pixel data (D-014)
  * the valid-data mask (.MASK.TIF)     -- authoritative validity layer (D-011)

Checksums come from the product's own PDS4 label, so integrity is verified
against the archive's published digest rather than merely against file size.

Usage:
    python3 scripts/acquire_malapert.py [--dry-run] [--validate-only]
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from crater import download as dl
from crater import manifest as mf

ARCHIVE = (
    "https://pds.mcp.nasa.gov/data/store/img/lunar_reconnaissance_orbiter/"
    "pds4/lroc/lro-l-lroc-5-rdr/LROLRC_2001"
)
VARIANTS = ["MALAPERTLO1", "MALAPERTLOA", "MALAPERTLOH"]  # D-013b
OUT = ROOT / "data" / "raw" / "malapert"
MANIFEST = ROOT / "data" / "manifests" / "manifest.csv"

# ROI overlap: every variant is on the identical 21000x21000 ROI grid, so the
# footprint overlap with the ROI is exactly 1.0 by construction.
ROI_OVERLAP = "1.0"


def urls(v: str) -> dict[str, str]:
    stem = f"NAC_ROI_{v}_P860S0003"
    return {
        "label": f"{ARCHIVE}/DATA/BDR/NAC_ROI/{v}/{stem}.xml",
        "image": f"{ARCHIVE}/DATA/BDR/NAC_ROI/{v}/{stem}.IMG",
        "mask": f"{ARCHIVE}/EXTRAS/BROWSE/NAC_ROI/{v}/{stem}.MASK.TIF",
    }


def parse_label(text: str) -> dict[str, str]:
    """Pull the fields we rely on out of the PDS4 label.

    The image size and digest come from the label's <File> block
    (<file_size>, <md5_checksum>).  Note that <object_length> is NOT the file
    size -- it is the length of the PDS3 attached header (42000 bytes), and
    using it caused the download validator to reject three good transfers.
    """
    def one(tag: str, scope: str | None = None) -> str:
        hay = scope if scope is not None else text
        m = re.search(rf"<[\w]*:?{tag}(?:\s[^>]*)?>(.*?)</[\w]*:?{tag}>", hay, re.S)
        return " ".join(m.group(1).split()) if m else mf.UNKNOWN

    fm = re.search(r"<File>.*?</File>", text, re.S)
    fblock = fm.group(0) if fm else ""

    out = {
        "processing_level": one("processing_level"),
        "start": one("start_date_time"),
        "stop": one("stop_date_time"),
        "pixel_scale": one("pixel_scale_x"),
        "image_name": one("file_name", fblock),
        "image_bytes": one("file_size", fblock),
        "image_md5": one("md5_checksum", fblock),
    }

    # Cross-check the declared size against the array geometry, so a label
    # defect cannot silently set a wrong expectation.
    els = [int(x) for x in re.findall(r"<elements>(\d+)</elements>", text)]
    hdr = one("object_length")
    if len(els) >= 2 and not mf.is_unknown(hdr):
        predicted = int(hdr) + els[0] * els[1] * 2  # UnsignedLSB2 = 2 bytes
        out["size_crosscheck"] = str(predicted)
        if not mf.is_unknown(out["image_bytes"]) and int(out["image_bytes"]) != predicted:
            out["size_crosscheck_mismatch"] = (
                f"label file_size {out['image_bytes']} != "
                f"header {hdr} + {els[0]}x{els[1]}x2 = {predicted}"
            )
    return out


def human(n: int | None) -> str:
    return "unknown" if n is None else f"{n/2**20:,.1f} MiB"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--validate-only", action="store_true")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    entries: list[mf.ManifestEntry] = []

    for v in VARIANTS:
        u = urls(v)
        stem = f"NAC_ROI_{v}_P860S0003"
        label_path = OUT / f"{stem}.xml"
        image_path = OUT / f"{stem}.IMG"
        mask_path = OUT / f"{stem}.MASK.TIF"

        print(f"\n=== {v} ===", flush=True)

        # 1. label (small, always fetched first: it carries the digests)
        if not args.validate_only:
            r = dl.fetch_to_path(u["label"], label_path, expect_raster=False,
                                 dry_run=args.dry_run,
                                 logger=lambda m: print("   ", m, flush=True))
            print(f"    label: {r.status if hasattr(r,'status') else r}", flush=True)
        meta = parse_label(label_path.read_text()) if label_path.exists() else {}
        exp_bytes = meta.get("image_bytes", mf.UNKNOWN)
        exp_md5 = meta.get("image_md5", mf.UNKNOWN)
        print(f"    label: {meta.get('image_name','?')} {exp_bytes} bytes md5={exp_md5}", flush=True)
        if "size_crosscheck" in meta:
            ok = "size_crosscheck_mismatch" not in meta
            print(f"    size cross-check vs array geometry: "
                  f"{meta['size_crosscheck']} -> {'OK' if ok else meta['size_crosscheck_mismatch']}",
                  flush=True)

        # 2. mask (small)
        if not args.validate_only:
            dl.fetch_to_path(u["mask"], mask_path, expect_raster=True,
                             dry_run=args.dry_run,
                             logger=lambda m: print("   ", m, flush=True))

        # 3. image (large) -- validated against the label's own MD5
        status = "pending" if not args.validate_only else "validate_only"
        notes = []
        if not args.validate_only:
            t0 = time.time()
            last = [0.0]

            def prog(done: int, total: int | None) -> None:
                now = time.time()
                if now - last[0] > 20:
                    last[0] = now
                    pct = f"{100*done/total:.1f}%" if total else "?"
                    rate = done / max(now - t0, 1e-9) / 2**20
                    print(f"      {human(done)} / {human(total)} ({pct})  {rate:,.1f} MiB/s",
                          flush=True)

            try:
                res = dl.fetch_to_path(
                    u["image"], image_path,
                    expected_size_bytes=None if mf.is_unknown(exp_bytes) else int(exp_bytes),
                    checksum=None if mf.is_unknown(exp_md5) else exp_md5,
                    checksum_algorithm="md5",
                    expect_raster=False,          # detached PDS4 label; see D-015
                    dry_run=args.dry_run,
                    progress=prog,
                    logger=lambda m: print("   ", m, flush=True),
                )
                status = "would_download" if args.dry_run else "accepted"
                if args.dry_run:
                    notes.append("dry run: nothing fetched")
                else:
                    notes.append(f"md5 verified against PDS4 label; {time.time()-t0:.0f}s")
                print(f"    image OK in {time.time()-t0:.0f}s", flush=True)
            except Exception as exc:
                status = "failed"
                notes.append(f"download failed: {dl.redact_text(str(exc))[:200]}")
                print(f"    image FAILED: {exc}", flush=True)

        size = image_path.stat().st_size if image_path.exists() else None
        entries.append(mf.ManifestEntry(
            product_id=f"bdr.nac_roi.{v.lower()}.nac_roi_{v.lower()}_p860s0003",
            observation_id=v,
            processing_level=meta.get("processing_level", mf.UNKNOWN),
            source_url=u["image"],
            metadata_url=u["label"],
            acquisition_time=meta.get("start", mf.UNKNOWN),
            nominal_resolution_m="1.0",
            incidence_deg=mf.UNKNOWN,   # not published in this RDR label
            emission_deg=mf.UNKNOWN,
            phase_deg=mf.UNKNOWN,
            footprint_reference="x[-11000,10000] y[111000,132000] m, polar stereographic, GeoTIFF geotransform (D-009)",
            file_size_bytes=str(size) if size else mf.UNKNOWN,
            checksum=f"md5:{exp_md5}" if not mf.is_unknown(exp_md5) else mf.UNKNOWN,
            local_path=str(image_path.relative_to(ROOT)) if image_path.exists() else mf.UNKNOWN,
            download_status=status,
            roi_overlap=ROI_OVERLAP,
            quality_notes="; ".join(notes) or "controlled mosaic, DN not I/F (D-016)",
        ))

    mf.write_manifest(entries, MANIFEST)
    print(f"\nwrote {MANIFEST}")
    val = mf.validate_manifest(mf.read_manifest(MANIFEST))
    print("manifest valid:", val.ok if hasattr(val, "ok") else val)
    s = mf.summarise_manifest(mf.read_manifest(MANIFEST))
    print("summary:", s)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
