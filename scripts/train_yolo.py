#!/usr/bin/env python3
"""Train a crater detector on the CraterDANet LRO-NAC dataset (real human labels).

CPU-only in this container (no GPU present), so this is a modest run with
early stopping, not the ~100-epoch budget the brief anticipates. The same
script runs unchanged on a GPU by passing --device 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="yolo11n.pt")
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--imgsz", type=int, default=512)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--patience", type=int, default=15)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--name", default="crater_yolo11n")
    args = ap.parse_args()

    from ultralytics import YOLO
    import torch, ultralytics

    data = ROOT / "data" / "yolo_external" / "dataset.yaml"
    if not data.exists():
        print(f"missing {data}; run scripts/build_external_dataset.py first")
        return 1

    print(f"ultralytics {ultralytics.__version__} | torch {torch.__version__} | "
          f"device {args.device} | cuda {torch.cuda.is_available()}")

    model = YOLO(args.model)
    model.train(
        data=str(data), epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        patience=args.patience, device=args.device, workers=args.workers,
        project=str(ROOT / "artifacts" / "runs"), name=args.name, exist_ok=True,
        seed=20261001, deterministic=True, pretrained=True, val=True, plots=True,
        # craters are small, dense and rotationally symmetric; flips are
        # physically harmless but they do change apparent illumination
        # direction, so they are kept mild and no shear/perspective is used
        # (that would corrupt diameter interpretation -- brief section 7).
        degrees=0.0, shear=0.0, perspective=0.0, scale=0.2, translate=0.1,
        fliplr=0.5, flipud=0.0, mosaic=0.5, mixup=0.0, erasing=0.0,
        hsv_h=0.0, hsv_s=0.0, hsv_v=0.3,
    )
    out = ROOT / "artifacts" / "runs" / args.name
    print(f"\ntraining finished -> {out}")

    print("\n=== validating on the untouched TEST split ===")
    metrics = model.val(data=str(data), split="test", imgsz=args.imgsz,
                        device=args.device, project=str(ROOT / "artifacts" / "runs"),
                        name=f"{args.name}_test", exist_ok=True)
    summary = {
        "ultralytics": ultralytics.__version__, "torch": torch.__version__,
        "model": args.model, "imgsz": args.imgsz, "epochs_requested": args.epochs,
        "device": args.device, "seed": 20261001,
        "test_metrics": {
            "mAP50": float(metrics.box.map50), "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp), "recall": float(metrics.box.mr),
        },
        "label_provenance": "human annotations, CraterDANet dataset (GPL-3.0)",
        "caveat": ("Metrics are for the Chang'E-4 region at 0.5 m/px. They are NOT "
                   "Malapert performance: different latitude, illumination and "
                   "pixel scale. Malapert performance is UNVALIDATED."),
    }
    (ROOT / "artifacts" / "training_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["test_metrics"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
