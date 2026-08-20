#!/usr/bin/env python3
"""Export YOLOv8 to ONNX. NMS stays outside the graph so every backend shares postprocess."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="yolov8n")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--out", default="")
    args = p.parse_args()
    from ultralytics import YOLO

    out_dir = Path("models")
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = args.model if args.model.endswith(".pt") else f"{args.model}.pt"
    yolo = YOLO(weights)
    dest = Path(args.out) if args.out else out_dir / f"{Path(weights).stem}.onnx"
    produced = Path(
        str(
            yolo.export(
                format="onnx",
                imgsz=args.imgsz,
                opset=args.opset,
                simplify=True,
                nms=False,
                dynamic=False,
            )
        )
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not produced.exists():
        print(f"ultralytics export finished but {produced} was not found. Expected {dest}")
        return 2
    if produced.resolve() != dest.resolve():
        dest.write_bytes(produced.read_bytes())
        print(f"copied {produced} -> {dest}")
    print(f"ONNX export done (opset={args.opset}, imgsz={args.imgsz}, nms=false). Path: {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
