#!/usr/bin/env python3
"""Build a TensorRT engine from ONNX via trtexec, or tell ORT-TRT to cache one."""

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--onnx", default="models/yolov8n.onnx")
    p.add_argument("--precision", choices=["fp32", "fp16"], default="fp16")
    p.add_argument("--out", default="")
    p.add_argument("--workspace-mib", type=int, default=1024)
    args = p.parse_args()
    onnx = Path(args.onnx)
    if not onnx.exists():
        print(f"missing {onnx}. Run python scripts/export_onnx.py first.")
        return 2
    out = Path(args.out) if args.out else Path("models") / f"{onnx.stem}_{args.precision}.engine"
    out.parent.mkdir(parents=True, exist_ok=True)
    trtexec = shutil.which("trtexec")
    if trtexec:
        cmd = [
            trtexec,
            f"--onnx={onnx}",
            f"--saveEngine={out}",
            f"--memPoolSize=workspace:{args.workspace_mib}M",
        ]
        if args.precision == "fp16":
            cmd.append("--fp16")
        print(" ".join(cmd))
        return subprocess.call(cmd)
    print("trtexec not on PATH. Native .engine not built.")
    print("The TensorRT backend will use ONNX Runtime's TensorRT EP, which writes an engine cache under models/ort_trt_cache.")
    print(f"First run: python -m cudatracker --backend tensorrt --precision {args.precision} --bench --source <video>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
