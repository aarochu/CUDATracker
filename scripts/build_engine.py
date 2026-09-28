#!/usr/bin/env python3
"""Build a TensorRT engine from ONNX via trtexec, or the Python TensorRT API."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
from pathlib import Path


def _windows_cuda_dlls() -> None:
    try:
        import torch

        lib = Path(torch.__file__).resolve().parent / "lib"
        if lib.is_dir() and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(lib))
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass
    try:
        import tensorrt_libs

        lib = Path(tensorrt_libs.__file__).resolve().parent
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(str(lib))
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
    except Exception:
        pass


def convert_onnx_fp16(src: Path, dst: Path) -> None:
    """TRT 11 dropped BuilderFlag.FP16; bake FP16 into ONNX (keep FP32 I/O)."""
    try:
        import modelopt.onnx.autocast as autocast
        import onnx

        converted = autocast.convert_to_mixed_precision(
            onnx_path=str(src), low_precision_type="fp16", keep_io_types=True
        )
        onnx.save(converted, str(dst))
        return
    except Exception:
        pass
    import onnx
    from onnxconverter_common import float16

    model = onnx.load(str(src))
    conv = float16.convert_float_to_float16(model, keep_io_types=True)
    onnx.save(conv, str(dst))


def build_with_python(onnx: Path, out: Path, precision: str, workspace_mib: int) -> int:
    _windows_cuda_dlls()
    try:
        import torch

        if torch.cuda.is_available():
            torch.empty(1, device="cuda")
    except Exception as exc:
        print(f"CUDA context init failed: {exc}")
        return 2
    try:
        import tensorrt as trt
    except ImportError:
        print("Python package 'tensorrt' is not importable. Install JetPack TensorRT or pip install tensorrt-cu13 (x86).")
        return 2
    parse_path = onnx
    tmp = None
    use_legacy_fp16 = precision == "fp16" and hasattr(trt.BuilderFlag, "FP16")
    if precision == "fp16" and not use_legacy_fp16:
        tmp = onnx.with_name(onnx.stem + "_fp16_typed.onnx")
        print(f"TensorRT {trt.__version__} is strongly typed; converting {onnx.name} to FP16 ONNX")
        convert_onnx_fp16(onnx, tmp)
        parse_path = tmp
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    flags = 0
    if int(trt.__version__.split(".")[0]) < 10:
        # TensorRT 8.x's ONNX parser needs an explicit-batch network; 10+ is always explicit.
        flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(parse_path.read_bytes()):
        for i in range(parser.num_errors):
            print(parser.get_error(i))
        return 2
    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib << 20)
    if use_legacy_fp16:
        config.set_flag(trt.BuilderFlag.FP16)
    print(f"building TensorRT {trt.__version__} {precision} -> {out}")
    serialized = builder.build_serialized_network(network, config)
    if serialized is None:
        print("build_serialized_network returned None")
        return 2
    out.write_bytes(bytes(serialized))
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


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
    if not trtexec:
        candidate = Path("/usr/src/tensorrt/bin/trtexec")
        if candidate.is_file():
            trtexec = str(candidate)
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
    print("trtexec not on PATH; using Python TensorRT API")
    return build_with_python(onnx, out, args.precision, args.workspace_mib)


if __name__ == "__main__":
    raise SystemExit(main())
