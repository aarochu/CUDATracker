from __future__ import annotations

import ctypes.util
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
CUDA_DIR = REPO_ROOT / "cuda"


def kernel_source() -> str:
    cuh = (CUDA_DIR / "kernels.cuh").read_text(encoding="utf-8")
    parts = [cuh]
    for name in (
        "resize.cu",
        "colorspace.cu",
        "normalize.cu",
        "layout.cu",
        "fused_preprocess.cu",
    ):
        text = (CUDA_DIR / name).read_text(encoding="utf-8")
        stripped = []
        for line in text.splitlines():
            if line.strip().startswith("#include"):
                continue
            stripped.append(line)
        parts.append("\n".join(stripped))
    return "\n\n".join(parts)


def _nvrtc_dirs() -> list[Path]:
    cuda_path = os.environ.get("CUDA_PATH")
    dirs: list[Path] = []
    if cuda_path:
        dirs.append(Path(cuda_path) / "bin")
    try:
        import torch

        dirs.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass
    # CUDA 13 on Windows keeps DLLs in bin/x64; Linux and Jetson toolkits use lib64.
    for root in (cuda_path, os.environ.get("CUDA_HOME"), "/usr/local/cuda"):
        if root:
            dirs += [Path(root) / "bin" / "x64", Path(root) / "lib64"]
    try:
        import nvidia  # pip CUDA wheels, e.g. nvidia/cuda_nvrtc/lib

        for base in nvidia.__path__:
            dirs += sorted(Path(base).glob("*/lib"))
    except Exception:
        pass
    return dirs


def find_nvrtc_dll() -> str:
    candidates = _nvrtc_dirs()
    names = [
        "nvrtc64_130_0.dll",
        "nvrtc64_120_0.dll",
        "nvrtc64_112_0.dll",
        "libnvrtc.so.13",
        "libnvrtc.so.12",
        "libnvrtc.so",
    ]
    for folder in candidates:
        for name in names:
            p = folder / name
            if p.exists():
                return str(p)
        for p in folder.glob("nvrtc64_*.dll"):
            return str(p)
        for p in folder.glob("libnvrtc.so*"):
            return str(p)
    found = ctypes.util.find_library("nvrtc")
    if found:
        return found
    raise FileNotFoundError(
        "nvrtc library not found in CUDA_PATH, CUDA_HOME, /usr/local/cuda, PyTorch, or pip nvidia wheels"
    )
