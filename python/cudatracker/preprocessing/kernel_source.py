from __future__ import annotations

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


def find_nvrtc_dll() -> str:
    env = os.environ.get("CUDA_PATH")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env) / "bin")
    try:
        import torch

        candidates.append(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass
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
    raise FileNotFoundError("nvrtc library not found next to PyTorch or CUDA_PATH")
