from __future__ import annotations

import os
from pathlib import Path


def _prepend_dir(path: Path) -> None:
    if not path.is_dir():
        return
    os.environ["PATH"] = str(path) + os.pathsep + os.environ.get("PATH", "")
    if hasattr(os, "add_dll_directory"):
        try:
            os.add_dll_directory(str(path))
        except OSError:
            pass


def prepend_torch_cuda_dlls() -> None:
    """Windows ORT/TensorRT need cudnn/cublas and nvinfer; PyTorch and pip TensorRT ship them."""
    try:
        import torch

        _prepend_dir(Path(torch.__file__).resolve().parent / "lib")
    except Exception:
        pass
    try:
        import tensorrt_libs

        _prepend_dir(Path(tensorrt_libs.__file__).resolve().parent)
    except Exception:
        pass
