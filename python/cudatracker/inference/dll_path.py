from __future__ import annotations

import os
from pathlib import Path


def prepend_torch_cuda_dlls() -> None:
    """Windows ORT/TensorRT need cudnn/cublas DLLs; PyTorch already ships them."""
    try:
        import torch

        lib = Path(torch.__file__).resolve().parent / "lib"
        if lib.is_dir():
            os.environ["PATH"] = str(lib) + os.pathsep + os.environ.get("PATH", "")
            if hasattr(os, "add_dll_directory"):
                try:
                    os.add_dll_directory(str(lib))
                except OSError:
                    pass
    except Exception:
        pass
