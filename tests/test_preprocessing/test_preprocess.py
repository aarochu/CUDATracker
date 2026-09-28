from __future__ import annotations

import numpy as np
import pytest

from cudatracker.preprocessing.cpu import cpu_preprocess
from cudatracker.preprocessing.geometry import letterbox_geometry


def test_cpu_tensor_shape():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    img[:] = (20, 80, 160)
    tensor, meta = cpu_preprocess(img, 640, True, 114)
    assert tensor.shape == (1, 3, 640, 640)
    assert tensor.dtype == np.float32
    assert tensor.max() <= 1.0 + 1e-5
    assert meta.imgsz == 640


def test_cpu_vs_cuda_close():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    from cudatracker.preprocessing.cuda import CudaPreprocessor

    rng = np.random.RandomState(0)
    img = rng.randint(0, 255, size=(247, 333, 3), dtype=np.uint8)
    cpu, meta = cpu_preprocess(img, 640, True, 114)
    pre = CudaPreprocessor(640, 114, fused=False, letterbox=True)
    gpu, meta2, _, _ = pre(img)
    gpu_np = gpu.detach().float().cpu().numpy()
    assert meta.pad_x == meta2.pad_x and meta.pad_y == meta2.pad_y
    assert meta.new_w == meta2.new_w and meta.new_h == meta2.new_h
    err = np.abs(cpu - gpu_np)
    assert err.max() <= 1.0 / 255.0 + 1e-3


def test_cpu_vs_cuda_fused_close():
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    from cudatracker.preprocessing.cuda import CudaPreprocessor

    rng = np.random.RandomState(1)
    img = rng.randint(0, 255, size=(247, 333, 3), dtype=np.uint8)
    cpu, meta = cpu_preprocess(img, 640, True, 114)
    pre = CudaPreprocessor(640, 114, fused=True, letterbox=True)
    gpu, meta2, _, _ = pre(img)
    gpu_np = gpu.detach().float().cpu().numpy()
    assert meta.pad_x == meta2.pad_x and meta.pad_y == meta2.pad_y
    err = np.abs(cpu - gpu_np)
    assert err.max() <= 1.0 / 255.0 + 1e-3


def test_nvrtc_lookup_checks_toolkit_lib_dirs(tmp_path, monkeypatch):
    import sys

    from cudatracker.preprocessing.kernel_source import find_nvrtc_dll

    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.setitem(sys.modules, "torch", None)
    monkeypatch.setitem(sys.modules, "nvidia", None)
    linux = tmp_path / "linux" / "lib64" / "libnvrtc.so.12"
    linux.parent.mkdir(parents=True)
    linux.touch()
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "linux"))
    assert find_nvrtc_dll() == str(linux)
    windows = tmp_path / "win" / "bin" / "x64" / "nvrtc64_130_0.dll"
    windows.parent.mkdir(parents=True)
    windows.touch()
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "win"))
    assert find_nvrtc_dll() == str(windows)


def test_letterbox_geometry_odd():
    meta = letterbox_geometry(333, 211, 640, True)
    assert meta.new_w + meta.pad_x <= 640
    assert meta.new_h + meta.pad_y <= 640
