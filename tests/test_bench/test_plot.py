from __future__ import annotations

import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _plot():
    spec = importlib.util.spec_from_file_location("ct_plot", ROOT / "benchmarks" / "plot.py")
    mod = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def test_paired_speedup_cpu_cuda():
    plot = _plot()
    live = [
        {"preprocess": "cpu", "backend": "onnx", "precision": "fp32", "resolution": "640x480", "model": "yolov8n", "fps": "10"},
        {"preprocess": "cuda", "backend": "onnx", "precision": "fp32", "resolution": "640x480", "model": "yolov8n", "fps": "20"},
    ]
    pairs = plot.paired_speedup(live, "preprocess", "cpu", "cuda", ("backend", "precision", "resolution", "model"))
    assert len(pairs) == 1
    assert abs(pairs[0][1] - 2.0) < 1e-9
