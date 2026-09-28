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


def test_resolution_axis_is_shared_across_series():
    plot = _plot()
    series = {
        "cpu/onnx": [("1920x1080", 30.0), ("640x480", 60.0)],
        "cuda/tensorrt": [("1280x720", 40.0)],
    }
    assert plot.resolution_order(series) == ["640x480", "1280x720", "1920x1080"]
