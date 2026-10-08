from types import SimpleNamespace

import numpy as np

from cudatracker.config import AppConfig
from cudatracker.pipeline import Pipeline


def test_cpu_onnx_preprocess_needs_no_torch(monkeypatch):
    import sys

    cfg = AppConfig()
    cfg.inference.backend = "onnx"
    pipe = object.__new__(Pipeline)
    pipe.cfg = cfg
    monkeypatch.setitem(sys.modules, "torch", None)
    for cuda_ep in (False, True):
        pipe.backend = SimpleNamespace(_cuda_ep=cuda_ep)
        tensor, meta, pre, transfer = pipe._tensor_cpu_path(np.zeros((24, 32, 3), dtype=np.uint8))
        assert tensor.shape == (1, 3, 640, 640)
        assert meta.src_w == 32
        assert pre.wall_ms >= 0
        assert transfer.wall_ms == 0


def test_onnx_session_accepts_numpy_without_torch(monkeypatch):
    import sys
    from cudatracker.inference.onnx_backend import OnnxBackend

    backend = object.__new__(OnnxBackend)
    backend._session = SimpleNamespace(run=lambda outputs, inputs: [inputs["input"]])
    backend._input = "input"
    backend._output = "output"
    backend._cuda_ep = False
    monkeypatch.setitem(sys.modules, "torch", None)
    arr = np.zeros((1, 3, 4, 4), dtype=np.float32)
    assert backend.infer(arr) is arr


def test_inference_timing_uses_active_backend_device():
    pipe = object.__new__(Pipeline)
    pipe.cfg = AppConfig()
    cpu = SimpleNamespace(is_cuda=False)
    gpu = SimpleNamespace(is_cuda=True)
    pipe.cfg.inference.backend = "onnx"
    pipe.backend = SimpleNamespace(_cuda_ep=False)
    assert not pipe._inference_uses_cuda(cpu)
    assert not pipe._inference_uses_cuda(gpu)
    pipe.backend._cuda_ep = True
    assert pipe._inference_uses_cuda(gpu)
    pipe.cfg.inference.backend = "pytorch"
    pipe.backend = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    assert not pipe._inference_uses_cuda(cpu)
    pipe.backend.device.type = "cuda"
    assert pipe._inference_uses_cuda(gpu)
    pipe.cfg.inference.backend = "tensorrt"
    assert pipe._inference_uses_cuda(cpu)


def test_headless_live_file_stops_at_eof(monkeypatch):
    from cudatracker import cli

    seen = []

    class EmptyPipeline:
        def __init__(self, cfg, loop_source):
            seen.append(loop_source)

        def open(self):
            pass

        def step(self):
            return None, None

        def close(self):
            pass

    monkeypatch.setattr(cli, "Pipeline", EmptyPipeline)
    cfg = AppConfig()
    cfg.visualization.enabled = False
    assert cli.run_live(cfg) == 0
    assert seen == [False]


def test_matrix_cell_skips_empty_measurement(monkeypatch, tmp_path):
    from benchmarks import benchmark

    instances = []

    class EmptyPipeline:
        def __init__(self, cfg, loop_source):
            self._e2e = []
            self.closed = False
            instances.append(self)

        def open(self):
            pass

        def step(self):
            return None, None

        def close(self):
            self.closed = True

    monkeypatch.setattr(benchmark, "Pipeline", EmptyPipeline)
    monkeypatch.setattr(benchmark, "write_run_artifacts", lambda *args: None)
    cfg = AppConfig()
    cfg.benchmark.results_dir = str(tmp_path)
    row = benchmark.run_cell(cfg, "cpu", "onnx", "fp32", "640x480", "yolov8n")
    assert row["skipped"] is True
    assert "measured frames" in row["skip_reason"]
    assert instances[0].closed


def test_matrix_cell_skips_missing_backend_dependency(monkeypatch):
    from benchmarks import benchmark

    def missing_dependency(*args, **kwargs):
        raise ModuleNotFoundError("No module named 'torch'")

    monkeypatch.setattr(benchmark, "Pipeline", missing_dependency)
    row = benchmark.run_cell(AppConfig(), "cpu", "pytorch", "fp32", "640x480", "yolov8n")
    assert row["skipped"] is True
    assert "torch" in row["skip_reason"]
