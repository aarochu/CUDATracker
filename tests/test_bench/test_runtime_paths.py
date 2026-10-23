from types import SimpleNamespace

import numpy as np
import pytest

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


@pytest.mark.parametrize(
    "failure",
    [ModuleNotFoundError("No module named 'torch'"), OSError("CUDA library blocked by the OS")],
)
def test_matrix_cell_skips_missing_backend_dependency(monkeypatch, failure):
    from benchmarks import benchmark

    def missing_dependency(*args, **kwargs):
        raise failure

    monkeypatch.setattr(benchmark, "Pipeline", missing_dependency)
    row = benchmark.run_cell(AppConfig(), "cpu", "pytorch", "fp32", "640x480", "yolov8n")
    assert row["skipped"] is True
    assert str(failure) in row["skip_reason"]


def test_bench_reports_blocked_runtime_without_traceback(monkeypatch):
    from cudatracker import cli

    def blocked(*args, **kwargs):
        raise OSError("CUDA library blocked by the OS")

    monkeypatch.setattr(cli, "Pipeline", blocked)
    assert cli.run_bench(AppConfig()) == 2


def test_native_tensorrt_resizes_output_and_uses_current_stream(monkeypatch):
    import sys

    from cudatracker.inference.tensorrt_backend import _NativeEngine

    class Tensor:
        def __init__(self, shape):
            self.shape = shape
            self.dtype = np.float32
            self.is_cuda = True

        def data_ptr(self):
            return id(self)

    stream = SimpleNamespace(cuda_stream=1)
    cuda = SimpleNamespace(current_stream=lambda: stream, synchronize=lambda: None)
    fake_torch = SimpleNamespace(Tensor=Tensor, float32=np.float32, float16=np.float16, cuda=cuda, empty=lambda shape, **kwargs: Tensor(shape))
    fake_trt = SimpleNamespace(nptype=lambda dtype: dtype)
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "tensorrt", fake_trt)

    class Context:
        output_shape = (1, 84, 8400)

        def set_input_shape(self, name, shape):
            pass

        def set_tensor_address(self, name, address):
            pass

        def get_tensor_shape(self, name):
            return self.output_shape

        def execute_async_v3(self, cuda_stream):
            self.last_stream = cuda_stream
            return True

    engine = object.__new__(_NativeEngine)
    engine.context = Context()
    engine.engine = SimpleNamespace(get_tensor_dtype=lambda name: np.float32)
    engine.inputs = ["images"]
    engine.outputs = ["output"]
    engine.out_bufs = {}
    first = engine.infer(Tensor((1, 3, 640, 640)))
    assert first.shape == (1, 84, 8400)
    assert engine.context.last_stream == 1

    engine.context.output_shape = (1, 84, 2100)
    stream.cuda_stream = 2
    second = engine.infer(Tensor((1, 3, 320, 320)))
    assert second.shape == (1, 84, 2100)
    assert second is not first
    assert engine.context.last_stream == 2
