from __future__ import annotations

from cudatracker.config import AppConfig
from cudatracker.inference.base import BackendNotAvailable


def create_backend(cfg: AppConfig):
    name = cfg.inference.backend.lower()
    if name not in ("pytorch", "onnx", "tensorrt"):
        raise BackendNotAvailable(
            f"Unknown inference backend '{cfg.inference.backend}'. Use pytorch, onnx, or tensorrt."
        )
    if name == "pytorch":
        from cudatracker.inference.pytorch_backend import PyTorchBackend

        return PyTorchBackend(cfg.model.weights, cfg.inference.device, cfg.inference.precision, cfg.model.imgsz)
    if name == "onnx":
        from cudatracker.inference.onnx_backend import OnnxBackend

        return OnnxBackend(cfg.model.onnx, cfg.inference.device, cfg.inference.precision)
    from cudatracker.inference.tensorrt_backend import TensorRTBackend

    engine = cfg.model.engine
    if cfg.inference.precision == "fp32" and cfg.model.engine_fp32:
        engine = cfg.model.engine_fp32
    if cfg.inference.precision == "fp16" and cfg.model.engine_fp16:
        engine = cfg.model.engine_fp16
    return TensorRTBackend(engine, cfg.model.onnx, cfg.inference.precision, cfg.inference.device)
