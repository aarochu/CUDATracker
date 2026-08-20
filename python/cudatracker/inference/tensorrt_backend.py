from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cudatracker.inference.base import BackendNotAvailable
from cudatracker.inference.dll_path import prepend_torch_cuda_dlls
from cudatracker.logutil import log


class TensorRTBackend:
    """Native TensorRT if the Python package and an engine file exist.

    Otherwise uses ONNX Runtime's TensorRT execution provider, which still
    builds/runs a TensorRT engine. That fallback is logged so the matrix stays honest.
    """

    name = "tensorrt"

    def __init__(self, engine_path: str, onnx_path: str, precision: str, device: str) -> None:
        if device != "gpu":
            raise BackendNotAvailable("TensorRT backend requires inference.device: gpu")
        prepend_torch_cuda_dlls()
        self.precision = precision
        self._native = None
        self._ort = None
        engine = Path(engine_path)
        try:
            import tensorrt as trt  # noqa: F401

            if engine.exists():
                self._native = _NativeEngine(str(engine))
                log("INFO", "infer", f"TensorRT native engine {engine.name} {precision}")
                return
            log("WARN", "infer", f"{engine} missing; trying ORT TensorRT EP on the ONNX file")
        except ImportError:
            log("WARN", "infer", "Python package 'tensorrt' not importable; using ORT TensorRT EP if present")
        onnx = Path(onnx_path)
        if not onnx.exists():
            raise BackendNotAvailable(
                f"Need {engine} or {onnx}. Export ONNX, then python scripts/build_engine.py"
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise BackendNotAvailable("onnxruntime is required for the TensorRT EP fallback.") from exc
        if "TensorrtExecutionProvider" not in ort.get_available_providers():
            raise BackendNotAvailable(
                "This ONNX Runtime build has no TensorrtExecutionProvider. Install TensorRT or a TRT-enabled ORT wheel."
            )
        cache = Path("models") / "ort_trt_cache"
        cache.mkdir(parents=True, exist_ok=True)
        providers = [
            (
                "TensorrtExecutionProvider",
                {
                    "device_id": 0,
                    "trt_fp16_enable": precision == "fp16",
                    "trt_max_workspace_size": 1 << 30,
                    "trt_engine_cache_enable": True,
                    "trt_engine_cache_path": str(cache),
                },
            ),
            "CUDAExecutionProvider",
            "CPUExecutionProvider",
        ]
        so = ort.SessionOptions()
        self._ort = ort.InferenceSession(str(onnx), so, providers=providers)
        self._input = self._ort.get_inputs()[0].name
        self._output = self._ort.get_outputs()[0].name
        used = self._ort.get_providers()
        log("INFO", "infer", f"TensorRT via ORT EP {onnx.name} {precision} providers={used}")
        if used[0] != "TensorrtExecutionProvider":
            raise BackendNotAvailable(
                f"ORT did not select TensorRT (got {used}). Check CUDA/TensorRT library versions."
            )

    def infer(self, tensor: Any) -> Any:
        if self._native is not None:
            return self._native.infer(tensor)
        import torch

        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            io = self._ort.io_binding()
            device_id = 0 if tensor.device.index is None else int(tensor.device.index)
            io.bind_input(
                name=self._input,
                device_type="cuda",
                device_id=device_id,
                element_type=np.float32,
                shape=tuple(tensor.shape),
                buffer_ptr=int(tensor.data_ptr()),
            )
            io.bind_output(self._output, "cuda")
            self._ort.run_with_iobinding(io)
            return io.copy_outputs_to_cpu()[0]
        arr = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor, dtype=np.float32)
        return self._ort.run([self._output], {self._input: arr})[0]

    def close(self) -> None:
        self._ort = None
        self._native = None


class _NativeEngine:
    def __init__(self, engine_path: str) -> None:
        import tensorrt as trt

        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        blob = Path(engine_path).read_bytes()
        self.engine = self.runtime.deserialize_cuda_engine(blob)
        if self.engine is None:
            raise BackendNotAvailable(
                f"Failed to deserialize {engine_path}. Rebuild the engine with this TensorRT version."
            )
        self.context = self.engine.create_execution_context()
        import torch

        self.stream = int(torch.cuda.current_stream().cuda_stream)
        self.torch = torch
        n = self.engine.num_io_tensors
        self.names = [self.engine.get_tensor_name(i) for i in range(n)]
        self.inputs = [name for name in self.names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
        self.outputs = [name for name in self.names if self.engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
        self.out_bufs = {}
        for name in self.outputs:
            shape = tuple(int(x) for x in self.engine.get_tensor_shape(name))
            if any(d < 0 for d in shape):
                continue
            dtype = trt.nptype(self.engine.get_tensor_dtype(name))
            torch_dtype = torch.float16 if np.dtype(dtype) == np.float16 else torch.float32
            self.out_bufs[name] = torch.empty(shape, device="cuda", dtype=torch_dtype)

    def infer(self, tensor) -> Any:
        import tensorrt as trt
        import torch

        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(np.asarray(tensor))
        if not tensor.is_cuda:
            tensor = tensor.to("cuda", non_blocking=True)
        inp = self.inputs[0]
        in_dtype = trt.nptype(self.engine.get_tensor_dtype(inp))
        want = torch.float16 if np.dtype(in_dtype) == np.float16 else torch.float32
        if tensor.dtype != want:
            tensor = tensor.to(want)
        if hasattr(self.context, "set_input_shape"):
            self.context.set_input_shape(inp, tuple(int(x) for x in tensor.shape))
        self.context.set_tensor_address(inp, int(tensor.data_ptr()))
        for name in self.outputs:
            if name not in self.out_bufs:
                shape = tuple(int(x) for x in self.context.get_tensor_shape(name))
                dtype = trt.nptype(self.engine.get_tensor_dtype(name))
                torch_dtype = torch.float16 if np.dtype(dtype) == np.float16 else torch.float32
                self.out_bufs[name] = torch.empty(shape, device="cuda", dtype=torch_dtype)
            self.context.set_tensor_address(name, int(self.out_bufs[name].data_ptr()))
        ok = self.context.execute_async_v3(self.stream)
        if not ok:
            raise RuntimeError("TensorRT execute_async_v3 failed")
        torch.cuda.synchronize()
        return self.out_bufs[self.outputs[0]]
