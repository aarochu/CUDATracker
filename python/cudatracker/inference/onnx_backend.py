from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from cudatracker.inference.base import BackendNotAvailable
from cudatracker.inference.dll_path import prepend_torch_cuda_dlls
from cudatracker.logutil import log


class OnnxBackend:
    name = "onnx"

    def __init__(self, onnx_path: str, device: str, precision: str) -> None:
        if precision == "fp16":
            log("WARN", "infer", "ONNX Runtime backend ignores fp16 weights; the graph is FP32 unless you exported FP16 ONNX")
        path = Path(onnx_path)
        if not path.exists():
            raise BackendNotAvailable(
                f"ONNX file '{onnx_path}' is missing. Run python scripts/export_onnx.py"
            )
        self._use_cv = False
        self._session = None
        self._input = None
        self._output = None
        self.device = device
        self._cuda_ep = False
        prepend_torch_cuda_dlls()
        try:
            import onnxruntime as ort

            providers = []
            if device == "gpu":
                if "CUDAExecutionProvider" in ort.get_available_providers():
                    providers.append("CUDAExecutionProvider")
                else:
                    log("WARN", "infer", "CUDAExecutionProvider not in this ORT build; using CPU")
            providers.append("CPUExecutionProvider")
            so = ort.SessionOptions()
            so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self._session = ort.InferenceSession(str(path), so, providers=providers)
            self._input = self._session.get_inputs()[0].name
            self._output = self._session.get_outputs()[0].name
            used = self._session.get_providers()
            self._cuda_ep = bool(used) and used[0] == "CUDAExecutionProvider"
            log("INFO", "infer", f"ONNX Runtime {path.name} providers={used}")
        except ImportError:
            import cv2

            self._use_cv = True
            self._net = cv2.dnn.readNetFromONNX(str(path))
            if device == "gpu":
                try:
                    self._net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
                    self._net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
                except Exception:
                    log("WARN", "infer", "OpenCV DNN CUDA not built; running ONNX on CPU")
            log("INFO", "infer", f"OpenCV DNN ONNX fallback {path.name}")

    def infer(self, tensor: Any) -> Any:
        if self._session is not None:
            import torch

            if isinstance(tensor, torch.Tensor) and tensor.is_cuda and self._cuda_ep:
                io = self._session.io_binding()
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
                self._session.run_with_iobinding(io)
                outs = io.copy_outputs_to_cpu()
                return outs[0]
            arr = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor, dtype=np.float32)
            return self._session.run([self._output], {self._input: arr})[0]
        arr = tensor.detach().float().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor, dtype=np.float32)
        self._net.setInput(arr)
        return self._net.forward()

    def close(self) -> None:
        self._session = None
