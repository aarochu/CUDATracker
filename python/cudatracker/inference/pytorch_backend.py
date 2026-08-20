from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from cudatracker.inference.base import BackendNotAvailable
from cudatracker.logutil import log


class PyTorchBackend:
    name = "pytorch"

    def __init__(self, weights: str, device: str, precision: str, imgsz: int) -> None:
        if precision not in ("fp32", "fp16"):
            raise BackendNotAvailable(f"PyTorch precision '{precision}' is not supported.")
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise BackendNotAvailable("ultralytics is required for the PyTorch backend.") from exc
        path = Path(weights)
        load_arg = str(path) if path.exists() else path.name  # 'yolov8n.pt' downloads
        yolo = YOLO(load_arg)
        if path.parent.as_posix() not in (".", "") and not path.exists():
            # YOLO downloaded to CWD; copy hint
            log("WARN", "infer", f"{weights} missing; ultralytics will fetch {path.name} into its default cache")
        self.model = yolo.model.eval()
        self.use_fp16 = precision == "fp16"
        self.device = torch.device("cuda" if device == "gpu" and torch.cuda.is_available() else "cpu")
        if self.device.type == "cpu" and device == "gpu":
            log("WARN", "infer", "GPU requested but CUDA is not available; PyTorch is on CPU")
        self.model.to(self.device)
        if self.use_fp16:
            if self.device.type != "cuda":
                raise BackendNotAvailable("PyTorch FP16 needs a CUDA device.")
            self.model.half()
        self.imgsz = imgsz
        log("INFO", "infer", f"PyTorch {path.name} on {self.device} {precision}")

    def infer(self, tensor: Any) -> Any:
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.from_numpy(tensor)
        tensor = tensor.to(self.device, non_blocking=True)
        if self.use_fp16:
            tensor = tensor.half()
        else:
            tensor = tensor.float()
        with torch.inference_mode():
            out = self.model(tensor)
        if isinstance(out, (list, tuple)):
            out = out[0]
        return out

    def close(self) -> None:
        self.model = None
