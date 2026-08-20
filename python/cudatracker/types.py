from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np


@dataclass
class LetterboxMeta:
    scale: float
    pad_x: int
    pad_y: int
    new_w: int
    new_h: int
    src_w: int
    src_h: int
    imgsz: int
    scale_x: float = 0.0
    scale_y: float = 0.0

    def __post_init__(self) -> None:
        if self.scale_x == 0.0:
            self.scale_x = self.scale
        if self.scale_y == 0.0:
            self.scale_y = self.scale


@dataclass
class Detection:
    x: float
    y: float
    w: float
    h: float
    confidence: float
    class_id: int
    class_name: str = ""
    track_id: Optional[int] = None

    def as_xyxy(self) -> tuple[float, float, float, float]:
        x1 = self.x - self.w / 2.0
        y1 = self.y - self.h / 2.0
        return x1, y1, x1 + self.w, y1 + self.h


@dataclass
class Track:
    track_id: int
    class_id: int
    class_name: str
    x: float
    y: float
    w: float
    h: float
    confidence: float
    age: int
    hits: int
    history: list[tuple[float, float]] = field(default_factory=list)


@dataclass
class StageTimes:
    """Per-frame clocks. *_ms is CPU wall (includes waiting on the GPU).

    gpu_* is CUDA-event time. sync_* is time blocked in synchronize.
    end_to_end_ms is the number that matters: this frame, capture through last stage.
    """

    capture_ms: float = 0.0
    preprocess_ms: float = 0.0
    transfer_ms: float = 0.0
    inference_ms: float = 0.0
    postprocess_ms: float = 0.0
    tracking_ms: float = 0.0
    render_ms: float = 0.0
    end_to_end_ms: float = 0.0
    frame_index: int = 0
    capture_gpu_ms: float = 0.0
    capture_sync_ms: float = 0.0
    preprocess_gpu_ms: float = 0.0
    preprocess_sync_ms: float = 0.0
    transfer_gpu_ms: float = 0.0
    transfer_sync_ms: float = 0.0
    inference_gpu_ms: float = 0.0
    inference_sync_ms: float = 0.0
    postprocess_gpu_ms: float = 0.0
    postprocess_sync_ms: float = 0.0
    tracking_gpu_ms: float = 0.0
    tracking_sync_ms: float = 0.0
    render_gpu_ms: float = 0.0
    render_sync_ms: float = 0.0
    gpu_ms: float = 0.0
    cpu_ms: float = 0.0
    sync_ms: float = 0.0

    STAGE_ORDER: ClassVar[tuple[tuple[str, str], ...]] = (
        ("capture", "Capture"),
        ("preprocess", "Preprocess"),
        ("transfer", "H->D transfer"),
        ("inference", "Inference"),
        ("postprocess", "NMS"),
        ("tracking", "Tracking"),
        ("render", "Visualization"),
    )

    def as_dict(self) -> dict[str, float]:
        return {
            "capture_ms": self.capture_ms,
            "preprocess_ms": self.preprocess_ms,
            "transfer_ms": self.transfer_ms,
            "inference_ms": self.inference_ms,
            "postprocess_ms": self.postprocess_ms,
            "tracking_ms": self.tracking_ms,
            "render_ms": self.render_ms,
            "end_to_end_ms": self.end_to_end_ms,
            "preprocess_gpu_ms": self.preprocess_gpu_ms,
            "transfer_gpu_ms": self.transfer_gpu_ms,
            "inference_gpu_ms": self.inference_gpu_ms,
            "gpu_ms": self.gpu_ms,
            "cpu_ms": self.cpu_ms,
            "sync_ms": self.sync_ms,
        }

    def _wall(self, key: str) -> float:
        return float(getattr(self, f"{key}_ms"))

    def _gpu(self, key: str) -> float:
        return float(getattr(self, f"{key}_gpu_ms"))

    def _sync(self, key: str) -> float:
        return float(getattr(self, f"{key}_sync_ms"))

    def slowest(self) -> tuple[str, float]:
        items = [(name, self._wall(name)) for name, _ in self.STAGE_ORDER]
        return max(items, key=lambda kv: kv[1])

    def finish_totals(self) -> None:
        self.gpu_ms = sum(self._gpu(k) for k, _ in self.STAGE_ORDER)
        self.sync_ms = sum(self._sync(k) for k, _ in self.STAGE_ORDER)
        self.cpu_ms = max(0.0, self.end_to_end_ms - self.sync_ms)

    def table_lines(self) -> list[str]:
        lines = [f"Frame {self.frame_index}"]
        for key, label in self.STAGE_ORDER:
            wall = self._wall(key)
            gpu = self._gpu(key)
            extra = f"   gpu {gpu:5.1f}" if gpu > 0.05 else ""
            lines.append(f"{label:<18} {wall:6.1f} ms{extra}")
        lines.append("----------------------------")
        lines.append(f"{'TOTAL':<18} {self.end_to_end_ms:6.1f} ms")
        slow, _ = self.slowest()
        lines.append(
            f"cpu {self.cpu_ms:5.1f}  gpu {self.gpu_ms:5.1f}  wait {self.sync_ms:5.1f}  bottleneck {slow}"
        )
        return lines



@dataclass
class FramePacket:
    bgr: np.ndarray
    timestamp_ns: int
    frame_index: int
    source_w: int
    source_h: int
