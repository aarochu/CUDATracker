from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from cudatracker.logutil import log
from cudatracker.types import FramePacket


class Capture:
    def __init__(
        self,
        source: str,
        width: int,
        height: int,
        fps_cap: int = 0,
        gstreamer: bool = False,
        loop: bool = False,
    ) -> None:
        self.source = str(source)
        self.width = width
        self.height = height
        self.fps_cap = fps_cap
        self.gstreamer = gstreamer
        self.loop = loop
        self._cap: cv2.VideoCapture | None = None
        self._index = 0
        self._min_dt_ns = int(1e9 / fps_cap) if fps_cap and fps_cap > 0 else 0
        self._last_read_ns = 0

    def open(self) -> None:
        src = self.source
        path = Path(src)
        if self.gstreamer or "nvarguscamerasrc" in src or " ! " in src or src.strip().endswith("!"):
            cap = cv2.VideoCapture(src, cv2.CAP_GSTREAMER)
        elif src.isdigit():
            cap = cv2.VideoCapture(int(src))
        else:
            if not path.exists() and not src.startswith("/dev/"):
                raise FileNotFoundError(
                    f"Cannot open source '{src}'. Put a video at that path, pass --source 0 for a webcam, "
                    "or run python scripts/fetch_sample.py"
                )
            cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(
                f"OpenCV could not open '{src}'. For CSI on Jetson, pass a GStreamer string and set input.gstreamer: true."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap = cap
        native_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        native_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 0
        log("INFO", "capture", f"opened {src} native={native_w}x{native_h} fps={native_fps:.2f} target={self.width}x{self.height}")

    def read(self) -> FramePacket | None:
        if self._cap is None:
            raise RuntimeError("Capture.read() called before open().")
        if self._min_dt_ns:
            now = time.perf_counter_ns()
            wait = self._min_dt_ns - (now - self._last_read_ns)
            if wait > 0:
                time.sleep(wait / 1e9)
        ok, frame = self._cap.read()
        ts = time.perf_counter_ns()
        self._last_read_ns = ts
        if not ok or frame is None:
            if self.loop:
                self._cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, frame = self._cap.read()
                ts = time.perf_counter_ns()
                self._last_read_ns = ts
            if not ok or frame is None:
                return None
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        packet = FramePacket(
            bgr=np.ascontiguousarray(frame),
            timestamp_ns=int(ts),
            frame_index=self._index,
            source_w=self.width,
            source_h=self.height,
        )
        self._index += 1
        return packet

    def close(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
