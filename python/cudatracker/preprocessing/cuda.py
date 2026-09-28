from __future__ import annotations

import time

import numpy as np
import torch

from cudatracker.logutil import log
from cudatracker.preprocessing.geometry import letterbox_geometry
from cudatracker.preprocessing.nvrtc_runtime import (
    CudaKernels,
    _c_f,
    _c_i,
    _c_ptr,
    _c_u8,
    grid_1d,
    grid_2d,
)


class CudaPreprocessor:
    def __init__(self, imgsz: int, pad_value: int = 114, fused: bool = False, letterbox: bool = True) -> None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA preprocessor requested but torch.cuda.is_available() is False.")
        self.imgsz = imgsz
        self.pad_value = pad_value
        self.fused = fused
        self.letterbox = letterbox
        self.kernels = CudaKernels()
        self.device = torch.device("cuda")
        n = imgsz * imgsz
        self.src_cap = 1920 * 1080 * 3
        self.d_src = torch.empty(self.src_cap, dtype=torch.uint8, device=self.device)
        self.d_canvas = torch.empty(n * 3, dtype=torch.uint8, device=self.device)
        self.d_rgb = torch.empty(n * 3, dtype=torch.uint8, device=self.device)
        self.d_hwc = torch.empty(n * 3, dtype=torch.float32, device=self.device)
        self.d_chw = torch.empty((1, 3, imgsz, imgsz), dtype=torch.float32, device=self.device)
        self._pinned: torch.Tensor | None = None
        self._pinned_n = 0
        log("INFO", "preprocess", f"CUDA preprocessor imgsz={imgsz} fused={fused}")

    def _stream(self) -> int:
        return int(torch.cuda.current_stream().cuda_stream)

    def _prepare_host(self, bgr: np.ndarray) -> int:
        h, w = bgr.shape[:2]
        n = h * w * 3
        if n > self.src_cap:
            self.src_cap = n
            self.d_src = torch.empty(self.src_cap, dtype=torch.uint8, device=self.device)
        if self._pinned is None or self._pinned_n < n:
            self._pinned = torch.empty(n, dtype=torch.uint8, pin_memory=True)
            self._pinned_n = n
        np.copyto(self._pinned[:n].numpy(), np.ascontiguousarray(bgr).reshape(-1))
        return n

    def __call__(self, bgr: np.ndarray):
        from cudatracker.telemetry.span import SpanResult

        src_h, src_w = bgr.shape[:2]
        meta = letterbox_geometry(src_w, src_h, self.imgsz, letterbox=self.letterbox)
        n = self._prepare_host(bgr)
        stream = self._stream()
        ev_a = torch.cuda.Event(enable_timing=True)
        ev_b = torch.cuda.Event(enable_timing=True)
        ev_c = torch.cuda.Event(enable_timing=True)

        t_x0 = time.perf_counter()
        ev_a.record()
        self.d_src[:n].copy_(self._pinned[:n], non_blocking=True)
        ev_b.record()
        t_wait0 = time.perf_counter()
        ev_b.synchronize()
        transfer = SpanResult(
            wall_ms=(time.perf_counter() - t_x0) * 1000.0,
            gpu_ms=float(ev_a.elapsed_time(ev_b)),
            sync_ms=(time.perf_counter() - t_wait0) * 1000.0,
        )

        k = self.kernels
        t_k0 = time.perf_counter()
        ev_b.record()
        if self.fused:
            g, b = grid_2d(self.imgsz, self.imgsz)
            k.launch(
                k.fn_fused,
                g,
                b,
                [
                    _c_ptr(self.d_src),
                    _c_i(src_w),
                    _c_i(src_h),
                    _c_i(src_w * 3),
                    _c_ptr(self.d_chw),
                    _c_i(self.imgsz),
                    _c_i(self.imgsz),
                    _c_i(meta.new_w),
                    _c_i(meta.new_h),
                    _c_i(meta.pad_x),
                    _c_i(meta.pad_y),
                    _c_u8(self.pad_value),
                    _c_f(1.0 / 255.0),
                ],
                stream,
            )
        else:
            g, b = grid_2d(self.imgsz, self.imgsz)
            k.launch(
                k.fn_resize,
                g,
                b,
                [
                    _c_ptr(self.d_src),
                    _c_i(src_w),
                    _c_i(src_h),
                    _c_i(src_w * 3),
                    _c_ptr(self.d_canvas),
                    _c_i(self.imgsz),
                    _c_i(self.imgsz),
                    _c_i(meta.new_w),
                    _c_i(meta.new_h),
                    _c_i(meta.pad_x),
                    _c_i(meta.pad_y),
                    _c_u8(self.pad_value),
                ],
                stream,
            )
            n_pix = self.imgsz * self.imgsz
            g1, b1 = grid_1d(n_pix)
            k.launch(
                k.fn_bgr,
                g1,
                b1,
                [_c_ptr(self.d_canvas), _c_ptr(self.d_rgb), _c_i(n_pix)],
                stream,
            )
            g1, b1 = grid_1d(n_pix * 3)
            k.launch(
                k.fn_norm,
                g1,
                b1,
                [_c_ptr(self.d_rgb), _c_ptr(self.d_hwc), _c_i(n_pix * 3), _c_f(1.0 / 255.0)],
                stream,
            )
            g, b = grid_2d(self.imgsz, self.imgsz)
            k.launch(
                k.fn_layout,
                g,
                b,
                [_c_ptr(self.d_hwc), _c_ptr(self.d_chw), _c_i(self.imgsz), _c_i(self.imgsz)],
                stream,
            )
        ev_c.record()
        t_wait1 = time.perf_counter()
        ev_c.synchronize()
        kernels = SpanResult(
            wall_ms=(time.perf_counter() - t_k0) * 1000.0,
            gpu_ms=float(ev_b.elapsed_time(ev_c)),
            sync_ms=(time.perf_counter() - t_wait1) * 1000.0,
        )
        return self.d_chw, meta, transfer, kernels
