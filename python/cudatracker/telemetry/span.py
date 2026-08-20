from __future__ import annotations

import time
from dataclasses import dataclass


def _cuda_events():
    try:
        import torch

        if torch.cuda.is_available():
            return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True), torch
    except Exception:
        pass
    return None, None, None


@dataclass
class SpanResult:
    wall_ms: float = 0.0
    gpu_ms: float = 0.0
    sync_ms: float = 0.0

    @property
    def cpu_ms(self) -> float:
        return max(0.0, self.wall_ms - self.sync_ms)


class Span:
    """Time a stage. Wall is always host clock.

    If cuda=True, CUDA events cover device work; sync_ms is time blocked in
    event.synchronize(). Naive timers dump that wait into the next stage.
    Sequential sync at stage boundaries is deliberate: it attributes work so
    you can name a bottleneck. It is not an overlapped pipeline.
    """

    def __init__(self, cuda: bool = False) -> None:
        self.cuda = cuda
        self._t0 = 0.0
        self._ev0 = self._ev1 = None
        self._torch = None
        self.result = SpanResult()

    def __enter__(self) -> Span:
        if self.cuda:
            self._ev0, self._ev1, self._torch = _cuda_events()
            if self._ev0 is None:
                self.cuda = False
        self._t0 = time.perf_counter()
        if self.cuda:
            self._ev0.record()
        return self

    def __exit__(self, *exc) -> None:
        t_launch = time.perf_counter()
        gpu_ms = 0.0
        sync_ms = 0.0
        if self.cuda and self._ev1 is not None:
            self._ev1.record()
            t_s0 = time.perf_counter()
            self._ev1.synchronize()
            sync_ms = (time.perf_counter() - t_s0) * 1000.0
            gpu_ms = float(self._ev0.elapsed_time(self._ev1))
        wall_ms = (time.perf_counter() - self._t0) * 1000.0
        # If CUDA events failed silently, still report wall.
        _ = t_launch
        self.result = SpanResult(wall_ms=wall_ms, gpu_ms=gpu_ms, sync_ms=sync_ms)
