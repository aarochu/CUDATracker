from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from cudatracker.analytics.stats import Analytics
from cudatracker.capture.camera import Capture
from cudatracker.config import AppConfig
from cudatracker.detection.postprocess import decode_yolo
from cudatracker.inference.factory import create_backend
from cudatracker.logutil import log
from cudatracker.preprocessing.cpu import cpu_preprocess
from cudatracker.preprocessing.geometry import percentiles
from cudatracker.telemetry.hardware import HardwareSampler, environment_metadata, json_sanitize
from cudatracker.telemetry.span import Span
from cudatracker.tracking.sort import SortTracker
from cudatracker.types import StageTimes
from cudatracker.visualization.renderer import draw


class Pipeline:
    def __init__(self, cfg: AppConfig, loop_source: bool = False) -> None:
        self.cfg = cfg
        self.capture = Capture(
            cfg.input.source,
            cfg.input.width,
            cfg.input.height,
            cfg.input.fps_cap,
            cfg.input.gstreamer,
            loop=loop_source,
        )
        self.backend = create_backend(cfg)
        self.tracker = SortTracker(
            cfg.tracker.max_age, cfg.tracker.min_hits, cfg.tracker.iou_threshold
        )
        self.cuda_pre = None
        if cfg.preprocess.backend == "cuda":
            from cudatracker.preprocessing.cuda import CudaPreprocessor

            self.cuda_pre = CudaPreprocessor(
                cfg.model.imgsz,
                cfg.preprocess.pad_value,
                cfg.preprocess.fused,
                cfg.preprocess.letterbox,
            )
        elif cfg.preprocess.backend != "cpu":
            raise RuntimeError(f"Unknown preprocess backend '{cfg.preprocess.backend}'")
        self.hw = HardwareSampler() if cfg.telemetry.hardware else None
        self.analytics = Analytics()
        self.times = StageTimes()
        self.hud = cfg.visualization.hud
        self.trails = cfg.visualization.trails
        self.hw_panel = cfg.visualization.hardware_panel
        self.status = ""
        self._e2e: list[float] = []
        self._stage: dict[str, list[float]] = {k: [] for k in StageTimes().as_dict()}
        self._writer = None
        if cfg.output.video:
            Path(cfg.output.video).parent.mkdir(parents=True, exist_ok=True)

    def open(self) -> None:
        self.capture.open()
        if self.hw:
            self.hw.start()

    def close(self) -> None:
        self.capture.close()
        if self.hw:
            self.hw.stop()
        if hasattr(self.backend, "close"):
            self.backend.close()
        if self._writer is not None:
            self._writer.release()

    def _tensor_cpu_path(self, bgr):
        from cudatracker.telemetry.span import SpanResult

        with Span(cuda=False) as pre:
            np_tensor, meta = cpu_preprocess(
                bgr, self.cfg.model.imgsz, self.cfg.preprocess.letterbox, self.cfg.preprocess.pad_value
            )
        xfer = SpanResult()
        if self.cfg.inference.backend == "onnx":
            if not getattr(self.backend, "_cuda_ep", False):
                return np_tensor, meta, pre.result, xfer
            try:
                import torch
            except ImportError:
                return np_tensor, meta, pre.result, xfer
        else:
            import torch

        tensor = torch.from_numpy(np_tensor)
        if self.cfg.inference.device == "gpu" and torch.cuda.is_available():
            with Span(cuda=True) as x:
                tensor = tensor.pin_memory().to("cuda", non_blocking=True)
            xfer = x.result
        return tensor, meta, pre.result, xfer

    def step(self) -> tuple[np.ndarray | None, StageTimes]:
        t_all = time.perf_counter()
        with Span(cuda=False) as cap_span:
            packet = self.capture.read()
        if packet is None:
            return None, self.times
        bgr = packet.bgr
        if self.cuda_pre is not None:
            tensor, meta, xfer, pre = self.cuda_pre(bgr)
        else:
            tensor, meta, pre, xfer = self._tensor_cpu_path(bgr)
        use_cuda = False
        try:
            import torch

            use_cuda = torch.cuda.is_available()
        except ImportError:
            pass
        with Span(cuda=use_cuda) as inf_span:
            raw = self.backend.infer(tensor)
        with Span(cuda=False) as post_span:
            dets = decode_yolo(raw, meta, self.cfg.model.conf, self.cfg.model.iou)
        with Span(cuda=False) as track_span:
            tracks = self.tracker.update(dets)
        self.analytics.update(dets, tracks)
        vis = bgr
        times = StageTimes(
            capture_ms=cap_span.result.wall_ms,
            preprocess_ms=pre.wall_ms,
            transfer_ms=xfer.wall_ms,
            inference_ms=inf_span.result.wall_ms,
            postprocess_ms=post_span.result.wall_ms,
            tracking_ms=track_span.result.wall_ms,
            render_ms=0.0,
            end_to_end_ms=0.0,
            frame_index=packet.frame_index,
            capture_gpu_ms=cap_span.result.gpu_ms,
            capture_sync_ms=cap_span.result.sync_ms,
            preprocess_gpu_ms=pre.gpu_ms,
            preprocess_sync_ms=pre.sync_ms,
            transfer_gpu_ms=xfer.gpu_ms,
            transfer_sync_ms=xfer.sync_ms,
            inference_gpu_ms=inf_span.result.gpu_ms,
            inference_sync_ms=inf_span.result.sync_ms,
            postprocess_gpu_ms=post_span.result.gpu_ms,
            postprocess_sync_ms=post_span.result.sync_ms,
            tracking_gpu_ms=track_span.result.gpu_ms,
            tracking_sync_ms=track_span.result.sync_ms,
        )
        if self.cfg.visualization.enabled:
            with Span(cuda=False) as render_span:
                fps = 1000.0 / max(self.times.end_to_end_ms, 1e-3)
                vis = draw(
                    bgr.copy(),
                    tracks,
                    times,
                    fps,
                    self.cfg.preprocess.backend,
                    self.cfg.inference.backend,
                    self.cfg.inference.precision,
                    self.hud,
                    self.trails,
                    self.hw_panel,
                    self.hw.latest if self.hw else None,
                    self.status,
                    packet.frame_index,
                )
                if self.cfg.output.video:
                    if self._writer is None:
                        fourcc = cv2_fourcc()
                        self._writer = cv2_writer(self.cfg.output.video, fourcc, 30, (vis.shape[1], vis.shape[0]))
                    self._writer.write(vis)
            times.render_ms = render_span.result.wall_ms
            times.render_gpu_ms = render_span.result.gpu_ms
            times.render_sync_ms = render_span.result.sync_ms
        times.end_to_end_ms = (time.perf_counter() - t_all) * 1000.0
        times.finish_totals()
        self.times = times
        for k, v in self.times.as_dict().items():
            self._stage.setdefault(k, []).append(v)
        self._e2e.append(times.end_to_end_ms)
        return vis, self.times

    def begin_measure(self) -> None:
        """Drop warmup analytics so JSON/CSV scene stats match the timed window."""
        self.analytics.reset()

    def record_window(self, skip: int) -> dict[str, Any]:
        e2e = self._e2e[skip:]
        def stats(xs: list[float]) -> dict[str, float]:
            if not xs:
                return {"mean": float("nan"), "p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
            p50, p95, p99 = percentiles(xs, [50, 95, 99])
            return {"mean": float(np.mean(xs)), "p50": p50, "p95": p95, "p99": p99}

        wall = sum(e2e) / 1000.0
        fps = len(e2e) / wall if wall > 0 else 0.0
        stages = {k.replace("_ms", ""): stats(v[skip:]) for k, v in self._stage.items()}
        return {
            "fps": fps,
            "frames": len(e2e),
            "end_to_end": stats(e2e),
            "stages": stages,
            "analytics": self.analytics.summary(),
            "hardware": self.hw.latest if self.hw else {},
        }


def cv2_fourcc():
    import cv2

    return cv2.VideoWriter_fourcc(*"mp4v")


def cv2_writer(path, fourcc, fps, size):
    import cv2

    return cv2.VideoWriter(path, fourcc, fps, size)


def file_hash(path: str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_run_artifacts(cfg: AppConfig, summary: dict[str, Any], extra: dict[str, Any] | None = None) -> Path:
    payload = {
        "config": cfg.snapshot(),
        "environment": environment_metadata(),
        "input_hash": file_hash(cfg.input.source),
        "summary": summary,
        "extra": extra or {},
    }
    payload = json_sanitize(payload)
    out = cfg.output.metrics
    if not out:
        Path("benchmarks/results").mkdir(parents=True, exist_ok=True)
        out = str(Path("benchmarks/results") / "last_run.json")
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = path.with_suffix(".csv")
    row = flatten_row(cfg, summary)
    write_csv_row(csv_path, row)
    log("INFO", "bench", f"wrote {path} and {csv_path}")
    return path


def _bottleneck(stages: dict[str, Any]) -> str:
    names = ("capture", "preprocess", "transfer", "inference", "postprocess", "tracking", "render")
    best = "inference"
    best_v = -1.0
    for name in names:
        v = (stages.get(name) or {}).get("mean")
        if v is None:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if fv > best_v:
            best_v = fv
            best = name
    return best


def flatten_row(cfg: AppConfig, summary: dict[str, Any]) -> dict[str, Any]:
    e2e = summary.get("end_to_end") or {}
    stages = summary.get("stages") or {}
    hw = summary.get("hardware") or {}
    an = summary.get("analytics") or {}

    def st(name: str, key: str) -> Any:
        return (stages.get(name) or {}).get(key)

    return {
        "preprocess": cfg.preprocess.backend,
        "fused": cfg.preprocess.fused,
        "backend": cfg.inference.backend,
        "precision": cfg.inference.precision,
        "resolution": cfg.input.resolution,
        "model": cfg.model.name,
        "fps": summary.get("fps"),
        "e2e_mean_ms": e2e.get("mean"),
        "e2e_p50_ms": e2e.get("p50"),
        "e2e_p95_ms": e2e.get("p95"),
        "e2e_p99_ms": e2e.get("p99"),
        "capture_ms": st("capture", "mean"),
        "preprocess_ms": st("preprocess", "mean"),
        "transfer_ms": st("transfer", "mean"),
        "inference_ms": st("inference", "mean"),
        "postprocess_ms": st("postprocess", "mean"),
        "tracking_ms": st("tracking", "mean"),
        "render_ms": st("render", "mean"),
        "gpu_ms": st("gpu", "mean") if st("gpu", "mean") is not None else summary.get("gpu_ms"),
        "cpu_ms": st("cpu", "mean") if st("cpu", "mean") is not None else None,
        "sync_ms": st("sync", "mean"),
        "inference_gpu_ms": st("inference_gpu", "mean"),
        "preprocess_gpu_ms": st("preprocess_gpu", "mean"),
        "transfer_gpu_ms": st("transfer_gpu", "mean"),
        "bottleneck": _bottleneck(stages),
        "cpu_util": hw.get("cpu_util"),
        "gpu_util": hw.get("gpu_util"),
        "ram_bytes": hw.get("ram_bytes"),
        "gpu_mem_bytes": hw.get("gpu_mem_bytes"),
        "gpu_temp_c": hw.get("gpu_temp_c"),
        "cpu_temp_c": hw.get("cpu_temp_c"),
        "power_w": hw.get("power_w"),
        "mean_detections_per_frame": an.get("mean_detections_per_frame"),
        "mean_confidence": an.get("mean_confidence"),
        "detection_frame_rate": an.get("detection_frame_rate"),
        "peak_tracks": an.get("peak_tracks"),
        "mean_active_tracks": an.get("mean_active_tracks"),
        "mean_track_age_frames": an.get("mean_track_age_frames"),
        "skipped": False,
        "skip_reason": "",
    }


def write_csv_row(path: Path, row: dict[str, Any]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(row.keys())
    header = None
    if path.exists():
        with path.open(newline="", encoding="utf-8") as f:
            header = next(csv.reader(f), None)
    if header is not None and header != fields:
        # Appending under another schema (older version, or the C++ binary) misaligns every column.
        n = 1
        while path.with_name(f"{path.stem}.{n}{path.suffix}").exists():
            n += 1
        moved = path.rename(path.with_name(f"{path.stem}.{n}{path.suffix}"))
        log("WARN", "bench", f"{path.name} has different columns; moved it to {moved.name}")
        header = None
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if header is None:
            w.writeheader()
        w.writerow(json_sanitize(row))
