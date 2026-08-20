from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cudatracker.config import AppConfig, apply_overrides, load_config
from cudatracker.inference.base import BackendNotAvailable
from cudatracker.logutil import log
from cudatracker.pipeline import Pipeline, write_run_artifacts


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="cudatracker", description="CUDATracker live pipeline and bench mode")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--source")
    p.add_argument("--resolution")
    p.add_argument("--preprocess", choices=["cpu", "cuda"])
    p.add_argument("--backend", choices=["pytorch", "onnx", "tensorrt"])
    p.add_argument("--precision", choices=["fp32", "fp16"])
    p.add_argument("--device", choices=["gpu", "cpu"])
    p.add_argument("--fused", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--no-vis", action="store_true")
    p.add_argument("--weights")
    p.add_argument("--onnx")
    p.add_argument("--engine")
    p.add_argument("--imgsz", type=int)
    p.add_argument("--output-video")
    p.add_argument("--output-metrics")
    p.add_argument("--bench", action="store_true", help="Warmup + measure, then write JSON/CSV and exit")
    p.add_argument("--warmup", type=int)
    p.add_argument("--frames", type=int)
    return p


def run_live(cfg: AppConfig) -> int:
    pipe = None
    try:
        pipe = Pipeline(cfg, loop_source=True)
        pipe.open()
    except (BackendNotAvailable, FileNotFoundError, RuntimeError) as exc:
        log("ERROR", "app", str(exc))
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass
        return 2
    log("INFO", "app", "keys: q quit  h hardware  t trails  d hud  p cpu/cuda  1 fp32  2 fp16")
    import cv2

    try:
        while True:
            vis, times = pipe.step()
            if vis is None:
                log("INFO", "capture", "end of source")
                break
            if cfg.visualization.enabled:
                cv2.imshow("CUDATracker", vis)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("h"):
                    pipe.hw_panel = not pipe.hw_panel
                if key == ord("t"):
                    pipe.trails = not pipe.trails
                if key == ord("d"):
                    pipe.hud = not pipe.hud
                if key == ord("p"):
                    _toggle_preprocess(pipe, cfg)
                if key in (ord("1"), ord("2")):
                    _reload_precision(pipe, cfg, "fp32" if key == ord("1") else "fp16")
            if pipe.analytics.frames % 30 == 0:
                for line in times.table_lines():
                    log("INFO", "perf", line)
    finally:
        pipe.close()
        if cfg.visualization.enabled:
            cv2.destroyAllWindows()
    return 0


def run_bench(cfg: AppConfig) -> int:
    cfg.visualization.enabled = False
    warmup = cfg.benchmark.warmup_frames
    measure = cfg.benchmark.measure_frames
    pipe = None
    try:
        pipe = Pipeline(cfg, loop_source=True)
        pipe.open()
    except (BackendNotAvailable, FileNotFoundError, RuntimeError) as exc:
        log("ERROR", "app", str(exc))
        if pipe is not None:
            try:
                pipe.close()
            except Exception:
                pass
        return 2
    try:
        for i in range(warmup + measure):
            vis, _ = pipe.step()
            if vis is None:
                log("WARN", "bench", f"source ended at frame {i}")
                break
            if i + 1 == warmup:
                pipe.begin_measure()
                log("INFO", "bench", f"warmup done ({warmup})")
        n = len(pipe._e2e)
        skip = min(warmup, n)
        if n - skip <= 0:
            log("ERROR", "bench", "no measure frames (video too short and looping failed)")
            return 2
        summary = pipe.record_window(skip=skip)
        write_run_artifacts(cfg, summary)
        log("INFO", "bench", f"fps={summary['fps']:.2f} e2e_mean={summary['end_to_end']['mean']:.2f}ms")
        for line in pipe.times.table_lines():
            log("INFO", "perf", line)
    finally:
        pipe.close()
    return 0


def _toggle_preprocess(pipe: Pipeline, cfg: AppConfig) -> None:
    nxt = "cuda" if cfg.preprocess.backend == "cpu" else "cpu"
    pipe.status = f"switching preprocess {nxt}"
    cfg.preprocess.backend = nxt
    try:
        from cudatracker.preprocessing.cuda import CudaPreprocessor

        if nxt == "cuda":
            pipe.cuda_pre = CudaPreprocessor(
                cfg.model.imgsz, cfg.preprocess.pad_value, cfg.preprocess.fused, cfg.preprocess.letterbox
            )
        else:
            pipe.cuda_pre = None
        pipe.status = ""
        log("INFO", "app", f"preprocess={nxt}")
    except Exception as exc:
        log("ERROR", "app", f"preprocess switch failed: {exc}")
        cfg.preprocess.backend = "cpu"
        pipe.cuda_pre = None
        pipe.status = "cuda preprocess failed"


def _reload_precision(pipe: Pipeline, cfg: AppConfig, precision: str) -> None:
    if precision == cfg.inference.precision:
        return
    pipe.status = f"loading {precision}"
    old_prec = cfg.inference.precision
    cfg.inference.precision = precision
    try:
        from cudatracker.inference.factory import create_backend

        new_backend = create_backend(cfg)
    except Exception as exc:
        cfg.inference.precision = old_prec
        log("ERROR", "app", str(exc))
        pipe.status = "reload failed"
        return
    old = pipe.backend
    pipe.backend = new_backend
    if hasattr(old, "close"):
        old.close()
    pipe.status = ""
    log("INFO", "app", f"precision={precision}")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        log("ERROR", "app", f"config not found: {cfg_path}")
        return 2
    cfg = load_config(cfg_path)
    cfg = apply_overrides(cfg, **vars(args))
    if args.warmup is not None:
        cfg.benchmark.warmup_frames = args.warmup
    if args.frames is not None:
        cfg.benchmark.measure_frames = args.frames
    log(
        "INFO",
        "app",
        f"source={cfg.input.source} pre={cfg.preprocess.backend} "
        f"infer={cfg.inference.backend}/{cfg.inference.precision} vis={cfg.visualization.enabled}",
    )
    if args.bench:
        return run_bench(cfg)
    return run_live(cfg)


if __name__ == "__main__":
    sys.exit(main())
