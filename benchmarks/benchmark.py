#!/usr/bin/env python3
"""Walk the benchmark matrix. Missing backends are skipped and recorded."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from copy import deepcopy
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from cudatracker.config import AppConfig, load_config  # noqa: E402
from cudatracker.inference.base import BackendNotAvailable  # noqa: E402
from cudatracker.logutil import log  # noqa: E402
from cudatracker.pipeline import Pipeline, flatten_row, write_csv_row, write_run_artifacts  # noqa: E402
from cudatracker.telemetry.hardware import json_sanitize  # noqa: E402


def legal(backend: str, precision: str) -> tuple[bool, str]:
    if backend == "onnx" and precision == "fp16":
        return False, "ONNX Runtime path uses an FP32 graph; skip fp16 cell"
    return True, ""


def run_cell(base: AppConfig, pre: str, backend: str, precision: str, resolution: str, model: str) -> dict:
    cfg = deepcopy(base)
    cfg.preprocess.backend = pre
    cfg.inference.backend = backend
    cfg.inference.precision = precision
    cfg.input.resolution = resolution
    cfg.model.name = model
    cfg.visualization.enabled = False
    cfg.output.metrics = str(
        Path(cfg.benchmark.results_dir)
        / f"{model}_{pre}_{backend}_{precision}_{resolution}.json"
    )
    ok, reason = legal(backend, precision)
    if not ok:
        row = flatten_row(cfg, {"fps": 0, "end_to_end": {}, "stages": {}, "hardware": {}, "analytics": {}})
        row["skipped"] = True
        row["skip_reason"] = reason
        return row
    pipe = None
    try:
        pipe = Pipeline(cfg, loop_source=True)
        pipe.open()
    except (BackendNotAvailable, FileNotFoundError, ImportError, RuntimeError) as exc:
        if pipe is not None:
            pipe.close()
        row = flatten_row(cfg, {"fps": 0, "end_to_end": {}, "stages": {}, "hardware": {}, "analytics": {}})
        row["skipped"] = True
        row["skip_reason"] = str(exc)
        log("WARN", "bench", f"skip {pre}/{backend}/{precision}/{resolution}: {exc}")
        return row
    warmup = cfg.benchmark.warmup_frames
    measure = cfg.benchmark.measure_frames
    try:
        for i in range(warmup + measure):
            vis, _ = pipe.step()
            if vis is None:
                break
            if i + 1 == warmup:
                pipe.begin_measure()
        n = len(pipe._e2e)
        skip = min(warmup, n)
        if n == skip:
            row = flatten_row(cfg, {"fps": 0, "end_to_end": {}, "stages": {}, "hardware": {}, "analytics": {}})
            row["skipped"] = True
            row["skip_reason"] = "source ended before any measured frames"
            return row
        summary = pipe.record_window(skip=skip)
        write_run_artifacts(cfg, summary)
        row = flatten_row(cfg, summary)
        row["skipped"] = False
        log("INFO", "bench", f"{pre}/{backend}/{precision}/{resolution} fps={row['fps']:.2f}")
        return row
    finally:
        pipe.close()


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/benchmark.yaml")
    p.add_argument("--quick", action="store_true", help="2 warmup / 8 measure frames for a smoke pass")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.quick:
        cfg.benchmark.warmup_frames = 2
        cfg.benchmark.measure_frames = 8
    results_dir = Path(cfg.benchmark.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = results_dir / "matrix.csv"
    if summary_csv.exists():
        summary_csv.unlink()
    rows = []
    for model in cfg.matrix.model:
        for pre in cfg.matrix.preprocess:
            for backend in cfg.matrix.backend:
                for precision in cfg.matrix.precision:
                    for resolution in cfg.matrix.resolution:
                        row = run_cell(cfg, pre, backend, precision, resolution, model)
                        rows.append(row)
                        write_csv_row(summary_csv, row)
    (results_dir / "matrix.json").write_text(json.dumps(json_sanitize(rows), indent=2), encoding="utf-8")
    log("INFO", "bench", f"matrix complete -> {summary_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
