#!/usr/bin/env python3
"""Charts from benchmarks/results/matrix.csv. No CSV -> exit 2. No invented numbers."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _f(row, key):
    v = row.get(key, "")
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _key(row, *fields):
    return tuple(str(row.get(f, "")) for f in fields)


def paired_speedup(live: list[dict], left_field: str, left_val: str, right_val: str, match: tuple[str, ...]):
    """Speedup right/left FPS for rows that share `match` keys."""
    by = defaultdict(dict)
    for r in live:
        by[_key(r, *match)][str(r.get(left_field, ""))] = r
    out = []
    for shared, group in by.items():
        a, b = group.get(left_val), group.get(right_val)
        if not a or not b:
            continue
        fa, fb = _f(a, "fps"), _f(b, "fps")
        if fa is None or fb is None or fa <= 0:
            continue
        label = "/".join(str(x) for x in shared if x)
        out.append((label or f"{left_val}->{right_val}", fb / fa))
    return out


def fps_by_resolution(live: list[dict]):
    series = defaultdict(list)
    for r in live:
        fps = _f(r, "fps")
        if fps is None:
            continue
        name = f"{r.get('preprocess')}/{r.get('backend')}/{r.get('precision')}/{r.get('model')}"
        series[name].append((str(r.get("resolution")), fps))
    return series


def resolution_order(series) -> list[str]:
    def pixels(res: str) -> int:
        try:
            w, h = res.lower().split("x")
            return int(w) * int(h)
        except ValueError:
            return 0

    return sorted({res for pts in series.values() for res, _ in pts}, key=lambda r: (pixels(r), r))


def _bar(ax, labels, vals, ylabel, title):
    ax.bar(range(len(vals)), vals)
    ax.set_xticks(range(len(vals)), labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(ylabel)
    ax.set_title(title)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="benchmarks/results")
    args = p.parse_args()
    csv_path = Path(args.input) / "matrix.csv"
    if not csv_path.exists():
        print(f"no {csv_path}. Run python benchmarks/benchmark.py first.")
        return 2
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib missing")
        return 2
    with csv_path.open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    live = [r for r in rows if str(r.get("skipped", "")).lower() in ("", "false", "0")]
    out = Path("benchmarks/plots")
    out.mkdir(parents=True, exist_ok=True)
    if not live:
        print("every cell was skipped; not drawing charts")
        return 0
    labels = [f"{r['preprocess']}/{r['backend']}/{r['precision']}/{r['resolution']}" for r in live]
    fps = [_f(r, "fps") or 0 for r in live]
    mean = [_f(r, "e2e_mean_ms") or 0 for r in live]
    p95 = [_f(r, "e2e_p95_ms") or 0 for r in live]

    fig, ax = plt.subplots(figsize=(10, 4))
    _bar(ax, labels, fps, "FPS", "FPS by cell")
    fig.tight_layout()
    fig.savefig(out / "fps.png", dpi=140)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(range(len(live)), mean, label="mean")
    ax.plot(range(len(live)), p95, "k.", label="p95")
    ax.set_xticks(range(len(live)), labels, rotation=75, ha="right", fontsize=7)
    ax.set_ylabel("ms")
    ax.set_title("End-to-end latency")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "latency.png", dpi=140)
    plt.close(fig)

    for col, ylabel, fname in (
        ("cpu_util", "CPU %", "cpu.png"),
        ("gpu_util", "GPU %", "gpu.png"),
    ):
        vals = [_f(r, col) for r in live]
        if all(v is None for v in vals):
            continue
        fig, ax = plt.subplots(figsize=(10, 4))
        _bar(ax, labels, [v if v is not None else 0 for v in vals], ylabel, ylabel)
        fig.tight_layout()
        fig.savefig(out / fname, dpi=140)
        plt.close(fig)

    speedups = [
        ("CPU to CUDA preprocess", paired_speedup(live, "preprocess", "cpu", "cuda", ("backend", "precision", "resolution", "model"))),
        ("PyTorch to TensorRT", paired_speedup(live, "backend", "pytorch", "tensorrt", ("preprocess", "precision", "resolution", "model"))),
        ("FP32 to FP16", paired_speedup(live, "precision", "fp32", "fp16", ("preprocess", "backend", "resolution", "model"))),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    drew = False
    for ax, (title, pairs) in zip(axes, speedups):
        if not pairs:
            ax.set_title(title)
            ax.set_xticks([])
            continue
        drew = True
        names, vals = zip(*pairs)
        ax.bar(range(len(vals)), vals)
        ax.set_xticks(range(len(vals)), names, rotation=80, ha="right", fontsize=6)
        ax.set_ylabel("x")
        ax.set_title(title)
    fig.tight_layout()
    if drew:
        fig.savefig(out / "speedup.png", dpi=140)
    plt.close(fig)

    series = fps_by_resolution(live)
    if series:
        order = resolution_order(series)
        pos = {res: i for i, res in enumerate(order)}
        fig, ax = plt.subplots(figsize=(8, 4))
        for name, pts in series.items():
            pts = sorted(pts, key=lambda p: pos[p[0]])
            ax.plot([pos[r] for r, _ in pts], [f for _, f in pts], marker="o", label=name)
        ax.set_xticks(range(len(order)), order, rotation=30, ha="right")
        ax.set_ylabel("FPS")
        ax.set_title("FPS vs resolution")
        ax.legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(out / "fps_vs_resolution.png", dpi=140)
        plt.close(fig)

    power = [(_f(r, "power_w"), _f(r, "fps")) for r in live]
    if any(pw is not None and fp is not None for pw, fp in power):
        fig, ax = plt.subplots(figsize=(6, 4))
        xs = [pw for pw, fp in power if pw is not None and fp is not None]
        ys = [fp for pw, fp in power if pw is not None and fp is not None]
        ax.scatter(xs, ys)
        ax.set_xlabel("W")
        ax.set_ylabel("FPS")
        ax.set_title("Power vs FPS")
        fig.tight_layout()
        fig.savefig(out / "power_vs_fps.png", dpi=140)
        plt.close(fig)

    print(f"wrote plots under {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
