from __future__ import annotations

import math

import numpy as np

from cudatracker.types import LetterboxMeta


def letterbox_geometry(
    src_w: int,
    src_h: int,
    imgsz: int,
    letterbox: bool = True,
) -> LetterboxMeta:
    if src_w <= 0 or src_h <= 0:
        raise ValueError(f"invalid source size {src_w}x{src_h}")
    if not letterbox:
        scale_x = imgsz / float(src_w)
        scale_y = imgsz / float(src_h)
        return LetterboxMeta(
            scale=scale_x,
            pad_x=0,
            pad_y=0,
            new_w=imgsz,
            new_h=imgsz,
            src_w=src_w,
            src_h=src_h,
            imgsz=imgsz,
            scale_x=scale_x,
            scale_y=scale_y,
        )
    scale = min(imgsz / float(src_w), imgsz / float(src_h))
    new_w = int(round(src_w * scale))
    new_h = int(round(src_h * scale))
    new_w = max(1, min(new_w, imgsz))
    new_h = max(1, min(new_h, imgsz))
    pad_x = (imgsz - new_w) // 2
    pad_y = (imgsz - new_h) // 2
    return LetterboxMeta(scale, pad_x, pad_y, new_w, new_h, src_w, src_h, imgsz, scale, scale)


def map_boxes_to_source(
    cx: np.ndarray,
    cy: np.ndarray,
    w: np.ndarray,
    h: np.ndarray,
    meta: LetterboxMeta,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = (cx - meta.pad_x) / meta.scale_x
    y = (cy - meta.pad_y) / meta.scale_y
    bw = w / meta.scale_x
    bh = h / meta.scale_y
    return x, y, bw, bh


def invert_letterbox_point(x: float, y: float, meta: LetterboxMeta) -> tuple[float, float]:
    return (x - meta.pad_x) / meta.scale_x, (y - meta.pad_y) / meta.scale_y


def percentiles(values: list[float], ps: list[float]) -> list[float]:
    if not values:
        return [math.nan] * len(ps)
    arr = np.sort(np.asarray(values, dtype=np.float64))
    out = []
    for p in ps:
        k = (len(arr) - 1) * (p / 100.0)
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            out.append(float(arr[int(k)]))
        else:
            out.append(float(arr[f] * (c - k) + arr[c] * (k - f)))
    return out
