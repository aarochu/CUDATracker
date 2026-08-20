from __future__ import annotations

import cv2
import numpy as np

from cudatracker.preprocessing.geometry import letterbox_geometry
from cudatracker.types import LetterboxMeta


def cpu_preprocess(
    bgr: np.ndarray,
    imgsz: int,
    letterbox: bool = True,
    pad_value: int = 114,
) -> tuple[np.ndarray, LetterboxMeta]:
    src_h, src_w = bgr.shape[:2]
    meta = letterbox_geometry(src_w, src_h, imgsz, letterbox=letterbox)
    if letterbox:
        resized = cv2.resize(bgr, (meta.new_w, meta.new_h), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((imgsz, imgsz, 3), pad_value, dtype=np.uint8)
        canvas[meta.pad_y : meta.pad_y + meta.new_h, meta.pad_x : meta.pad_x + meta.new_w] = resized
    else:
        canvas = cv2.resize(bgr, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
    chw = np.transpose(rgb.astype(np.float32) / 255.0, (2, 0, 1))
    tensor = np.expand_dims(np.ascontiguousarray(chw), 0)
    return tensor, meta
