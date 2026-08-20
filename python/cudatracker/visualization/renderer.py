from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from cudatracker.types import StageTimes, Track


def _class_color(class_id: int) -> tuple[int, int, int]:
    rng = np.random.RandomState(class_id * 97 + 13)
    bgr = rng.randint(40, 220, size=3).tolist()
    return int(bgr[0]), int(bgr[1]), int(bgr[2])


def draw(
    frame: np.ndarray,
    tracks: list[Track],
    times: StageTimes,
    fps: float,
    preprocess: str,
    backend: str,
    precision: str,
    hud: bool,
    trails: bool,
    hardware_panel: bool,
    hardware: dict[str, Any] | None,
    status: str = "",
    frame_index: int = 0,
) -> np.ndarray:
    vis = frame
    h, w = vis.shape[:2]
    for tr in tracks:
        color = _class_color(tr.class_id)
        x1 = int(tr.x - tr.w / 2)
        y1 = int(tr.y - tr.h / 2)
        x2 = int(tr.x + tr.w / 2)
        y2 = int(tr.y + tr.h / 2)
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = f"{tr.class_name}  ID {tr.track_id}"
        cv2.putText(vis, label, (x1, max(16, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        if trails and len(tr.history) > 1:
            pts = np.array(tr.history, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(vis, [pts], False, color, 1, cv2.LINE_AA)
    if not hud:
        return vis
    slow_name, slow_ms = times.slowest()
    lines = [
        f"CUDATracker  #{times.frame_index or frame_index}",
        f"{times.end_to_end_ms:5.1f} ms   {fps:5.1f} FPS   bottleneck {slow_name}",
        f"{preprocess}  |  {backend} {precision}  |  SORT",
    ]
    for key, label in times.STAGE_ORDER:
        wall = times._wall(key)
        gpu = times._gpu(key)
        mark = " <" if key == slow_name else ""
        gpu_bit = f"  gpu {gpu:4.1f}" if gpu > 0.05 else ""
        lines.append(f"{label:<16} {wall:5.1f}{gpu_bit}{mark}")
    lines.append(f"{'TOTAL':<16} {times.end_to_end_ms:5.1f}")
    lines.append(f"cpu {times.cpu_ms:4.1f}  gpu {times.gpu_ms:4.1f}  wait {times.sync_ms:4.1f}")
    if status:
        lines.append(status)
    if hardware_panel and hardware:
        def fmt(key, suffix):
            v = hardware.get(key)
            return "na" if v is None else f"{v:.0f}{suffix}" if not isinstance(v, str) else v

        lines.append(
            f"CPU {fmt('cpu_util', '%')}  GPU {fmt('gpu_util', '%')}  "
            f"GPU {fmt('gpu_temp_c', 'C')}  {fmt('power_w', 'W')}"
        )
    pad = 10
    line_h = 18
    bar_h = pad * 2 + line_h * len(lines)
    bar_w = min(w, 560)
    overlay = vis.copy()
    cv2.rectangle(overlay, (0, 0), (bar_w, bar_h), (0, 0, 0), -1)
    vis = cv2.addWeighted(overlay, 0.45, vis, 0.55, 0)
    y = pad + 14
    for i, text in enumerate(lines):
        scale = 0.72 if i == 1 else 0.48
        thick = 2 if i == 1 else 1
        cv2.putText(vis, text, (pad, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (240, 240, 240), thick, cv2.LINE_AA)
        y += 22 if i == 1 else line_h
    return vis
