from __future__ import annotations

import csv
from pathlib import Path

from cudatracker.config import AppConfig
from cudatracker.pipeline import flatten_row, write_csv_row


def test_csv_row_written(tmp_path):
    cfg = AppConfig()
    summary = {
        "fps": 10.0,
        "end_to_end": {"mean": 100, "p50": 90, "p95": 120, "p99": 130},
        "stages": {"preprocess": {"mean": 4}},
        "hardware": {},
        "analytics": {},
    }
    row = flatten_row(cfg, summary)
    path = tmp_path / "t.csv"
    write_csv_row(path, row)
    write_csv_row(path, row)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["backend"] == "pytorch"


def test_skip_reason_field():
    cfg = AppConfig()
    row = flatten_row(cfg, {"fps": 0, "end_to_end": {}, "stages": {}, "hardware": {}, "analytics": {}})
    row["skipped"] = True
    row["skip_reason"] = "no engine"
    assert row["skip_reason"] == "no engine"


def test_analytics_reset_drops_warmup():
    from cudatracker.analytics.stats import Analytics
    from cudatracker.types import Detection, Track

    an = Analytics()
    det = Detection(x=1, y=1, w=10, h=10, confidence=0.9, class_id=0, class_name="person")
    tr = Track(track_id=1, class_id=0, class_name="person", x=1, y=1, w=10, h=10, confidence=0.9, age=3, hits=3)
    an.update([det], [tr])
    an.reset()
    assert an.frames == 0
    assert an.summary()["peak_tracks"] == 0
    assert an.summary()["mean_detections_per_frame"] == 0


def test_capture_loops_short_file(tmp_path):
    import cv2
    import numpy as np

    from cudatracker.capture.camera import Capture

    path = tmp_path / "tiny.avi"
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (64, 48))
    for i in range(3):
        w.write(np.full((48, 64, 3), i, dtype=np.uint8))
    w.release()
    cap = Capture(str(path), 64, 48, loop=True)
    cap.open()
    try:
        frames = [cap.read() for _ in range(7)]
    finally:
        cap.close()
    assert all(p is not None for p in frames)
    assert frames[-1].frame_index == 6
