from __future__ import annotations

from cudatracker.analytics.stats import Analytics
from cudatracker.types import Detection, Track


def test_mean_track_age_keeps_history():
    a = Analytics()
    a.update([], [Track(track_id=1, class_id=0, class_name="person", x=0, y=0, w=10, h=10, confidence=0.9, age=1, hits=1)])
    a.update([], [Track(track_id=1, class_id=0, class_name="person", x=0, y=0, w=10, h=10, confidence=0.9, age=5, hits=5)])
    a.update([], [])
    s = a.summary()
    assert s["mean_track_age_frames"] == 5
    assert s["peak_tracks"] == 1
    assert s["mean_active_tracks"] == 2 / 3


def test_counts_detections():
    a = Analytics()
    d = Detection(x=1, y=1, w=2, h=2, confidence=0.8, class_id=0, class_name="person")
    a.update([d], [])
    s = a.summary()
    assert s["mean_detections_per_frame"] == 1
    assert s["detection_frame_rate"] == 1
    assert s["class_counts"]["person"] == 1
