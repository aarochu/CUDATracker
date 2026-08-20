from __future__ import annotations

import numpy as np

from cudatracker.tracking.hungarian import linear_sum_assignment
from cudatracker.tracking.sort import SortTracker
from cudatracker.types import Detection


def _box(x, y, w=40, h=40, cid=0, conf=0.9) -> Detection:
    return Detection(x=x, y=y, w=w, h=h, confidence=conf, class_id=cid, class_name="person")


def test_ids_persist_on_parallel_motion():
    tr = SortTracker(max_age=5, min_hits=1, iou_threshold=0.3)
    a_id = b_id = None
    for t in range(8):
        dets = [_box(50 + t * 5, 50), _box(200 + t * 5, 80)]
        tracks = tr.update(dets)
        assert len(tracks) == 2
        ids = sorted(t.track_id for t in tracks)
        if a_id is None:
            a_id, b_id = ids
        else:
            assert ids == [a_id, b_id]


def test_max_age_drops_track():
    tr = SortTracker(max_age=2, min_hits=1, iou_threshold=0.3)
    tr.update([_box(40, 40)])
    tr.update([])
    tr.update([])
    leftover = tr.update([])
    assert leftover == []
    assert all(t.time_since_update > 2 for t in tr.tracks) or tr.tracks == []


def test_closer_track_keeps_the_only_detection():
    tr = SortTracker(max_age=5, min_hits=1, iou_threshold=0.1)
    first = tr.update([_box(50, 50), _box(400, 50)])
    assert len(first) == 2
    left_id = next(t.track_id for t in first if t.x < 200)
    later = tr.update([_box(55, 50)])
    assert len(later) == 1
    assert later[0].track_id == left_id


def test_ids_survive_crossing():
    tr = SortTracker(max_age=10, min_hits=1, iou_threshold=0.2)
    first = tr.update([_box(50, 80, 30, 50), _box(250, 80, 30, 50)])
    left_id = next(t.track_id for t in first if t.x < 150)
    right_id = next(t.track_id for t in first if t.x >= 150)
    for step in range(1, 16):
        x_left = 50 + step * 12
        x_right = 250 - step * 12
        tracks = tr.update([_box(x_left, 80, 30, 50), _box(x_right, 80, 30, 50)])
        assert len(tracks) == 2
        near_left = min(tracks, key=lambda t: abs(t.x - x_left))
        near_right = min(tracks, key=lambda t: abs(t.x - x_right))
        assert near_left.track_id == left_id
        assert near_right.track_id == right_id


def test_hungarian_square_and_rectangular():
    rows, cols = linear_sum_assignment(np.array([[0.1, 0.9], [0.8, 0.2]]))
    assert sorted(zip(rows.tolist(), cols.tolist())) == [(0, 0), (1, 1)]
    rows, cols = linear_sum_assignment(np.array([[0.9, 0.1, 0.8]]))
    assert rows.tolist() == [0]
    assert cols.tolist() == [1]
    rows, cols = linear_sum_assignment(np.zeros((0, 3)))
    assert rows.size == 0 and cols.size == 0
