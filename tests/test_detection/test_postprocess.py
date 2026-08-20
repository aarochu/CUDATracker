from __future__ import annotations

import numpy as np

from cudatracker.detection.postprocess import _as_index_array, decode_yolo, nms_xywh
from cudatracker.preprocessing.geometry import invert_letterbox_point, letterbox_geometry


def test_letterbox_pad_even_split():
    meta = letterbox_geometry(1280, 720, 640, letterbox=True)
    assert meta.new_w + 2 * meta.pad_x <= meta.imgsz + 1
    assert meta.new_h + meta.pad_y <= meta.imgsz
    assert meta.pad_x >= 0 and meta.pad_y >= 0


def test_letterbox_odd_width():
    meta = letterbox_geometry(333, 200, 640, letterbox=True)
    assert meta.new_w <= 640
    assert meta.new_h <= 640
    src = invert_letterbox_point(meta.pad_x + meta.new_w / 2, meta.pad_y + meta.new_h / 2, meta)
    assert abs(src[0] - 166.5) < 2.0


def test_mapping_invertible_center():
    meta = letterbox_geometry(640, 480, 640, True)
    x, y = invert_letterbox_point(meta.pad_x + 100 * meta.scale, meta.pad_y + 80 * meta.scale, meta)
    assert abs(x - 100) < 1e-3
    assert abs(y - 80) < 1e-3


def test_stretch_uses_independent_axes():
    meta = letterbox_geometry(640, 480, 640, letterbox=False)
    assert abs(meta.scale_x - 1.0) < 1e-6
    assert abs(meta.scale_y - 640.0 / 480.0) < 1e-6
    x, y = invert_letterbox_point(320, 320, meta)
    assert abs(x - 320) < 1e-3
    assert abs(y - 240) < 1e-3


def test_nms_drops_overlap():
    boxes = np.array(
        [
            [10, 10, 40, 40],
            [12, 12, 40, 40],
            [200, 200, 20, 20],
        ],
        dtype=np.float32,
    )
    scores = np.array([0.9, 0.8, 0.7], dtype=np.float32)
    keep = nms_xywh(boxes, scores, 0.5)
    assert 0 in keep
    assert 2 in keep
    assert 1 not in keep


def test_class_aware_nms_keeps_two_classes():
    from cudatracker.detection.postprocess import decode_yolo

    arr = np.zeros((2, 84), dtype=np.float32)
    arr[0, :4] = [20, 20, 30, 30]
    arr[0, 4] = 0.9
    arr[1, :4] = [22, 22, 30, 30]
    arr[1, 5] = 0.85
    meta = letterbox_geometry(640, 640, 640, True)
    dets = decode_yolo(arr[None, ...], meta, 0.25, 0.45)
    assert len(dets) == 2
    assert {d.class_id for d in dets} == {0, 1}


def test_nms_index_shapes():
    assert _as_index_array(None).size == 0
    assert _as_index_array([]).size == 0
    assert _as_index_array(0).tolist() == [0]
    assert _as_index_array([[2], [5]]).tolist() == [2, 5]


def _yolo_cn(c=84, n=256) -> np.ndarray:
    arr = np.zeros((c, n), dtype=np.float32)
    arr[0, 0] = 320
    arr[1, 0] = 240
    arr[2, 0] = 40
    arr[3, 0] = 50
    arr[4, 0] = 0.91
    return arr


def test_decode_yolo_layouts_and_letterbox_map():
    meta = letterbox_geometry(640, 480, 640, True)
    raw = _yolo_cn()
    for shaped in (raw, raw.T, raw[None, ...], raw[None, None, ...], raw.T[None, ...]):
        dets = decode_yolo(shaped, meta, 0.25, 0.45)
        assert len(dets) == 1
        assert dets[0].class_id == 0
        assert abs(dets[0].x - 320.0) < 1e-2
        assert abs(dets[0].y - (240.0 - meta.pad_y) / meta.scale_y) < 1e-2
        assert abs(dets[0].w - 40.0) < 1e-2
        assert abs(dets[0].h - 50.0) < 1e-2
