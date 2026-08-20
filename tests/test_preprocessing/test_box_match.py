from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from cudatracker.detection.postprocess import decode_yolo
from cudatracker.preprocessing.cpu import cpu_preprocess
from cudatracker.tracking.hungarian import linear_sum_assignment
from cudatracker.tracking.sort import _iou


def _xyxy(d):
    x1, y1, x2, y2 = d.as_xyxy()
    return [x1, y1, x2, y2]


def test_cpu_vs_cuda_high_conf_boxes():
    """SOW: same backend, CUDA vs CPU preprocess — IoU ≥ 0.99 or center error < 1 px on 640."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA GPU")
    weights = Path("models/yolov8n.pt")
    clip = Path("samples/vtest.avi")
    if not weights.exists() or not clip.exists():
        pytest.skip("need models/yolov8n.pt and samples/vtest.avi")
    import cv2

    from cudatracker.inference.pytorch_backend import PyTorchBackend
    from cudatracker.preprocessing.cuda import CudaPreprocessor

    cap = cv2.VideoCapture(str(clip))
    ok, bgr = cap.read()
    cap.release()
    if not ok or bgr is None:
        pytest.skip("could not read sample frame")
    bgr = cv2.resize(bgr, (640, 480))
    backend = PyTorchBackend(str(weights), "gpu", "fp32", 640)
    cpu_t, meta = cpu_preprocess(bgr, 640, True, 114)
    cuda_pre = CudaPreprocessor(640, 114, fused=False, letterbox=True)
    gpu_t, meta2, _, _ = cuda_pre(bgr)
    assert meta.pad_x == meta2.pad_x and meta.pad_y == meta2.pad_y
    cpu_dets = [d for d in decode_yolo(backend.infer(torch.from_numpy(cpu_t)), meta, 0.5, 0.45) if d.confidence >= 0.5]
    gpu_dets = [d for d in decode_yolo(backend.infer(gpu_t), meta2, 0.5, 0.45) if d.confidence >= 0.5]
    backend.close()
    if not cpu_dets or not gpu_dets:
        pytest.skip("no high-conf boxes on this frame")
    a = np.stack([_xyxy(d) for d in cpu_dets])
    b = np.stack([_xyxy(d) for d in gpu_dets])
    iou = _iou(a, b)
    rows, cols = linear_sum_assignment(1.0 - iou)
    matched = 0
    for r, c in zip(rows, cols):
        if iou[r, c] < 0.3:
            continue
        matched += 1
        da, db = cpu_dets[int(r)], gpu_dets[int(c)]
        center = abs(da.x - db.x) < 1.0 and abs(da.y - db.y) < 1.0
        assert iou[r, c] >= 0.99 or center, (float(iou[r, c]), da.x, da.y, db.x, db.y)
    assert matched >= 1
