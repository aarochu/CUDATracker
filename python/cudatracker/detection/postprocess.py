from __future__ import annotations

import cv2
import numpy as np

from cudatracker.detection.coco import class_name
from cudatracker.preprocessing.geometry import map_boxes_to_source
from cudatracker.types import Detection, LetterboxMeta


def _to_numpy(output) -> np.ndarray:
    if isinstance(output, (list, tuple)):
        output = output[0]
    if hasattr(output, "detach"):
        output = output.detach()
    if hasattr(output, "float"):
        output = output.float()
    if hasattr(output, "cpu"):
        output = output.cpu()
    if hasattr(output, "numpy"):
        output = output.numpy()
    return np.asarray(output)


def _squeeze_yolo(arr: np.ndarray) -> np.ndarray:
    """Normalize YOLO raw output to (N, 4+C). Accepts (1,C,N), (1,N,C), (C,N), (N,C), extra leading 1s."""
    while arr.ndim > 2 and arr.shape[0] == 1:
        arr = arr[0]
    if arr.ndim == 3:
        arr = arr[0]
    if arr.ndim != 2:
        raise ValueError(f"Unexpected YOLO output shape {arr.shape}")
    # Ultralytics v8 is 84 (4+80); v5 is 85 (4+1+80). Prefer that over "smaller dim first",
    # which breaks (C, N) when N is tiny (N < C).
    channel_like = {84, 85}
    if arr.shape[0] in channel_like and arr.shape[1] not in channel_like:
        arr = arr.T
    elif arr.shape[0] not in channel_like and arr.shape[1] in channel_like:
        pass
    elif arr.shape[0] < arr.shape[1] and arr.shape[0] <= 85:
        arr = arr.T
    if arr.shape[1] < 5:
        raise ValueError(f"YOLO output last dim too small: {arr.shape}")
    return arr


def _as_index_array(idxs) -> np.ndarray:
    # OpenCV NMSBoxes may return None, [], (k,1), or a scalar when k==1.
    if idxs is None:
        return np.zeros((0,), dtype=np.int32)
    arr = np.asarray(idxs).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int32)
    return arr.astype(np.int32, copy=False)


def decode_yolo(
    raw,
    meta: LetterboxMeta,
    conf_thres: float,
    iou_thres: float,
) -> list[Detection]:
    arr = _squeeze_yolo(_to_numpy(raw))
    boxes = arr[:, :4]
    scores = arr[:, 4:]
    # YOLOv5 has objectness in col 4 then classes; YOLOv8 is classes only.
    if scores.shape[1] == 81:
        obj = scores[:, 0]
        cls = scores[:, 1:]
        class_ids = cls.argmax(axis=1)
        conf = obj * cls.max(axis=1)
    else:
        class_ids = scores.argmax(axis=1)
        conf = scores.max(axis=1)
    keep = conf >= conf_thres
    boxes = boxes[keep]
    conf = conf[keep]
    class_ids = class_ids[keep]
    if boxes.size == 0:
        return []
    cx, cy, w, h = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    x1 = cx - w / 2.0
    y1 = cy - h / 2.0
    xywh = np.stack([x1, y1, w, h], axis=1).astype(np.float32)
    score_list = conf.astype(np.float32).tolist()
    box_list = xywh.tolist()
    nms_batched = getattr(cv2.dnn, "NMSBoxesBatched", None)
    if nms_batched is not None:
        idxs = nms_batched(
            box_list,
            score_list,
            class_ids.astype(np.int32).tolist(),
            float(conf_thres),
            float(iou_thres),
        )
    else:
        idxs = cv2.dnn.NMSBoxes(box_list, score_list, float(conf_thres), float(iou_thres))
    idxs = _as_index_array(idxs)
    if idxs.size == 0:
        return []
    cx, cy, w, h = map_boxes_to_source(cx[idxs], cy[idxs], w[idxs], h[idxs], meta)
    conf = conf[idxs]
    class_ids = class_ids[idxs]
    dets: list[Detection] = []
    for i in range(len(idxs)):
        cid = int(class_ids[i])
        dets.append(
            Detection(
                x=float(cx[i]),
                y=float(cy[i]),
                w=float(w[i]),
                h=float(h[i]),
                confidence=float(conf[i]),
                class_id=cid,
                class_name=class_name(cid),
            )
        )
    return dets


def nms_xywh(
    boxes: np.ndarray,
    scores: np.ndarray,
    iou_thres: float,
) -> np.ndarray:
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.int32)
    idxs = cv2.dnn.NMSBoxes(
        boxes.astype(np.float32).tolist(),
        scores.astype(np.float32).tolist(),
        0.0,
        float(iou_thres),
    )
    return _as_index_array(idxs)
