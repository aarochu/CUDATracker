from __future__ import annotations

import numpy as np

from cudatracker.tracking.hungarian import linear_sum_assignment
from cudatracker.types import Detection, Track


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    # a,b: (N,4) / (M,4) as x1,y1,x2,y2
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    ax1, ay1, ax2, ay2 = a[:, 0:1], a[:, 1:2], a[:, 2:3], a[:, 3:4]
    bx1, by1, bx2, by2 = b[:, 0], b[:, 1], b[:, 2], b[:, 3]
    iw = np.maximum(0.0, np.minimum(ax2, bx2) - np.maximum(ax1, bx1))
    ih = np.maximum(0.0, np.minimum(ay2, by2) - np.maximum(ay1, by1))
    inter = iw * ih
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-9)


class _Kalman:
    """SORT 7-state box Kalman: x,y,s,r,vx,vy,vs."""

    def __init__(self, cx: float, cy: float, s: float, r: float) -> None:
        self.x = np.array([cx, cy, s, r, 0.0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(7, dtype=np.float64)
        self.P[4:, 4:] *= 1000.0
        self.P *= 10.0
        self.F = np.eye(7, dtype=np.float64)
        self.F[0, 4] = 1
        self.F[1, 5] = 1
        self.F[2, 6] = 1
        self.H = np.zeros((4, 7), dtype=np.float64)
        self.H[0, 0] = 1
        self.H[1, 1] = 1
        self.H[2, 2] = 1
        self.H[3, 3] = 1
        self.R = np.eye(4, dtype=np.float64)
        self.R[2:, 2:] *= 10.0
        self.Q = np.eye(7, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01

    def predict(self) -> np.ndarray:
        if self.x[2] + self.x[6] <= 0:
            self.x[6] = 0
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:4].copy()

    def update(self, z: np.ndarray) -> None:
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(7) - K @ self.H) @ self.P

    def as_xywh(self) -> tuple[float, float, float, float]:
        cx, cy, s, r = self.x[:4]
        s = max(s, 1e-6)
        r = max(r, 1e-6)
        w = np.sqrt(s * r)
        h = s / w
        return float(cx), float(cy), float(w), float(h)


def _xywh_to_xysr(d: Detection) -> np.ndarray:
    s = max(d.w * d.h, 1e-6)
    r = d.w / max(d.h, 1e-6)
    return np.array([d.x, d.y, s, r], dtype=np.float64)


def _det_xyxy(d: Detection) -> np.ndarray:
    x1, y1, x2, y2 = d.as_xyxy()
    return np.array([x1, y1, x2, y2], dtype=np.float64)


def _xywh_to_xyxy(cx: float, cy: float, w: float, h: float) -> list[float]:
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


class _Track:
    def __init__(self, det: Detection, track_id: int) -> None:
        self.id = track_id
        self.kf = _Kalman(*_xywh_to_xysr(det))
        self.class_id = det.class_id
        self.class_name = det.class_name
        self.conf = det.confidence
        self.time_since_update = 0
        self.hits = 1
        self.age = 1
        self.history: list[tuple[float, float]] = [(det.x, det.y)]

    def predict(self) -> None:
        self.kf.predict()
        self.age += 1
        self.time_since_update += 1
        cx, cy, _, _ = self.kf.as_xywh()
        self.history.append((cx, cy))
        if len(self.history) > 30:
            self.history = self.history[-30:]

    def update(self, det: Detection) -> None:
        self.kf.update(_xywh_to_xysr(det))
        self.time_since_update = 0
        self.hits += 1
        self.class_id = det.class_id
        self.class_name = det.class_name
        self.conf = det.confidence
        cx, cy, _, _ = self.kf.as_xywh()
        if self.history:
            self.history[-1] = (cx, cy)
        else:
            self.history.append((cx, cy))


class SortTracker:
    def __init__(self, max_age: int = 30, min_hits: int = 3, iou_threshold: float = 0.3) -> None:
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks: list[_Track] = []
        self._next = 1
        self._frames = 0

    def update(self, detections: list[Detection]) -> list[Track]:
        self._frames += 1
        for t in self.tracks:
            t.predict()
        if len(self.tracks) == 0:
            matched_t, matched_d = [], []
        elif len(detections) == 0:
            matched_t, matched_d = [], []
        else:
            pred = np.array([_xywh_to_xyxy(*t.kf.as_xywh()) for t in self.tracks])
            dets = np.stack([_det_xyxy(d) for d in detections], axis=0)
            iou = _iou(pred, dets)
            cost = 1.0 - iou
            rows, cols = linear_sum_assignment(cost)
            matched_t, matched_d = [], []
            for r, c in zip(rows, cols):
                if iou[r, c] >= self.iou_threshold:
                    matched_t.append(int(r))
                    matched_d.append(int(c))
        unmatched_t = [i for i in range(len(self.tracks)) if i not in matched_t]
        unmatched_d = [i for i in range(len(detections)) if i not in matched_d]
        for ti, di in zip(matched_t, matched_d):
            self.tracks[ti].update(detections[di])
        for di in unmatched_d:
            self.tracks.append(_Track(detections[di], self._next))
            self._next += 1
        self.tracks = [t for i, t in enumerate(self.tracks) if not (i in unmatched_t and t.time_since_update > self.max_age)]
        out: list[Track] = []
        for t in self.tracks:
            if t.time_since_update > 0:
                continue
            # SORT: unconfirmed tracks only show during the stream's first min_hits frames.
            if t.hits < self.min_hits and self._frames > self.min_hits:
                continue
            cx, cy, w, h = t.kf.as_xywh()
            out.append(
                Track(
                    track_id=t.id,
                    class_id=t.class_id,
                    class_name=t.class_name,
                    x=cx,
                    y=cy,
                    w=w,
                    h=h,
                    confidence=t.conf,
                    age=t.age,
                    hits=t.hits,
                    history=list(t.history),
                )
            )
        return out
