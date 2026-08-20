from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class Analytics:
    frames: int = 0
    detections_total: int = 0
    frames_with_det: int = 0
    conf_sum: float = 0.0
    class_counts: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    peak_tracks: int = 0
    active_sum: int = 0
    track_last_age: dict[int, int] = field(default_factory=dict)

    def reset(self) -> None:
        self.frames = 0
        self.detections_total = 0
        self.frames_with_det = 0
        self.conf_sum = 0.0
        self.class_counts = defaultdict(int)
        self.peak_tracks = 0
        self.active_sum = 0
        self.track_last_age = {}

    def update(self, detections, tracks) -> None:
        self.frames += 1
        self.detections_total += len(detections)
        if detections:
            self.frames_with_det += 1
            self.conf_sum += sum(d.confidence for d in detections)
            for d in detections:
                self.class_counts[d.class_name or str(d.class_id)] += 1
        self.peak_tracks = max(self.peak_tracks, len(tracks))
        self.active_sum += len(tracks)
        for t in tracks:
            self.track_last_age[t.track_id] = t.age

    def summary(self) -> dict:
        mean_det = self.detections_total / max(self.frames, 1)
        mean_conf = self.conf_sum / max(self.detections_total, 1)
        det_rate = self.frames_with_det / max(self.frames, 1)
        ages = list(self.track_last_age.values())
        mean_age = sum(ages) / max(len(ages), 1)
        mean_active = self.active_sum / max(self.frames, 1)
        return {
            "mean_detections_per_frame": mean_det,
            "mean_confidence": mean_conf,
            "detection_frame_rate": det_rate,
            "peak_tracks": self.peak_tracks,
            "mean_active_tracks": mean_active,
            "mean_track_age_frames": mean_age,
            "class_counts": dict(self.class_counts),
        }
