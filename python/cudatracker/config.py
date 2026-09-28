from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml


@dataclass
class InputConfig:
    source: str = "samples/vtest.avi"
    resolution: str = "640x480"
    fps_cap: int = 0
    gstreamer: bool = False

    @property
    def width(self) -> int:
        return int(self.resolution.lower().split("x")[0])

    @property
    def height(self) -> int:
        return int(self.resolution.lower().split("x")[1])


@dataclass
class PreprocessConfig:
    backend: str = "cpu"
    letterbox: bool = True
    pad_value: int = 114
    fused: bool = False


@dataclass
class ModelConfig:
    name: str = "yolov8n"
    weights: str = "models/yolov8n.pt"
    onnx: str = "models/yolov8n.onnx"
    engine: str = "models/yolov8n_fp16.engine"
    engine_fp32: str = "models/yolov8n_fp32.engine"
    engine_fp16: str = "models/yolov8n_fp16.engine"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.45


@dataclass
class InferenceConfig:
    backend: str = "pytorch"
    precision: str = "fp32"
    device: str = "gpu"


@dataclass
class TrackerConfig:
    type: str = "sort"
    max_age: int = 30
    min_hits: int = 3
    iou_threshold: float = 0.3


@dataclass
class VisualizationConfig:
    enabled: bool = True
    trails: bool = True
    hud: bool = True
    hardware_panel: bool = False


@dataclass
class TelemetryConfig:
    hardware: bool = True


@dataclass
class BenchmarkConfig:
    warmup_frames: int = 50
    measure_frames: int = 300
    results_dir: str = "benchmarks/results"


@dataclass
class OutputConfig:
    video: str = ""
    metrics: str = ""


@dataclass
class MatrixConfig:
    preprocess: list[str] = field(default_factory=lambda: ["cpu", "cuda"])
    backend: list[str] = field(default_factory=lambda: ["pytorch", "onnx", "tensorrt"])
    precision: list[str] = field(default_factory=lambda: ["fp32", "fp16"])
    resolution: list[str] = field(default_factory=lambda: ["640x480", "1280x720", "1920x1080"])
    model: list[str] = field(default_factory=lambda: ["yolov8n"])


@dataclass
class AppConfig:
    input: InputConfig = field(default_factory=InputConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    tracker: TrackerConfig = field(default_factory=TrackerConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    telemetry: TelemetryConfig = field(default_factory=TelemetryConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    matrix: MatrixConfig = field(default_factory=MatrixConfig)

    def snapshot(self) -> dict[str, Any]:
        return asdict(self)


def _merge(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            dst[k] = _merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _from_dict(cls, data: dict[str, Any]):
    fields = {f.name for f in cls.__dataclass_fields__.values()}
    return cls(**{k: v for k, v in data.items() if k in fields})


def load_config(path: str | Path) -> AppConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    return AppConfig(
        input=_from_dict(InputConfig, raw.get("input") or {}),
        preprocess=_from_dict(PreprocessConfig, raw.get("preprocess") or {}),
        model=_from_dict(ModelConfig, raw.get("model") or {}),
        inference=_from_dict(InferenceConfig, raw.get("inference") or {}),
        tracker=_from_dict(TrackerConfig, raw.get("tracker") or {}),
        visualization=_from_dict(VisualizationConfig, raw.get("visualization") or {}),
        telemetry=_from_dict(TelemetryConfig, raw.get("telemetry") or {}),
        benchmark=_from_dict(BenchmarkConfig, raw.get("benchmark") or {}),
        output=_from_dict(OutputConfig, raw.get("output") or {}),
        matrix=_from_dict(MatrixConfig, raw.get("matrix") or {}),
    )


def apply_overrides(cfg: AppConfig, **kwargs: Any) -> AppConfig:
    if kwargs.get("source"):
        cfg.input.source = str(kwargs["source"])
    if kwargs.get("resolution"):
        cfg.input.resolution = str(kwargs["resolution"])
    if kwargs.get("preprocess"):
        cfg.preprocess.backend = str(kwargs["preprocess"]).lower()
    if kwargs.get("backend"):
        cfg.inference.backend = str(kwargs["backend"]).lower()
    if kwargs.get("precision"):
        cfg.inference.precision = str(kwargs["precision"]).lower()
    if kwargs.get("device"):
        cfg.inference.device = str(kwargs["device"]).lower()
    if kwargs.get("fused") is not None:
        cfg.preprocess.fused = bool(kwargs["fused"])
    if kwargs.get("no_vis"):
        cfg.visualization.enabled = False
    if kwargs.get("weights"):
        cfg.model.weights = str(kwargs["weights"])
    if kwargs.get("onnx"):
        cfg.model.onnx = str(kwargs["onnx"])
    if kwargs.get("engine"):
        cfg.model.engine = str(kwargs["engine"])
        if cfg.inference.precision == "fp16":
            cfg.model.engine_fp16 = cfg.model.engine
        else:
            cfg.model.engine_fp32 = cfg.model.engine
    if kwargs.get("imgsz"):
        cfg.model.imgsz = int(kwargs["imgsz"])
    if kwargs.get("output_video"):
        cfg.output.video = str(kwargs["output_video"])
    if kwargs.get("output_metrics"):
        cfg.output.metrics = str(kwargs["output_metrics"])
    return cfg
