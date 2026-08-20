from __future__ import annotations

from cudatracker.config import AppConfig, apply_overrides, load_config
from cudatracker.inference.base import BackendNotAvailable
from cudatracker.inference.factory import create_backend


def test_yaml_loads():
    cfg = load_config("configs/default.yaml")
    assert cfg.inference.backend == "pytorch"
    assert cfg.preprocess.backend == "cpu"
    assert cfg.model.imgsz == 640


def test_unknown_backend_rejected():
    cfg = AppConfig()
    cfg.inference.backend = "magic"
    try:
        create_backend(cfg)
        raise AssertionError("expected BackendNotAvailable")
    except BackendNotAvailable:
        pass


def test_fused_none_keeps_yaml():
    cfg = load_config("configs/default.yaml")
    cfg.preprocess.fused = True
    apply_overrides(cfg, fused=None)
    assert cfg.preprocess.fused is True
    apply_overrides(cfg, fused=False)
    assert cfg.preprocess.fused is False
