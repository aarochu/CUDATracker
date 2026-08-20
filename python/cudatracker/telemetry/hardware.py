from __future__ import annotations

import json
import math
import os
import platform
import subprocess
import threading
import time
from pathlib import Path
from typing import Any


def _na() -> None:
    return None


class HardwareSampler:
    """1 Hz sampler. Missing sensors stay None and serialize as 'na'."""

    def __init__(self) -> None:
        self.latest: dict[str, Any] = {
            "cpu_util": None,
            "gpu_util": None,
            "ram_bytes": None,
            "gpu_mem_bytes": None,
            "gpu_temp_c": None,
            "cpu_temp_c": None,
            "power_w": None,
        }
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._nvml = None
        self._nvml_dev = None
        self._init_nvml()

    def _init_nvml(self) -> None:
        try:
            import pynvml

            pynvml.nvmlInit()
            self._nvml = pynvml
            self._nvml_dev = pynvml.nvmlDeviceGetHandleByIndex(0)
        except Exception:
            self._nvml = None

    def start(self) -> None:
        try:
            import psutil

            psutil.cpu_percent(interval=None)
        except Exception:
            pass
        self.latest = self.sample()
        self._thread = threading.Thread(target=self._loop, name="hw-telemetry", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2)
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:
                pass

    def _loop(self) -> None:
        while not self._stop.is_set():
            self.latest = self.sample()
            self._stop.wait(1.0)

    def sample(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "cpu_util": None,
            "gpu_util": None,
            "ram_bytes": None,
            "gpu_mem_bytes": None,
            "gpu_temp_c": None,
            "cpu_temp_c": None,
            "power_w": None,
        }
        try:
            import psutil

            out["cpu_util"] = float(psutil.cpu_percent(interval=None))
            out["ram_bytes"] = int(psutil.virtual_memory().used)
        except Exception:
            pass
        if self._nvml and self._nvml_dev is not None:
            n = self._nvml
            h = self._nvml_dev
            try:
                out["gpu_util"] = float(n.nvmlDeviceGetUtilizationRates(h).gpu)
            except Exception:
                pass
            try:
                mem = n.nvmlDeviceGetMemoryInfo(h)
                out["gpu_mem_bytes"] = int(mem.used)
            except Exception:
                pass
            try:
                out["gpu_temp_c"] = float(n.nvmlDeviceGetTemperature(h, n.NVML_TEMPERATURE_GPU))
            except Exception:
                pass
            try:
                out["power_w"] = float(n.nvmlDeviceGetPowerUsage(h)) / 1000.0
            except Exception:
                pass
        else:
            try:
                import torch

                if torch.cuda.is_available():
                    free, total = torch.cuda.mem_get_info()
                    out["gpu_mem_bytes"] = int(total - free)
            except Exception:
                pass
        jet = _jetson_sysfs()
        out.update({k: v for k, v in jet.items() if v is not None})
        return out


def _read_float(path: str) -> float | None:
    try:
        return float(Path(path).read_text().strip())
    except Exception:
        return None


def _jetson_sysfs() -> dict[str, Any]:
    out: dict[str, Any] = {}
    # thermal zones vary by board
    cpu_temps = []
    gpu_temps = []
    tz = Path("/sys/class/thermal")
    if tz.exists():
        for zone in sorted(tz.glob("thermal_zone*")):
            tname = ""
            try:
                tname = (zone / "type").read_text().strip().lower()
            except Exception:
                continue
            val = _read_float(str(zone / "temp"))
            if val is None:
                continue
            c = val / 1000.0 if val > 200 else val
            if "cpu" in tname or "cpu-thermal" in tname:
                cpu_temps.append(c)
            if "gpu" in tname or "gpu-thermal" in tname:
                gpu_temps.append(c)
        if cpu_temps:
            out["cpu_temp_c"] = sum(cpu_temps) / len(cpu_temps)
        if gpu_temps:
            out["gpu_temp_c"] = sum(gpu_temps) / len(gpu_temps)
    # INA / power — walk sysfs instead of a hardcoded board address.
    power = _jetson_power_w()
    if power is not None:
        out["power_w"] = power
    return out


def _jetson_power_w() -> float | None:
    roots = [
        Path("/sys/bus/i2c/drivers/ina3221x"),
        Path("/sys/devices/platform/ina3221x"),
        Path("/sys/bus/i2c/devices"),
        Path("/sys/class/hwmon"),
    ]
    seen: set[str] = set()
    rails: list[float] = []
    for root in roots:
        if not root.exists():
            continue
        for name in ("in_power*_input", "power1_input", "power*_input"):
            for p in root.rglob(name):
                key = str(p.resolve()) if p.exists() else str(p)
                if key in seen:
                    continue
                seen.add(key)
                val = _read_float(str(p))
                if val is None:
                    continue
                rails.append(val / 1000.0 if val > 20 else val)
    if rails:
        return float(sum(rails))
    return None


def environment_metadata() -> dict[str, Any]:
    meta: dict[str, Any] = {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "soc": None,
        "jetpack": None,
        "cuda": None,
        "tensorrt": None,
        "opencv": None,
        "nvpmodel": None,
        "jetson_clocks": None,
        "git": _git_hash(),
        "gpu_name": None,
    }
    try:
        import cv2

        meta["opencv"] = cv2.__version__
    except Exception:
        pass
    try:
        import torch

        meta["cuda"] = torch.version.cuda
        if torch.cuda.is_available():
            meta["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        pass
    try:
        import tensorrt as trt

        meta["tensorrt"] = trt.__version__
    except Exception:
        pass
    nv_tegra = Path("/etc/nv_tegra_release")
    if nv_tegra.exists():
        meta["jetpack"] = nv_tegra.read_text(errors="replace").strip().splitlines()[0]
    soc = Path("/proc/device-tree/model")
    if soc.exists():
        meta["soc"] = soc.read_text(errors="replace").replace("\x00", "").strip()
    nvp = _which_run(["nvpmodel", "-q"])
    if nvp:
        meta["nvpmodel"] = nvp.strip()
    # jetson_clocks has no standard "am I on" flag; record if the binary exists
    meta["jetson_clocks"] = bool(_which("jetson_clocks"))
    return meta


def _git_hash() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:
        return None


def _which(cmd: str) -> str | None:
    from shutil import which

    return which(cmd)


def _which_run(argv: list[str]) -> str | None:
    try:
        return subprocess.check_output(argv, stderr=subprocess.DEVNULL, text=True)
    except Exception:
        return None


def json_sanitize(value: Any) -> Any:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "na"
    if isinstance(value, dict):
        return {k: json_sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_sanitize(v) for v in value]
    return value
