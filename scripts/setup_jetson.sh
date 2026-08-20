#!/usr/bin/env bash
# Jetson / L4T packages. Run on the board, not on a Windows laptop.
# Follow https://docs.nvidia.com/jetson/index.html for the image that matches this module.
set -euo pipefail

if [[ "$(uname -m)" != "aarch64" ]]; then
  echo "This script is for Jetson (aarch64). This machine is $(uname -m)."
  exit 2
fi
if [[ ! -f /etc/nv_tegra_release ]]; then
  echo "No /etc/nv_tegra_release — this does not look like Jetson Linux."
  exit 2
fi

echo "L4T: $(head -n 1 /etc/nv_tegra_release)"
if command -v nvpmodel >/dev/null 2>&1; then
  nvpmodel -q || true
fi

sudo apt-get update
# Compute stack (CUDA, cuDNN, TensorRT, headers). No-op if already installed.
if ! dpkg -s nvidia-jetpack >/dev/null 2>&1; then
  sudo apt-get install -y nvidia-jetpack || echo "Install nvidia-jetpack from the JetPack repo for this L4T if apt cannot find it."
fi

sudo apt-get install -y \
  build-essential cmake git python3-pip python3-venv python3-yaml \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  gstreamer1.0-plugins-good gstreamer1.0-plugins-bad

# Prefer JetPack OpenCV (GStreamer/Argus). Ubuntu libopencv-dev can replace it.
if ! dpkg -s libopencv-dev >/dev/null 2>&1 && ! dpkg -s nvidia-opencv >/dev/null 2>&1; then
  sudo apt-get install -y libopencv-dev
fi

python3 - <<'PY'
import sys
print("python", sys.version)
try:
    import cv2
    print("cv2", cv2.__version__, "file", getattr(cv2, "__file__", "?"))
except Exception as exc:
    print("cv2 missing:", exc)
PY

echo
echo "Do not pip-install torch, opencv-python, onnxruntime-gpu, or tensorrt-cu13 from PyPI on this board."
echo "PyTorch: https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html"
echo "Then: python3 -m pip install --user numpy PyYAML matplotlib pytest psutil ultralytics onnx onnxslim"
echo
echo "TensorRT CLI: /usr/src/tensorrt/bin/trtexec"
echo "Build: cmake -S . -B build -DWITH_CUDA=ON -DWITH_TENSORRT=ON && cmake --build build -j"
echo "Before a published matrix: sudo nvpmodel -q  and  sudo jetson_clocks --show  (keep both fixed)."
