#!/usr/bin/env bash
# Jetson / L4T packages. Run on the board, not on a Windows laptop.
set -euo pipefail
sudo apt-get update
sudo apt-get install -y \
  build-essential cmake git python3-pip python3-venv \
  libopencv-dev libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  python3-yaml
python3 -m pip install --user -r requirements.txt || true
echo "Install a JetPack PyTorch wheel for this L4T version, then:"
echo "  cmake -S . -B build -DWITH_CUDA=ON -DWITH_TENSORRT=ON"
echo "  cmake --build build -j"
echo "Record nvpmodel and jetson_clocks in every published bench."
