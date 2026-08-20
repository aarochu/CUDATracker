# Jetson setup

1. Flash JetPack for the board you have (Orin Nano / NX / AGX, or Xavier).
2. `bash scripts/setup_jetson.sh`
3. Install the PyTorch wheel that matches this L4T from NVIDIA's Jetson zoo, then `pip install -r requirements.txt` (skip `onnxruntime-gpu` on aarch64 if you use the Jetson ORT/TensorRT packages instead).
4. Camera: USB via `--source 0`. CSI via a GStreamer string and `input.gstreamer: true`, for example `nvarguscamerasrc ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink`.
5. C++ binary:

```text
cmake -S . -B build -DWITH_CUDA=ON -DWITH_TENSORRT=ON
cmake --build build -j$(nproc)
./build/cudatracker --config configs/default.yaml --source samples/vtest.avi
```

6. Before a published matrix: pick an `nvpmodel` mode, optionally `sudo jetson_clocks`, and leave them alone for every cell. The result JSON records `nvpmodel` when the binary exists.

Windows/x86 laptops run `python -m cudatracker` (this repo's Python package). The CMake target wants OpenCV C++ and, on Windows, MSVC as nvcc's host compiler.
