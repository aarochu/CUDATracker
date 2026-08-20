# Results

These numbers are from `python benchmarks/benchmark.py --config configs/benchmark.yaml` on **this laptop**, not a Jetson. Official Jetson tables still need the same command on the board with a fixed `nvpmodel` and `jetson_clocks --show` ([docs/jetson_setup.md](jetson_setup.md)). Do not mix this table with a Jetson run.

- Host: Jarvis, Windows 11, Python 3.14.5
- GPU: NVIDIA GeForce RTX 5060 Laptop GPU, CUDA 13.2 (PyTorch), TensorRT 11.2.1.2
- Clip: `samples/vtest.avi` (SHA-256 `45cddc94…10516cf`), YOLOv8n, `imgsz` 640, conf 0.25, iou 0.45, visualization off
- Protocol: 50 warmup / **300** measure frames, looped source
- Engines: `models/yolov8n_fp32.engine`, `models/yolov8n_fp16.engine` (Python TensorRT API; TRT 11 has no `BuilderFlag.FP16`, FP16 ONNX is typed then built)
- Skipped: ONNX × fp16 (FP32 graph). CSV keeps those rows.

Plots: `benchmarks/plots/fps.png`, `latency.png`, `speedup.png`, `fps_vs_resolution.png`, `cpu.png`, `gpu.png`, `power_vs_fps.png`.

## 640×480 (primary cell)

| preprocess | backend | precision | FPS | e2e mean | bottleneck |
| --- | --- | --- | --- | --- | --- |
| cpu | pytorch | fp32 | 46.2 | 21.7 ms | inference |
| cpu | pytorch | fp16 | 40.2 | 24.9 ms | inference |
| cpu | onnx | fp32 | 44.4 | 22.5 ms | inference |
| cpu | tensorrt | fp32 | 55.6 | 18.0 ms | preprocess |
| cpu | tensorrt | fp16 | 60.4 | 16.6 ms | preprocess |
| cuda | pytorch | fp32 | 45.1 | 22.2 ms | inference |
| cuda | onnx | fp32 | 52.3 | 19.1 ms | inference |
| cuda | tensorrt | fp32 | 61.4 | 16.3 ms | inference |
| cuda | tensorrt | fp16 | 77.1 | 13.0 ms | postprocess |

TensorRT FP16 + CUDA preprocess is the fastest cell here. CUDA preprocess did **not** help PyTorch FP32 at 640 (46.2 → 45.1 FPS); inference still owned that frame. Full CSV: `benchmarks/results/matrix.csv`.

## C++ binary (not in the Python matrix)

`build/Release/cudatracker.exe` on the same clip, OpenCV 4.11 Windows pack (no CUDA DNN — falls back to CPU), `--backend onnx --preprocess cpu --bench`: **12.9 FPS**, e2e ~78 ms, infer ~68 ms. Use `python -m cudatracker` for the TensorRT numbers above.

A Jetson 50/300 with recorded `nvpmodel` is still unmeasured.
