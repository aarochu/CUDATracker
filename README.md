# CUDATracker

Run a Jetson camera (or a video file) through detection and tracking, then prove why a frame took 33 ms: CPU vs CUDA preprocess, PyTorch/ONNX vs TensorRT, FP32 vs FP16, with a stage clock on every hop.

[SOW.md](SOW.md) is the construction spec. If this file and SOW disagree, SOW wins.

**Status:** pipeline code is in the tree. The live runner is `python -m cudatracker` (Windows/x86 and Jetson). The CMake `cudatracker` binary is OpenCV DNN ONNX, optional CUDA kernels, optional TensorRT (`-DWITH_TENSORRT=ON`), SORT, and `--bench` JSON/CSV. FPS numbers belong in `docs/results.md` only after `benchmarks/results/` has a run.

---

## Setup

```text
pip install -e .
python scripts/fetch_sample.py
python scripts/export_onnx.py --model yolov8n --imgsz 640
```

Jetson extra steps: [docs/jetson_setup.md](docs/jetson_setup.md). Weights: [models/README.md](models/README.md).

---

## Run

From the repo root (`PYTHONPATH=python` if you skipped `pip install -e .`):

```text
python -m cudatracker --source samples/vtest.avi --config configs/default.yaml
python -m cudatracker --source 0 --config configs/default.yaml
python -m cudatracker --source samples/vtest.avi --preprocess cpu  --backend pytorch --precision fp32
python -m cudatracker --source samples/vtest.avi --preprocess cuda --backend tensorrt --precision fp16
python -m cudatracker --source samples/vtest.avi --bench --warmup 20 --frames 100 --no-vis
```

Keys in the window: `q` quit, `h` hardware drawer, `t` trails, `d` HUD, `p` CPU/CUDA preprocess, `1`/`2` FP32/FP16 (reloads the backend).

C++ (after CMake): `./build/cudatracker --config configs/default.yaml --source samples/vtest.avi`

---

## Benchmark

Same clip, visualization off:

```text
python benchmarks/benchmark.py --config configs/benchmark.yaml
python benchmarks/plot.py --input benchmarks/results
```

`--quick` is a smoke pass (2+8 frames). Published tables should use the YAML 50/300 counts and a fixed `nvpmodel` / `jetson_clocks` on Jetson.

Matrix: preprocess `{cpu, cuda}` × backend `{pytorch, onnx, tensorrt}` × precision `{fp32, fp16}` where it exists × `640x480` / `1280x720` / `1920x1080`. Skipped cells stay in the CSV with a reason.

---

## CPU vs CUDA preprocess

CPU: OpenCV resize / cvtColor / CHW, reference tensor. CUDA: kernels in `cuda/` (resize, color, normalize, layout), launched from C++ (nvcc) or Python (NVRTC). Same letterbox ints; float CHW within `1/255`. After CUDA preprocess, infer should bind the device buffer (PyTorch tensor / ORT IO binding / TRT `set_tensor_address`).

`preprocess.fused: true` is the one-launch kernel. Leave it off until you time the four-kernel path.

---

## TensorRT

`python scripts/export_onnx.py` writes ONNX with NMS **outside** the graph. `python scripts/build_engine.py` calls `trtexec` when it exists. If it does not, the TensorRT backend uses ONNX Runtime's TensorRT EP and caches engines under `models/ort_trt_cache/`.

---

## Overlay

One video window. Thin bar: preprocess path, backend + precision, FPS, slowest stage. Hardware on `h`. Official benches use `--no-vis`.

---

## Tests

```text
python -m pytest tests -q
```

GPU preprocess comparison skips if CUDA is missing.

---

## Repo map

| Path | Role |
| --- | --- |
| `SOW.md` | Spec |
| `python/cudatracker/` | Live runner, backends, NVRTC preprocess |
| `src/` | C++ binary |
| `cuda/` | Project kernels |
| `configs/` | YAML including the matrix |
| `scripts/` | Export, engine, sample clip, Jetson packages |
| `benchmarks/` | Matrix runner and plots |
| `docs/` | Architecture, setup, bench protocol, results |
| `tests/` | Letterbox, NMS, SORT, CSV, CUDA closeness |

---

## Limits

Clocks and power mode change the table. PyTorch on Orin Nano may only be a 640 baseline. CSI strings differ by carrier. Power is `na` when the sensor is missing. This is latency engineering on a public YOLOv8n, not a COCO mAP paper.

---

## Later

Fused preprocess, overlap copy+infer, ByteTrack, a second model size: only after a timed CPU baseline exists in `benchmarks/results/`.
