# CUDATracker

Run a Jetson camera (or a video file) through detection and tracking, then prove why a frame took 33 ms: CPU vs CUDA preprocess, PyTorch/ONNX vs TensorRT, FP32 vs FP16, with a stage clock on every hop.

[SOW.md](SOW.md) is the construction spec. If this file and SOW disagree, SOW wins.

**Status:** Python live runner works on Windows/x86 and Jetson. Native TensorRT engines are `python scripts/build_engine.py` (Python TensorRT API if `trtexec` is missing). CMake `cudatracker` is OpenCV DNN ONNX + optional CUDA kernels + optional TensorRT. Published FPS live in `docs/results.md` after `benchmarks/results/matrix.csv`.

---

## Setup

```text
pip install -r requirements.txt
pip install -e .
python scripts/fetch_sample.py
python scripts/export_onnx.py --model yolov8n --imgsz 640
```

Jetson extra steps (NVIDIA docs mapped onto this repo): [docs/jetson_setup.md](docs/jetson_setup.md). Weights: [models/README.md](models/README.md).

The requirements file is for Windows/x86. On Jetson, follow the board-specific package steps in the Jetson guide and set `PYTHONPATH=python` when running from the repo root.
The optional Jetson Docker image uses the base image's OpenCV and installs only the Python core. Mount an ONNX model and a video source at the configured paths before running its default benchmark command.

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
With `--no-vis` outside bench mode, a video file is processed once and the command exits at end of file. Bench mode loops short clips to fill its requested frame count.

C++ (after CMake). Jetson: `cmake -S . -B build -DWITH_CUDA=ON -DWITH_TENSORRT=ON`. Windows: OpenCV official pack is vc16 — point CMake at `.../opencv/build/x64/vc16/lib` if `find_package(OpenCV)` rejects VS 2026, put that `bin` on PATH, and use `-DWITH_TENSORRT=OFF` unless you have `NvInfer.h`.

```text
./build/cudatracker --config configs/default.yaml --source samples/vtest.avi
# Windows Release:
# build\Release\cudatracker.exe --config configs/default.yaml --source samples/vtest.avi --backend onnx --bench
```

---

## Benchmark

Same clip, visualization off:

```text
python benchmarks/benchmark.py --config configs/benchmark.yaml
python benchmarks/plot.py --input benchmarks/results
```

`--quick` is a smoke pass (2+8 frames). Published tables use the YAML 50/300 counts on one Jetson, one `nvpmodel` mode, and a recorded `jetson_clocks --show`. See [docs/jetson_setup.md](docs/jetson_setup.md).

Matrix: preprocess `{cpu, cuda}` × backend `{pytorch, onnx, tensorrt}` × precision `{fp32, fp16}` where it exists × `640x480` / `1280x720` / `1920x1080`. Skipped cells stay in the CSV with a reason.

---

## CPU vs CUDA preprocess

CPU: OpenCV resize / cvtColor / CHW, reference tensor. CUDA: kernels in `cuda/` (resize, color, normalize, layout), launched from C++ (nvcc) or Python (NVRTC). Same letterbox ints; float CHW within `1/255`. After CUDA preprocess, infer should bind the device buffer (PyTorch tensor / ORT IO binding / TRT `set_tensor_address`).

`preprocess.fused: true` is the one-launch kernel. Leave it off until you time the four-kernel path.

---

## TensorRT

`python scripts/export_onnx.py` writes ONNX with NMS **outside** the graph. `python scripts/build_engine.py` calls `trtexec` (`/usr/src/tensorrt/bin/trtexec` on Jetson after JetPack). On x86, if `trtexec` is missing the TensorRT backend can use ORT's TensorRT EP when `nvinfer` is installed — not a substitute for JetPack TensorRT on the board.

---

## Overlay

One video window. Thin bar: preprocess path, backend + precision, FPS, slowest stage. Hardware on `h`. Official benches use `--no-vis`.

---

## Tests

```text
python -m pytest tests -q
```

GPU preprocess comparison skips if CUDA is missing.

CI (`.github/workflows/ci.yml`) runs on every PR and push to `main`: ruff, pytest on Linux (Python 3.9 / 3.12 / 3.14) and Windows, the C++ CPU build, and the CUDA kernels plus TensorRT backend compiled against CUDA 12.9 and 13.4 with TensorRT headers. Hosted runners have no GPU, so GPU tests skip there.

Release: push a tag (`git tag v0.2.0 && git push origin v0.2.0`). `.github/workflows/release.yml` reruns CI, builds the Linux x86_64 binary with CUDA 13.4 and TensorRT, and publishes a GitHub release with it attached.

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

Clocks and `nvpmodel` change the table. PyTorch on Orin Nano may only be a 640 baseline. CSI pipelines need Argus + NVMM and differ by sensor. Power is `na` when `tegrastats` / INA are missing. This is latency engineering on a public YOLOv8n, not a COCO mAP paper.

---

## Later

Fused preprocess, overlap copy+infer, ByteTrack, a second model size: only after a timed CPU baseline exists in `benchmarks/results/`. That baseline is in the laptop matrix; fused is still unmeasured.
