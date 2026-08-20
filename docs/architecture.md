# Architecture

CUDATracker is a staged pipeline: capture → preprocess (CPU OpenCV or project CUDA kernels) → infer → YOLO decode/NMS → SORT → overlay → timers/telemetry.

## Two runtimes

`python -m cudatracker` is the complete runner (PyTorch, ONNX Runtime, TensorRT EP or native engine, NVRTC-launched kernels from `cuda/*.cu`, HUD, matrix). Use it on Windows and on Jetson until the C++ binary is built.

`cudatracker` from CMake is the C++ path: OpenCV capture, CPU or CUDA preprocess, OpenCV DNN ONNX, optional native TensorRT (`-DWITH_TENSORRT=ON`) bound to the CUDA preprocess device buffer, SORT (Kalman + Hungarian), HUD, and `--bench` JSON/CSV. On Jetson, TensorRT is the JetPack library (`/usr/src/tensorrt/bin/trtexec`, `libnvinfer`). On this Windows box, engines come from `scripts/build_engine.py` via the pip TensorRT 11 API. nvcc on CUDA 13 emits Orin `sm_87` plus desktop SMs through `sm_120`; Xavier `sm_72` is only compiled on CUDA < 13.

Interfaces in Python: `Capture`, `CudaPreprocessor` / `cpu_preprocess`, `InferBackend` (`pytorch` / `onnx` / `tensorrt`), `decode_yolo`, `SortTracker`. C++ mirrors capture, preprocess, `IInferBackend`, `SortTracker`.

## CUDA

Kernels live in `cuda/`. Phase 6 launches four of them (resize, BGR→RGB, normalize, HWC→CHW). `preprocess.fused: true` switches on the Phase 10 fused kernel. Python compiles that source with NVRTC (no MSVC required). CMake compiles the same files with nvcc.

Preprocess output is a `1×3×imgsz×imgsz` float32 CHW tensor. CUDA path writes it in device memory; PyTorch/ORT/TRT bind that pointer when the backend allows it.

## Timing

The metric that matters is **end-to-end wall time for the frame**, not kernel time and not inference time alone.

Each frame is timed as a sequential breakdown (sync at stage boundaries on purpose, so work cannot hide in the next stage):

```text
Frame 1842
Capture              1.8 ms
Preprocess           3.7 ms   gpu 3.1
H->D transfer        0.4 ms   gpu 0.4
Inference           14.2 ms   gpu 13.8
NMS                  0.8 ms
Tracking             1.1 ms
Visualization        2.0 ms
----------------------------
TOTAL               24.0 ms
cpu 8.1  gpu 17.3  wait 5.2  bottleneck inference
```

- **wall** (`*_ms`): host clock, including waiting for the GPU
- **gpu** (`*_gpu_ms`): CUDA events, device work only
- **wait** (`*_sync_ms`): time blocked in `synchronize`
- **cpu**: wall minus wait

Official benches turn visualization off so overlay time does not pollute the table. FPS is `measure_frames / sum(end_to_end)`.

