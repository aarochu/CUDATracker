# Optimization

Rule: a row in this file needs a before/after in `benchmarks/results/`. Numbers below are the **laptop** matrix on 2026-08-20 (`matrix.csv`, RTX 5060, 640×480 unless noted). Not a Jetson table.

## What we wrote vs what we call

Custom CUDA (project kernels in `cuda/`): bilinear letterbox resize, BGR→RGB, uint8→float /255, HWC→CHW. Optional fused kernel (`preprocess.fused`) was **off** for this matrix.

Not custom: detector GEMM/conv (cuDNN / TensorRT), NMS (`cv2.dnn.NMSBoxes`), SORT (CPU). OpenCV CPU `resize` / `cvtColor` is the preprocess reference.

## Runs

### 1. CUDA preprocess (same TensorRT FP16 backend)

Trace: after TensorRT FP16, CPU preprocess was the bottleneck (`preprocess` 6.6 ms, infer 3.0 ms, 60.4 FPS).

| | preprocess | FPS | e2e mean | bottleneck |
| --- | --- | --- | --- | --- |
| before | cpu | 60.4 | 16.6 ms | preprocess |
| after | cuda | 77.1 | 13.0 ms | postprocess |

Preprocess wall dropped from 6.6 ms to 0.67 ms. End-to-end moved. NMS/postprocess is now the largest remaining stage on that cell (~3.5 ms).

Same swap on **PyTorch FP32** did not win: cpu 46.2 FPS vs cuda 45.1 FPS. Inference stayed ~7–13 ms; four kernel launches plus H2D of the camera frame did not pay for themselves. Kept as data.

### 2. TensorRT vs PyTorch (CPU preprocess, FP32)

| | backend | FPS | e2e mean | infer mean |
| --- | --- | --- | --- | --- |
| before | pytorch | 46.2 | 21.7 ms | 7.5 ms |
| after | tensorrt | 55.6 | 18.0 ms | 4.3 ms |

Bottleneck moved off inference onto preprocess. That is why step 1 used TRT FP16, not PyTorch.

### 3. FP16 vs FP32 TensorRT (CPU preprocess)

TRT 11 builds FP16 from a typed ONNX (no `BuilderFlag.FP16`). Boxes still appeared on `vtest.avi` (mean detections/frame 8.9 both cells).

| | precision | FPS | e2e mean | infer mean |
| --- | --- | --- | --- | --- |
| before | fp32 | 55.6 | 18.0 ms | 4.3 ms |
| after | fp16 | 60.4 | 16.6 ms | 3.0 ms |

### 4. HUD off

Official matrix already has `visualization.enabled: false`. Render mean is 0.0 ms in every live row. No paired HUD-on row in this CSV.

## Not timed yet

Fused preprocess (`preprocess.fused: true`), overlap copy+infer, C++ TensorRT (`-DWITH_TENSORRT=ON` needs `NvInfer.h` from JetPack, not the pip wheel).
