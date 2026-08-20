# Optimization

Rule: a row in this file needs a before/after in `benchmarks/results/`. No file, no claim.

## What we wrote vs what we call

Custom CUDA (project kernels in `cuda/`): bilinear letterbox resize, BGR→RGB, uint8→float /255, HWC→CHW. Optional fused kernel that does all four in one launch.

Not custom: detector GEMM/conv (cuDNN / TensorRT), NMS (`cv2.dnn.NMSBoxes` or an in-engine plugin if you export that way), SORT association (CPU). OpenCV CPU `resize` / `cvtColor` is the preprocess reference.

## Hypotheses to test, in order

1. **Infer dominates at 640.** CUDA preprocess may lose or tie because four extra kernel launches plus H2D of the camera frame cost more than OpenCV on a 640×480 frame. Write that down if it happens.
2. **Fused kernel.** If preprocess time is launch-bound, `preprocess.fused: true` should cut it. Compare four-kernel CUDA vs fused vs CPU on the same backend.
3. **FP16 TensorRT.** If FP32 TRT infer is the bulk of e2e, FP16 should move FPS. Check boxes still look right on `vtest.avi`.
4. **Zero-copy bindings.** CUDA preprocess already writes a device CHW buffer. PyTorch uses that tensor in-place. ORT/TRT use IO binding when the tensor is on CUDA. CPU preprocess still pays H2D (timed as transfer).
5. **HUD off.** Official matrix already disables drawing.

Until those runs exist, this page is a plan, not a result list.
