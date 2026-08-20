# CUDATracker SOW

Agents and humans building this repo treat this file as law. If README, comments, or a chat prompt disagree with it, this file wins until someone edits it on purpose.

CUDATracker is a Jetson pipeline that captures frames, detects objects, keeps IDs on them, and times every stage so you can say where a millisecond went. Detection is the workload. The actual work is performance engineering: CPU vs CUDA preprocess, PyTorch/ONNX vs TensorRT, FP32 vs FP16, and memory copies you can point at in a profiler.

A webcam overlay with Ultralytics defaults and a verbal "it hits 30 FPS" fails this spec. So does inventing a speedup table before `benchmarks/results/` has a file.

The question you keep answering, with numbers:

Where does frame time go, and which change moved it?

The story the repo has to make tellable in an interview is linear: measure a baseline, name a bottleneck, change one thing, measure again. Skip the hop from idea to fused kernel.

---

## Build order

Ship a boring correct path before CUDA. Phases are gates. A later phase may rewrite earlier code; it may not replace a missing baseline.

**1. Capture.** USB camera or a video file through OpenCV (GStreamer for CSI on Jetson). Print size and FPS. File input stays forever so you can work without a lens.

**2. Baseline detect.** CPU preprocess, then PyTorch or ONNX Runtime, then boxes on the frame. Log a crude FPS even if the timer is ugly.

**3. Track.** SORT: Kalman + IoU/Hungarian, monotonic IDs, drop after `max_age`. IDs should survive a person walking across a clip. If they reassign every frame, stop; nothing later fixes that.

**4. ONNX → TensorRT.** One export script, one engine-build command, boxes that still look right.

**5. FP32 and FP16 engines.** Precision is a config flag. FP16 may shift boxes a little. Collapse (empty frames, garbage classes) is a bug.

**6. CUDA preprocess.** Project kernels for resize, BGR→RGB, normalize, HWC→CHW. Equivalence tests vs CPU. Run end-to-end with `--preprocess cuda`. If it is slower at 640×480 because inference is 18 ms and resize was 1 ms, keep the result. That is data.

**7. Stage clocks.** capture, preprocess, transfer, inference, postprocess, tracking, render, end-to-end. CUDA events for GPU work. Steady clock for CPU. Mean, p50, p95, p99.

**8. Board telemetry.** CPU%, GPU%, RAM, GPU mem, temps, power if INA/`tegrastats` exist. Missing sysfs → write `na`, don't crash. Sample off the hot path (~1 Hz).

**9. Matrix runner.** One command walks configs, writes JSON + CSV.

**10. Change only what the trace named.** Candidates: CUDA preprocess, FP16, overlap copy+infer, bind TensorRT to the preprocess device buffer, fused kernel, HUD off during official runs. Each entry in `docs/optimization.md` needs a before and after.

**11. Plots and overlay.** Charts from CSV. Live HUD that shows the feed and the bottleneck, not forty stats.

**12. Docs freeze.** Setup, architecture, how to bench, what you changed, real results. README commands work from a clean clone plus model export.

---

## What you are not building

A new detector trained from scratch. Multi-camera fusion. ReID / DeepSORT. Cloud infer. A custom GEMM. A rewrite of EfficientNMS or cuDNN convs. Speedups measured on synthetic noise that looks nothing like a camera frame.

Custom CUDA exists for work we own and can explain in `docs/optimization.md`. If OpenCV CUDA or NPP already does a step well, you may call it, and you write that down as library vs kernel.

---

## Machines

Primary: NVIDIA Jetson (Orin Nano, Orin NX, AGX Orin; Xavier if that is what is on the desk). Do not hardcode one SoC.

Secondary: x86 with an NVIDIA GPU for export, kernel tests, plots. Jetson-only bits (`nvarguscamerasrc`, `nvpmodel`, some thermal zones, INA power) stub or skip with a log line.

Inputs, in this order: USB (`/dev/video0` or index `0`); CSI via a GStreamer string; a `.mp4` / `.avi` / image sequence (required for CI); a folder of stills for unit tests.

Expected stack, versions recorded from the board that produced published numbers, not baked into C++:

JetPack CUDA + cuDNN + TensorRT, OpenCV with GStreamer (CUDA modules if present), Python 3.8–3.12, a Jetson PyTorch wheel or an ONNX-only baseline if libtorch is too heavy, ONNX Runtime (CUDA EP, CPU EP fallback), CMake 3.18+, g++ / nvcc.

Fairness: a published comparison records `nvpmodel`, whether `jetson_clocks` ran, SoC, JetPack, CUDA, TensorRT. Mixing 15W and MAXN in one table is an invalid run.

---

## Pipeline

```text
camera or file
    → capture (OpenCV / GStreamer)
    → preprocess (CPU OpenCV  |  CUDA kernels)
    → infer (PyTorch | ORT | TensorRT FP32 | TensorRT FP16)
    → decode boxes / NMS / map back to frame pixels
    → SORT
    → overlay + counters
    → stage + hardware telemetry
    → JSON/CSV
    → plots / writeup
```

Each box is a module behind an interface. `main` wires them; it does not contain the algorithm.

---

## Languages

Hot path in C++/CUDA: capture, preprocess, TensorRT, tracker, overlay, live HUD. Kernels live in `cuda/*.cu`.

Python for export (`scripts/export_onnx.py`), engine build wrapping `trtexec` or a builder, the matrix runner, plots.

A Python OpenCV loop is allowed as a Phase 2 prototype. It is not the thing you show as the optimized path. If Jetson Python TensorRT is a stopgap, label it that way in docs.

Sensible split: `cudatracker` binary for live + per-frame timers; `benchmarks/benchmark.py` launches it across the matrix.

---

## Layout

Keep the roles even if a filename moves.

```text
CUDATracker/
├── SOW.md
├── README.md
├── LICENSE                 # MIT unless we say otherwise
├── CMakeLists.txt
├── requirements.txt
├── configs/                # default, cpu, cuda, tensorrt, benchmark
├── models/                 # README + .gitkeep; no fat binaries in git
├── src/                    # main, capture, preprocessing, inference, detection,
│                           # tracking, telemetry, visualization, app
├── cuda/                   # resize, colorspace, normalize, layout, kernels.cuh
├── benchmarks/             # benchmark.py, plot.py, results/, plots/
├── tests/
├── scripts/                # export_onnx.py, build_engine.py, benchmark.sh, setup_jetson.sh
└── docs/                   # architecture, benchmarking, optimization, jetson_setup, results
```

gitignore engines, large onnx, and raw result dumps. `models/README.md` says how to fetch/export. Attribute Ultralytics / COCO licenses there.

---

## Config

YAML owns the pipeline. CLI may override. No hidden defaults that only exist in a `.cpp` literal for backend, precision, or source.

| Key | Meaning |
| --- | --- |
| `input.source` | `0`, `/dev/video0`, CSI pipeline, `clip.mp4` |
| `input.resolution` | `640x480`, `1280x720`, `1920x1080` |
| `input.fps_cap` | optional |
| `preprocess.backend` | `cpu` \| `cuda` |
| `preprocess.letterbox` | bool; must match export |
| `model.name` / `model.weights` / `model.imgsz` | e.g. yolov8n, 640 |
| `inference.backend` | `pytorch` \| `onnx` \| `tensorrt` |
| `inference.precision` | `fp32` \| `fp16` |
| `inference.device` | `gpu` \| `cpu` |
| `tracker.type` | `sort` first; ByteTrack later if you want |
| `tracker.max_age`, `min_hits`, `iou_threshold` | numbers |
| `visualization.enabled` | off for official matrix |
| `telemetry.hardware` | bool |
| `benchmark.warmup_frames` / `measure_frames` | 50 / 300+ as defaults |
| `output.video` / `output.metrics` | optional paths |

`configs/benchmark.yaml` lists the matrix cells.

Suggested CLI:

```text
cudatracker --config configs/default.yaml
cudatracker --source video.mp4 --preprocess cuda --backend tensorrt --precision fp16
cudatracker --bench --config configs/benchmark.yaml
```

---

## Modules

### Capture

BGR (or a documented native format) plus `timestamp_ns` and `frame_index`. OpenCV for USB and files; GStreamer string for CSI. Fail with a message a human can act on.

Phase 1–2 may copy to host. A later optimization may keep NVMM/`GpuMat` and skip the round trip; don't lock the interface into "must be a `cv::Mat`" if you can avoid it.

### Preprocess

Camera frame → network tensor. For a typical YOLO export: letterbox or stretch to `imgsz`, BGR→RGB, float, divide by 255 (or whatever the graph used), HWC→CHW, batch, leave the buffer where the backend wants it.

CPU path is OpenCV and is the numerical reference.

CUDA path is kernels we wrote. Phase 6 ships them separable so tests can blame one op. A fused resize+cvt+norm+CHW kernel is Phase 10, after a trace says launch overhead or memory traffic is the problem.

Letterbox scale, pad, and pad color must match on both paths and get handed to box mapping. Integer geometry matches exactly.

Pinned host memory when you still memcpy. After CUDA preprocess, do not D2H then H2D into TensorRT; bind the engine input to that device buffer.

### Infer

Common `load` / `infer` interface. Fixed shape `1×3×imgsz×imgsz` for v1 engines. Dynamic shape is later.

PyTorch is the baseline (libtorch in-process if you can; a documented Python subprocess only if you can't). ONNX Runtime next. TensorRT FP32 and FP16 from the same ONNX, FP16 via `BuilderFlag::kFP16`. Workspace size, ONNX path, and precision go into the engine-build log.

### Postprocess

`Detection`: `x,y,w,h` in original frame pixels, confidence, class id.

NMS thresholds come from config and stay shared across backends so the comparison is fair. If you export EfficientNMS inside the engine, say so: TRT postprocess time will look better because the work moved. Map letterbox space back to the frame. A one-pixel systematic bias here wrecks tracking.

### Tracker

SORT on CPU. Persistent ints, short centroid history for a trail, no ID reuse in a session. ByteTrack is optional later. DeepSORT/ReID is out until someone proves appearance is the failure mode; it would confound infer benches.

### Analytics

Running class counts, detections/sec, live track count, mean track age. A tripwire / ROI entry-exit is useful after tracking works; it is not a Phase 3 blocker.

### Overlay (how it should feel)

The frame is the product. Metrics are a thin layer on top of it, like broadcast chrome: always in the same corner, never a second "dashboard app" competing with the video.

Purpose of the window: see boxes + IDs, and see which stage is expensive. If a number doesn't help those two jobs, it belongs behind a key (hardware drawer) or only in JSON.

Hierarchy: video and tracks first; current config + FPS + the slowest stage second; CPU/GPU/temp/power third. Don't paint twenty labels at full weight. The FPS figure can be large; "GR3D 87%" should not shout.

Labels name contents, not departments. `TRT FP16  14.8 ms` beats `INFERENCE MODULE`. `CUDA preprocess` beats `PIPELINE STATUS: ACCELERATED`.

Wayfinding: the overlay always shows backend, precision, and preprocess path so you know which cell you're in. Switching CPU↔CUDA or FP32↔FP16 should be a key or a config reload. Prefer toggling without restarting capture. If the engine must reload, keep the window up and show a short status ("building engine") instead of freezing with no feedback. Instant press feedback; don't wait for the next second of telemetry.

Agency: HUD on/off, trails on/off, hardware panel on/off. Official benches run with drawing off so render time doesn't pollute the table. Destructive actions (overwrite an engine, wipe results) need a confirm; toggling FP16 does not.

Motion: numbers that change every frame (FPS, stage ms) may update continuously. Temps and power at ~1 Hz so they don't strobe. No rainbow boxes, no emoji, no fake "AI" glow. Class colors stay stable. Trails are short and the same stroke as the box.

Type: platform UI font. Tight leading on the FPS, slightly more on the breakdown. Weight for hierarchy, not a new hue per row. Darken the strip enough that white text holds on a bright wall; a dim translucent bar is enough, stacking two glass panels is not.

Warnings earn color: dropped frames, thermal throttle, missing engine. Green-for-good on every row trains people to ignore it.

Reduced clutter is a feature. Headless mode is the same pipeline with the window gone.

Config switch in a live demo is allowed to restart the infer backend. Capture and tracker should survive if you can keep them.

### Profiler and hardware

The number that matters is end-to-end wall time for the frame, not kernel time and not inference time alone.

Every frame records this sequential breakdown (sync at stage boundaries so GPU work cannot hide in the next stage):

capture, preprocess, H→D transfer, inference, NMS, tracking, visualization, TOTAL.

For each stage, store CPU wall, CUDA-event GPU time, and time blocked in synchronize. Sum those to cpu / gpu / wait for the frame and name the bottleneck as the stage with the largest wall.

FPS = measured frames / sum(end_to_end) after warmup. Visualization off for official benches.

Hardware sources: `/proc/stat`, `cudaMemGetInfo`, thermal zones, `tegrastats` GR3D and `VDD_*` on Jetson, NVML on a dGPU laptop. A background thread may parse `tegrastats`; the infer thread should not.

### Logs on disk

Each run: JSON (full) + one CSV row.

JSON includes git hash, hostname, SoC, JetPack, CUDA, TensorRT, OpenCV, nvpmodel, jetson_clocks, the config snapshot, warmup/measure counts, every metric below, input path, and a content hash if the source is a file.

Derived speedups (CPU→CUDA, PT→TRT, FP32→FP16) are computed from those files. Do not type them into the HUD by hand.

---

## CUDA rules

Kernels we owe: bilinear resize **or** a fused preprocess that includes resize; BGR→RGB; normalize; HWC→CHW. Not a CuPy one-liner sold as the CUDA story.

Document launch geometry (16×16 / 32×32, etc.). Check every `cudaError_t` and TRT status.

Leave GEMM, conv, detector softmax, and v1 Hungarian on GPU alone.

---

## Model

Start with a small COCO detector that can actually run on Jetson: YOLOv8n, YOLOv5n, YOLOX-Nano, or equivalent. Optional second cell: the small vs the `s` variant of the same family.

`scripts/export_onnx.py` is the supported export. Record opset (12 or 17 are common), whether NMS is in-graph, and `imgsz`. No custom-trained net required. Still watch a known clip so boxes aren't nonsense.

---

## CPU vs CUDA match

| Check | Bar |
| --- | --- |
| Float CHW vs CPU reference | abs `1/255` or rel `1e-3`, pick one in tests and keep it |
| Letterbox scale/pad | exact ints |
| Same backend, CUDA vs CPU preprocess, high-conf boxes | IoU ≥ 0.99 **or** center error < 1 px on 640; document which |

FP16 vs FP32 output tensors will not match bitwise. Compare boxes, not bits. Keep preprocess buffers FP32 unless a later experiment times FP16 preprocess.

---

## What to record

Timing: FPS; e2e mean/p50/p95/p99; preprocess, transfer, infer, postprocess, tracking, render.

Board: CPU%, GPU%, RAM, GPU mem, GPU temp, CPU temp, power when the sensor exists.

Scene: mean detections/frame, mean accepted conf, fraction of frames with ≥1 box, live ID count, mean track duration.

Paired runs: CPU vs CUDA preprocess (e2e and preprocess-only); PyTorch vs TRT; FP32 vs FP16; FPS vs resolution.

---

## Matrix

Same video file, same clocks, same conf/iou for a published table. Vis off.

Factors: preprocess `{cpu, cuda}`; backend `{pytorch, onnx, tensorrt}`; precision `{fp32, fp16}` where the backend has it; capture size `{640x480, 1280x720, 1920x1080}`; model small, optionally medium.

Skip illegal cells (PyTorch FP16 if you never implemented it) and list them as skipped.

Protocol: metadata → warmup (default 50) → measure (default 300+) → JSON + CSV.

```text
python benchmarks/benchmark.py --config configs/benchmark.yaml
# or ./scripts/benchmark.sh
```

Plots from that CSV: FPS by cell; latency mean+p95; CPU%; GPU%; speedup bars; power vs FPS if power exists; FPS vs resolution. `docs/results.md` is prose next to those plots. Empty until you have runs. Invented tables are a spec violation.

---

## Tests

Preprocess: CPU vs CUDA closeness; letterbox; a weird size (odd width).

Detection: NMS on a tiny synthetic tensor; mapping invertible.

Tracker: two crossing boxes keep two IDs; `max_age` kills a track.

Config: YAML loads; unknown backend exits non-zero.

Bench harness: CSV appears; skips recorded.

GitHub CI can be CPU-only. GPU tests run on the box with a GPU.

---

## Failure behavior

Logs with time and stage (`INFO` / `WARN` / `ERROR`). Missing weights, missing engine, OOM, dead camera: non-zero exit, sentence a human can use. No `assert` for recoverable live errors. TRT engine built with the wrong version: detect and tell them to rebuild.

---

## Sixty-second demo

File or camera. Boxes and IDs. HUD with stage times. Flip CPU preprocess → CUDA; read FPS and e2e. Flip TRT FP32 → FP16. Quote the overall gain vs baseline from `docs/results.md`, which came from the matrix, not from memory.

If CUDA lost, say so and point at the breakdown (often: infer still owns the frame). Hiding a slowdown is worse than a slow kernel.

---

## README vs this file

README is how a stranger runs it and what the last measured table said. This SOW is how you construct it. README must not invent FPS. Until a real run exists, it says results are unmeasured and points at the bench command.

---

## Agent notes

Interfaces: `ICapture`, `IPreprocessor`, `IInferBackend`, `ITracker`. Default tracker SORT. Default live source a file so demos replay. TensorRT C++ (`NvInfer`) preferred. Product name in HUD and docs is CUDATracker (not CUDAEdgeVision).

Done means all of this exists: file/camera app with tracks; CPU preprocess; CUDA preprocess + tests; ONNX + TRT FP32/FP16; a PyTorch or ORT baseline; stage timers + percentiles; Jetson telemetry with fallback; one-command matrix; plots; `docs/results.md` with real numbers; architecture/setup/bench/optimization writeups; a README a stranger can follow. Until then, leave the phase list visible.
