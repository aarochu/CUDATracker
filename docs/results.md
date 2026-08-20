# Results

Official Jetson matrix (50 warmup / 300 measure, fixed nvpmodel): not run yet. Use `python benchmarks/benchmark.py --config configs/benchmark.yaml` and paste from `benchmarks/results/matrix.csv`.

Smoke on an RTX 5060 laptop (not a Jetson), `samples/vtest.avi` resized to 640×480, YOLOv8n, visualization off, handful of frames only:

| preprocess | backend | precision | FPS | e2e mean |
| --- | --- | --- | --- | --- |
| cpu | pytorch | fp32 | 48.0 | 20.8 ms |
| cuda | pytorch | fp32 | 56.4 | 17.8 ms |
| cpu | onnx | fp32 | 50.1 | 20.0 ms |

Treat that as a wiring check, not a paper table. CUDA preprocess was faster here; infer still owned most of the frame. Re-run with the official protocol before quoting a percent.
