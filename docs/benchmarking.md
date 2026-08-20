# Benchmarking

Same clip, same conf/iou, visualization off.

```text
python scripts/fetch_sample.py
python scripts/export_onnx.py --model yolov8n
python benchmarks/benchmark.py --config configs/benchmark.yaml
python benchmarks/plot.py --input benchmarks/results
```

`--quick` on the matrix runner is 2 warmup / 8 measure frames for a smoke pass. Published numbers should use the YAML defaults (50 / 300) and a recorded `nvpmodel` / `jetson_clocks` state on Jetson.

Cells that cannot run (missing engine, ONNX+fp16, no TensorRT EP) write `skipped=true` and a reason. Do not delete those rows.

JSON per cell includes git hash, host, CUDA/OpenCV versions, config snapshot, stage percentiles, hardware sample, and a SHA-256 of a file source when the source is a file.

Plots: `benchmarks/plots/fps.png`, `latency.png`, plus CPU/GPU util when the CSV has numbers.
