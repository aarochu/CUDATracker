# models/

Do not commit `.pt`, `.onnx`, or `.engine` files.

YOLOv8n is the default (Ultralytics, AGPL-3.0 for the Ultralytics package; COCO labels). Export is the supported path:

```text
python scripts/export_onnx.py --model yolov8n --imgsz 640 --opset 17
```

That writes ONNX **without** in-graph NMS so PyTorch, ORT, and TensorRT share the same postprocess.

Engine (when `trtexec` exists):

```text
python scripts/build_engine.py --onnx models/yolov8n.onnx --precision fp16
python scripts/build_engine.py --onnx models/yolov8n.onnx --precision fp32
```

If `trtexec` is missing, the TensorRT backend uses ONNX Runtime's TensorRT EP and caches engines under `models/ort_trt_cache/`.
