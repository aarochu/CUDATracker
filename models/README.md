# models/

Do not commit `.pt`, `.onnx`, or `.engine` files.

YOLOv8n is the default (Ultralytics, AGPL-3.0 for the Ultralytics package; COCO labels). Export is the supported path:

```text
python scripts/export_onnx.py --model yolov8n --imgsz 640 --opset 17
```

That writes ONNX **without** in-graph NMS so PyTorch, ORT, and TensorRT share the same postprocess.

Engine (when `trtexec` exists — on Jetson that is `/usr/src/tensorrt/bin/trtexec` after `nvidia-jetpack`):

```text
python scripts/build_engine.py --onnx models/yolov8n.onnx --precision fp16
python scripts/build_engine.py --onnx models/yolov8n.onnx --precision fp32
```

If `trtexec` is missing, the script uses the Python TensorRT API. TensorRT 11 is strongly typed and has no `BuilderFlag.FP16`; the script converts ONNX to FP16 (keep FP32 I/O) then builds. Do not `pip install tensorrt-cu13` on Jetson.
