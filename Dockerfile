# Optional Jetson/L4T image. Adjust the tag to your JetPack.
FROM nvcr.io/nvidia/l4t-tensorrt:r10.0-aarch64
WORKDIR /opt/cudatracker
ENV PYTHONPATH=/opt/cudatracker/python
COPY . .
# JetPack supplies CUDA, TensorRT, and OpenCV. Desktop requirements can replace those libraries.
RUN python3 -m pip install --no-cache-dir numpy PyYAML psutil \
    && python3 -c "import cv2"
CMD ["python3", "-m", "cudatracker", "--config", "configs/default.yaml", "--backend", "onnx", "--device", "cpu", "--no-vis", "--bench"]
