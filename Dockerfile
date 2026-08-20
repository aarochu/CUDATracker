# Optional Jetson/L4T image. Adjust the tag to your JetPack.
FROM nvcr.io/nvidia/l4t-tensorrt:r10.0-aarch64
WORKDIR /opt/cudatracker
COPY . .
RUN python3 -m pip install -r requirements.txt || true
CMD ["python3", "-m", "cudatracker", "--config", "configs/default.yaml", "--no-vis", "--bench"]
