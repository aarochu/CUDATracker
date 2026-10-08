# Jetson setup

Source of truth for flashing, packages, cameras, and power modes is NVIDIA’s [Jetson documentation hub](https://docs.nvidia.com/jetson/index.html). This page only maps that stack onto CUDATracker. If a command here disagrees with the Developer Guide for your L4T release, the Developer Guide wins.

CUDATracker does not hardcode one SoC. Record the board you actually ran (`/proc/device-tree/model`) in every published JSON.

## Which JetPack

| Hardware | What NVIDIA ships |
| --- | --- |
| Orin Nano, Orin NX, AGX Orin | JetPack 6.x (Jetson Linux R36) and, on current AGX Orin kits, JetPack 7.x (R39). Flash the image for **this** kit; Orin Nano SD-card units need the firmware update NVIDIA documents before JetPack 6. |
| Xavier NX, AGX Xavier | JetPack 5.x (Jetson Linux R35) is the stack those modules were built for. Do not assume a JetPack 6/7 image will boot them. |

Check what is on the board (L4T is not the same string as JetPack):

```text
cat /etc/nv_tegra_release          # L4T, e.g. R36 or R39
dpkg-query -W nvidia-jetpack       # JetPack metapackage version, if installed
cat /proc/device-tree/model        # module + carrier
```

After a Linux BSP flash, install the matching compute stack:

```text
sudo apt update
sudo apt install nvidia-jetpack
```

See [Install and configure JetPack](https://docs.nvidia.com/jetson/jetpack/install-setup/index.html).

## Packages this repo needs

`bash scripts/setup_jetson.sh` on the **board** (not on a Windows laptop).

JetPack already provides CUDA, cuDNN, TensorRT (`/usr/src/tensorrt/bin/trtexec`), and usually OpenCV. Do **not** `pip install torch` or `opencv-python` from PyPI on aarch64 — those wheels are not the Jetson stack.

PyTorch: use NVIDIA’s [PyTorch for Jetson](https://docs.nvidia.com/deeplearning/frameworks/install-pytorch-jetson-platform/index.html) wheel for this JetPack/L4T. If that wheel is too heavy, run ONNX Runtime / TensorRT only.

Then:

```text
python3 -m pip install --user numpy PyYAML matplotlib pytest psutil ultralytics onnx onnxslim
export PYTHONPATH="$PWD/python"
python3 scripts/fetch_sample.py
python3 scripts/export_onnx.py --model yolov8n --imgsz 640
```

Skip `onnxruntime-gpu` from PyPI on Jetson. TensorRT is the JetPack library (`libnvinfer`); Python `tensorrt` is the JetPack binding, not `pip install tensorrt-cu13` (that package is x86 CUDA).

## Camera

USB: `--source 0` (V4L2).

CSI / Argus: OpenCV must be built with GStreamer, and the pipeline must use NVMM as NVIDIA documents in [Accelerated GStreamer](https://docs.nvidia.com/jetson/archives/r39.2/DeveloperGuide/SD/Multimedia/AcceleratedGstreamer.html) and the Argus camera chapter. Width, height, framerate, and `sensor-id` are module-specific.

```text
python3 -m cudatracker --source "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM), width=1280, height=720, framerate=30/1, format=NV12 ! nvvidconv ! video/x-raw, format=BGRx ! videoconvert ! video/x-raw, format=BGR ! appsink" --config configs/default.yaml
```

Set `input.gstreamer: true` in YAML if the string is not auto-detected. USB cameras with an onboard ISP use V4L2 (`v4l2src` / `/dev/video0`), not `nvarguscamerasrc`. Jetson Thor camera bring-up uses SIPL (`nvsiplsrc`); that is a different plugin.

Confirm GStreamer in OpenCV: `python3 -c "import cv2; print(cv2.getBuildInformation())"` and look for GStreamer YES. A pip `opencv-python` wheel usually will not talk to Argus.

## C++ binary

Headers for TensorRT on device are typically under `/usr/include/aarch64-linux-gnu`. `trtexec` is `/usr/src/tensorrt/bin/trtexec` once `nvidia-jetpack` / `tensorrt` is installed.

```text
export PATH="/usr/src/tensorrt/bin:$PATH"
cmake -S . -B build -DWITH_CUDA=ON -DWITH_TENSORRT=ON
cmake --build build -j$(nproc)
./build/cudatracker --config configs/default.yaml --source samples/vtest.avi
```

nvcc targets Orin `sm_87`. Xavier `sm_72` is added when the CUDA toolkit is older than 13 (JetPack 5). CUDA 13 dropped `sm_72`. Desktop CUDA 13 also emits `sm_120` for current GeForce.

## Clocks before a published matrix

Power modes and max-clocks are described in the Jetson Linux [power and performance](https://docs.nvidia.com/jetson/archives/r36.5/DeveloperGuide/SD/PlatformPowerAndPerformance/JetsonOrinNanoSeriesJetsonOrinNxSeriesAndJetsonAgxOrinSeries.html) chapter for Orin (Xavier has its own chapter). Commands:

```text
sudo nvpmodel -q --verbose          # current mode; modes live in /etc/nvpmodel.conf
sudo nvpmodel -m <id>               # pick one mode and keep it
sudo jetson_clocks --show
sudo jetson_clocks                  # pin CPU/GPU/EMC to the max of the current nvpmodel
```

`nvpmodel -m 0` is MAXN only on SKUs that define mode 0 as MAXN. Orin Nano / NX “Super” MAXN_SUPER exists only if you flashed a `*-super*` board config. NVIDIA documents MAXN as an unconstrained/experimental mode, not “always fastest, no throttle.” Mixing two `nvpmodel` modes (or clocks on vs off) in one results table is an invalid run.

`tegrastats` (interval 1 s) is what this repo samples for GR3D and module power when `/etc/nv_tegra_release` exists. JSON also stores L4T, `nvidia-jetpack` if present, `nvpmodel -q`, and `jetson_clocks --show`.

Windows/x86 is the export and kernel-test machine. Jetson-only bits (`nvarguscamerasrc`, `nvpmodel`, INA/`tegrastats`) stay `na` there.
