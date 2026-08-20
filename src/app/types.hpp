#pragma once

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace ct {

struct LetterboxMeta {
    float scale = 1.f;
    float scale_x = 1.f;
    float scale_y = 1.f;
    int pad_x = 0;
    int pad_y = 0;
    int new_w = 0;
    int new_h = 0;
    int src_w = 0;
    int src_h = 0;
    int imgsz = 640;
};

struct Detection {
    float x = 0, y = 0, w = 0, h = 0;
    float confidence = 0;
    int class_id = -1;
    std::string class_name;
    int track_id = -1;
};

struct Track {
    int track_id = 0;
    int class_id = 0;
    std::string class_name;
    float x = 0, y = 0, w = 0, h = 0;
    float confidence = 0;
    int age = 0;
    std::vector<std::pair<float, float>> history;
};

struct StageTimes {
    double capture_ms = 0;
    double preprocess_ms = 0;
    double transfer_ms = 0;
    double inference_ms = 0;
    double postprocess_ms = 0;
    double tracking_ms = 0;
    double render_ms = 0;
    double end_to_end_ms = 0;
};

struct AppConfig {
    std::string source = "samples/vtest.avi";
    int width = 640;
    int height = 480;
    int fps_cap = 0;
    bool gstreamer = false;
    std::string preprocess = "cpu";
    bool letterbox = true;
    int pad_value = 114;
    bool fused = false;
    std::string model_name = "yolov8n";
    std::string weights = "models/yolov8n.pt";
    std::string onnx = "models/yolov8n.onnx";
    std::string engine = "models/yolov8n_fp16.engine";
    std::string engine_fp32 = "models/yolov8n_fp32.engine";
    std::string engine_fp16 = "models/yolov8n_fp16.engine";
    int imgsz = 640;
    float conf = 0.25f;
    float iou = 0.45f;
    std::string backend = "onnx";
    std::string precision = "fp32";
    std::string device = "gpu";
    int max_age = 30;
    int min_hits = 3;
    float track_iou = 0.3f;
    bool vis = true;
    bool trails = true;
    bool hud = true;
    bool hardware_panel = false;
    bool hardware = true;
    int warmup_frames = 50;
    int measure_frames = 300;
    std::string output_video;
    std::string output_metrics;
};

AppConfig load_config(const std::string& path);
void log_line(const char* level, const char* stage, const std::string& msg);

}  // namespace ct
