#include "detector.hpp"

#include "app/types.hpp"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <map>
#include <opencv2/dnn.hpp>
#include <stdexcept>
#include <vector>

namespace ct {
namespace {

const char* kCoco[] = {
    "person",    "bicycle",      "car",          "motorcycle", "airplane",     "bus",
    "train",     "truck",        "boat",         "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench",    "bird",         "cat",        "dog",          "horse",
    "sheep",     "cow",          "elephant",     "bear",       "zebra",        "giraffe",
    "backpack",  "umbrella",     "handbag",      "tie",        "suitcase",     "frisbee",
    "skis",      "snowboard",    "sports ball",  "kite",       "baseball bat", "baseball glove",
    "skateboard","surfboard",    "tennis racket","bottle",     "wine glass",   "cup",
    "fork",      "knife",        "spoon",        "bowl",       "banana",       "apple",
    "sandwich",  "orange",       "broccoli",     "carrot",     "hot dog",      "pizza",
    "donut",     "cake",         "chair",        "couch",      "potted plant", "bed",
    "dining table", "toilet",    "tv",           "laptop",     "mouse",        "remote",
    "keyboard",  "cell phone",   "microwave",    "oven",       "toaster",      "sink",
    "refrigerator", "book",      "clock",        "vase",       "scissors",     "teddy bear",
    "hair drier","toothbrush"};

}  // namespace

OpenCvDnnBackend::OpenCvDnnBackend(const std::string& onnx, bool try_cuda) {
    std::ifstream probe(onnx);
    if (!probe) throw std::runtime_error("ONNX file missing: " + onnx + ". Run python scripts/export_onnx.py");
    net_ = cv::dnn::readNetFromONNX(onnx);
    if (net_.empty()) throw std::runtime_error("OpenCV DNN failed to load " + onnx);
    bool cuda_ok = false;
    if (try_cuda) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
        try {
            const int sz[4] = {1, 3, 640, 640};
            cv::Mat blob(4, sz, CV_32F, cv::Scalar(0));
            net_.setInput(blob);
            std::vector<cv::Mat> outs;
            net_.forward(outs);
            cuda_ok = true;
            log_line("INFO", "infer", "OpenCV DNN CUDA target");
        } catch (const cv::Exception& e) {
            log_line("WARN", "infer", std::string("OpenCV DNN CUDA not in this OpenCV build: ") + e.err);
        }
    }
    if (!cuda_ok) {
        net_.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
        net_.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
        log_line("INFO", "infer", "OpenCV DNN CPU");
    }
}

std::vector<cv::Mat> OpenCvDnnBackend::infer(const cv::Mat& nchw) {
    net_.setInput(nchw);
    std::vector<cv::Mat> outs;
    net_.forward(outs);
    return outs;
}

std::vector<Detection> decode_yolo(const std::vector<cv::Mat>& outs, const LetterboxMeta& meta, float conf,
                                   float iou) {
    if (outs.empty() || outs[0].empty()) return {};
    cv::Mat a = outs[0];
    if (a.depth() != CV_32F) a.convertTo(a, CV_32F);
    if (!a.isContinuous()) a = a.clone();
    while (a.dims > 2 && a.size[0] == 1) {
        std::vector<int> sz;
        sz.reserve(static_cast<size_t>(a.dims - 1));
        for (int d = 1; d < a.dims; ++d) sz.push_back(a.size[d]);
        a = a.reshape(1, static_cast<int>(sz.size()), sz.data());
    }
    auto channels_are_rows = [](int rows, int cols) {
        const bool r_ch = (rows == 84 || rows == 85);
        const bool c_ch = (cols == 84 || cols == 85);
        if (r_ch && !c_ch) return true;
        if (!r_ch && c_ch) return false;
        return rows <= 85 && rows < cols;
    };
    if (a.dims == 3) {
        int d0 = a.size[0];
        a = a.reshape(1, d0);
        if (channels_are_rows(a.rows, a.cols)) cv::transpose(a, a);
    }
    if (a.dims == 2 && channels_are_rows(a.rows, a.cols)) cv::transpose(a, a);
    std::vector<cv::Rect2d> boxes;
    std::vector<float> scores;
    std::vector<int> class_ids;
    std::vector<float> cxs, cys, ws, hs;
    for (int i = 0; i < a.rows; ++i) {
        const float* row = a.ptr<float>(i);
        int ncls = a.cols - 4;
        int best = 0;
        float best_s = 0;
        if (ncls == 81) {
            float obj = row[4];
            for (int c = 0; c < 80; ++c) {
                float s = obj * row[5 + c];
                if (s > best_s) {
                    best_s = s;
                    best = c;
                }
            }
        } else {
            for (int c = 0; c < ncls; ++c) {
                if (row[4 + c] > best_s) {
                    best_s = row[4 + c];
                    best = c;
                }
            }
        }
        if (best_s < conf) continue;
        float cx = row[0], cy = row[1], w = row[2], h = row[3];
        boxes.emplace_back(cx - w / 2.0, cy - h / 2.0, w, h);
        scores.push_back(best_s);
        class_ids.push_back(best);
        cxs.push_back(cx);
        cys.push_back(cy);
        ws.push_back(w);
        hs.push_back(h);
    }
    std::vector<int> keep;
    std::map<int, std::vector<int>> by_class;
    for (int i = 0; i < static_cast<int>(class_ids.size()); ++i) by_class[class_ids[i]].push_back(i);
    for (const auto& kv : by_class) {
        std::vector<cv::Rect2d> b;
        std::vector<float> s;
        b.reserve(kv.second.size());
        s.reserve(kv.second.size());
        for (int i : kv.second) {
            b.push_back(boxes[i]);
            s.push_back(scores[i]);
        }
        std::vector<int> local;
        cv::dnn::NMSBoxes(b, s, conf, iou, local);
        for (int k : local) keep.push_back(kv.second[k]);
    }
    const float sx = meta.scale_x != 0.f ? meta.scale_x : meta.scale;
    const float sy = meta.scale_y != 0.f ? meta.scale_y : meta.scale;
    std::vector<Detection> dets;
    for (int idx : keep) {
        Detection d;
        d.x = (cxs[idx] - meta.pad_x) / sx;
        d.y = (cys[idx] - meta.pad_y) / sy;
        d.w = ws[idx] / sx;
        d.h = hs[idx] / sy;
        d.confidence = scores[idx];
        d.class_id = class_ids[idx];
        if (d.class_id >= 0 && d.class_id < 80) d.class_name = kCoco[d.class_id];
        dets.push_back(d);
    }
    return dets;
}

}  // namespace ct
