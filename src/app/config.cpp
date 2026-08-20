#include "types.hpp"

#include <cctype>
#include <fstream>
#include <iostream>
#include <map>
#include <sstream>
#include <stdexcept>

namespace ct {
namespace {

std::string trim(std::string s) {
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.front()))) s.erase(s.begin());
    while (!s.empty() && std::isspace(static_cast<unsigned char>(s.back()))) s.pop_back();
    if (!s.empty() && (s.front() == '"' || s.front() == '\'')) {
        char q = s.front();
        if (s.back() == q && s.size() >= 2) s = s.substr(1, s.size() - 2);
    }
    return s;
}

int as_int(const std::string& v, int d) {
    if (v.empty()) return d;
    try {
        return std::stoi(v);
    } catch (...) {
        return d;
    }
}

float as_float(const std::string& v, float d) {
    if (v.empty()) return d;
    try {
        return std::stof(v);
    } catch (...) {
        return d;
    }
}

bool as_bool(const std::string& v, bool d) {
    if (v.empty()) return d;
    return v == "true" || v == "True" || v == "1" || v == "yes";
}

}  // namespace

AppConfig load_config(const std::string& path) {
    std::ifstream in(path);
    if (!in) throw std::runtime_error("cannot open config " + path);
    AppConfig c;
    std::string section;
    std::string line;
    std::map<std::string, std::map<std::string, std::string>> kv;
    while (std::getline(in, line)) {
        auto hash = line.find('#');
        if (hash != std::string::npos) line = line.substr(0, hash);
        line = trim(line);
        if (line.empty()) continue;
        if (line.back() == ':' && line.find(' ') == std::string::npos) {
            section = line.substr(0, line.size() - 1);
            continue;
        }
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        auto key = trim(line.substr(0, colon));
        auto val = trim(line.substr(colon + 1));
        kv[section][key] = val;
    }
    auto g = [&](const char* sec, const char* key, const std::string& def) {
        auto it = kv.find(sec);
        if (it == kv.end()) return def;
        auto jt = it->second.find(key);
        return jt == it->second.end() ? def : jt->second;
    };
    c.source = g("input", "source", c.source);
    auto res = g("input", "resolution", "640x480");
    auto x = res.find('x');
    if (x != std::string::npos) {
        c.width = as_int(res.substr(0, x), 640);
        c.height = as_int(res.substr(x + 1), 480);
    }
    c.fps_cap = as_int(g("input", "fps_cap", "0"), 0);
    c.gstreamer = as_bool(g("input", "gstreamer", "false"), false);
    c.preprocess = g("preprocess", "backend", c.preprocess);
    c.letterbox = as_bool(g("preprocess", "letterbox", "true"), true);
    c.pad_value = as_int(g("preprocess", "pad_value", "114"), 114);
    c.fused = as_bool(g("preprocess", "fused", "false"), false);
    c.model_name = g("model", "name", c.model_name);
    c.weights = g("model", "weights", c.weights);
    c.onnx = g("model", "onnx", c.onnx);
    c.engine = g("model", "engine", c.engine);
    c.engine_fp32 = g("model", "engine_fp32", c.engine_fp32);
    c.engine_fp16 = g("model", "engine_fp16", c.engine_fp16);
    c.imgsz = as_int(g("model", "imgsz", "640"), 640);
    c.conf = as_float(g("model", "conf", "0.25"), 0.25f);
    c.iou = as_float(g("model", "iou", "0.45"), 0.45f);
    c.backend = g("inference", "backend", c.backend);
    c.precision = g("inference", "precision", c.precision);
    c.device = g("inference", "device", c.device);
    c.max_age = as_int(g("tracker", "max_age", "30"), 30);
    c.min_hits = as_int(g("tracker", "min_hits", "3"), 3);
    c.track_iou = as_float(g("tracker", "iou_threshold", "0.3"), 0.3f);
    c.vis = as_bool(g("visualization", "enabled", "true"), true);
    c.trails = as_bool(g("visualization", "trails", "true"), true);
    c.hud = as_bool(g("visualization", "hud", "true"), true);
    c.hardware_panel = as_bool(g("visualization", "hardware_panel", "false"), false);
    c.hardware = as_bool(g("telemetry", "hardware", "true"), true);
    c.warmup_frames = as_int(g("benchmark", "warmup_frames", "50"), 50);
    c.measure_frames = as_int(g("benchmark", "measure_frames", "300"), 300);
    c.output_video = g("output", "video", "");
    c.output_metrics = g("output", "metrics", "");
    return c;
}

}  // namespace ct
