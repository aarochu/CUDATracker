#include "camera.hpp"

#include "app/types.hpp"
#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <thread>

namespace ct {

bool Capture::open(const AppConfig& cfg, bool loop) {
    cfg_ = cfg;
    loop_ = loop;
    bool ok = false;
    if (cfg.gstreamer || cfg.source.find("nvarguscamerasrc") != std::string::npos ||
        cfg.source.find(" ! ") != std::string::npos) {
        ok = cap_.open(cfg.source, cv::CAP_GSTREAMER);
    } else if (!cfg.source.empty() &&
               std::all_of(cfg.source.begin(), cfg.source.end(),
                           [](unsigned char ch) { return std::isdigit(ch) != 0; })) {
        ok = cap_.open(std::stoi(cfg.source));
    } else {
        if (cfg.source.compare(0, 5, "/dev/") != 0 && !std::filesystem::exists(cfg.source)) {
            log_line("ERROR", "capture",
                     "Cannot open source '" + cfg.source +
                         "'. Put a video at that path, pass --source 0, or run python scripts/fetch_sample.py");
            return false;
        }
        ok = cap_.open(cfg.source);
    }
    if (!ok) {
        log_line("ERROR", "capture",
                 "OpenCV could not open " + cfg.source +
                     ". For CSI on Jetson, pass a GStreamer string and set input.gstreamer: true.");
        return false;
    }
    cap_.set(cv::CAP_PROP_FRAME_WIDTH, cfg.width);
    cap_.set(cv::CAP_PROP_FRAME_HEIGHT, cfg.height);
    log_line("INFO", "capture", "opened " + cfg.source);
    return true;
}

bool Capture::read(cv::Mat& bgr, std::uint64_t& timestamp_ns, std::uint64_t& index) {
    if (cfg_.fps_cap > 0) {
        const auto min_dt = std::chrono::nanoseconds(1'000'000'000 / cfg_.fps_cap);
        auto now = std::chrono::steady_clock::now();
        auto wait = min_dt - (now - last_read_);
        if (wait > std::chrono::nanoseconds::zero()) std::this_thread::sleep_for(wait);
    }
    if (!cap_.read(bgr) || bgr.empty()) {
        if (loop_) {
            cap_.set(cv::CAP_PROP_POS_FRAMES, 0);
            if (!cap_.read(bgr) || bgr.empty()) return false;
        } else {
            return false;
        }
    }
    last_read_ = std::chrono::steady_clock::now();
    if (bgr.cols != cfg_.width || bgr.rows != cfg_.height) {
        cv::resize(bgr, bgr, cv::Size(cfg_.width, cfg_.height), 0, 0, cv::INTER_LINEAR);
    }
    timestamp_ns = static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(last_read_.time_since_epoch()).count());
    index = index_++;
    return true;
}

void Capture::close() { cap_.release(); }

}  // namespace ct
