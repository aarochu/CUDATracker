#pragma once

#include "app/types.hpp"
#include <chrono>
#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

namespace ct {

class Capture {
public:
    bool open(const AppConfig& cfg, bool loop = false);
    bool read(cv::Mat& bgr, std::uint64_t& timestamp_ns, std::uint64_t& index);
    void close();

private:
    cv::VideoCapture cap_;
    AppConfig cfg_{};
    std::uint64_t index_ = 0;
    std::chrono::steady_clock::time_point last_read_{};
    bool loop_ = false;
};

}  // namespace ct
