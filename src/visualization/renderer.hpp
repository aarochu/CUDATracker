#pragma once

#include "app/types.hpp"
#include "tracking/tracker.hpp"
#include <opencv2/core.hpp>

namespace ct {

cv::Mat draw_overlay(const cv::Mat& frame, const std::vector<Track>& tracks, const StageTimes& times, double fps,
                     const AppConfig& cfg, bool hud, bool trails, const std::string& hw_line);

}  // namespace ct
