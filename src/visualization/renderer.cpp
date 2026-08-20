#include "renderer.hpp"

#include <algorithm>
#include <iomanip>
#include <opencv2/imgproc.hpp>
#include <sstream>

namespace ct {

namespace {
cv::Scalar class_color(int id) {
    int r = (id * 37 + 80) % 180 + 40;
    int g = (id * 67 + 40) % 180 + 40;
    int b = (id * 97 + 20) % 180 + 40;
    return cv::Scalar(b, g, r);
}
}  // namespace

cv::Mat draw_overlay(const cv::Mat& frame, const std::vector<Track>& tracks, const StageTimes& times, double fps,
                     const AppConfig& cfg, bool hud, bool trails, const std::string& hw_line) {
    cv::Mat vis = frame.clone();
    for (const auto& tr : tracks) {
        auto color = class_color(tr.class_id);
        int x1 = int(tr.x - tr.w / 2), y1 = int(tr.y - tr.h / 2);
        int x2 = int(tr.x + tr.w / 2), y2 = int(tr.y + tr.h / 2);
        cv::rectangle(vis, {x1, y1}, {x2, y2}, color, 2);
        cv::putText(vis, tr.class_name + "  ID " + std::to_string(tr.track_id), {x1, std::max(16, y1 - 6)},
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
        if (trails && tr.history.size() > 1) {
            for (size_t i = 1; i < tr.history.size(); ++i) {
                cv::line(vis, {int(tr.history[i - 1].first), int(tr.history[i - 1].second)},
                         {int(tr.history[i].first), int(tr.history[i].second)}, color, 1, cv::LINE_AA);
            }
        }
    }
    if (!hud) return vis;
    std::ostringstream l2, l3, l4;
    l2 << std::fixed << std::setprecision(1) << fps << " FPS";
    l3 << cfg.preprocess << "  |  " << cfg.backend << ' ' << cfg.precision << "  |  SORT";
    l4 << "e2e " << times.end_to_end_ms << " ms   infer " << times.inference_ms << "  pre " << times.preprocess_ms;
    std::vector<std::string> lines = {"CUDATracker", l2.str(), l3.str(), l4.str()};
    if (!hw_line.empty()) lines.push_back(hw_line);
    int bar_h = 14 + int(lines.size()) * 18;
    cv::Mat overlay = vis.clone();
    cv::rectangle(overlay, {0, 0}, {520, bar_h}, {0, 0, 0}, -1);
    cv::addWeighted(overlay, 0.45, vis, 0.55, 0, vis);
    int y = 22;
    for (size_t i = 0; i < lines.size(); ++i) {
        double scale = i == 1 ? 0.72 : 0.48;
        cv::putText(vis, lines[i], {10, y}, cv::FONT_HERSHEY_SIMPLEX, scale, {240, 240, 240}, i == 1 ? 2 : 1,
                    cv::LINE_AA);
        y += i == 1 ? 22 : 18;
    }
    return vis;
}

}  // namespace ct
