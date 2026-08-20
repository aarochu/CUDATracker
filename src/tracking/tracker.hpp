#pragma once

#include "app/types.hpp"
#include <string>
#include <vector>

namespace ct {

class SortTracker {
public:
    SortTracker(int max_age, int min_hits, float iou);
    std::vector<Track> update(const std::vector<Detection>& dets);

private:
    int max_age_;
    int min_hits_;
    float iou_;
    int next_id_ = 1;

    struct Kalman {
        double x[7]{};
        double P[7][7]{};
        Kalman() = default;
        Kalman(double cx, double cy, double s, double r);
        void predict();
        void correct(double cx, double cy, double s, double r);
        void as_xywh(float& cx, float& cy, float& w, float& h) const;
    };

    struct Internal {
        int id = 0;
        int class_id = 0;
        std::string class_name;
        float conf = 0;
        int age = 0;
        int hits = 0;
        int time_since_update = 0;
        Kalman kf;
        std::vector<std::pair<float, float>> history;
    };
    std::vector<Internal> tracks_;
};

}  // namespace ct
