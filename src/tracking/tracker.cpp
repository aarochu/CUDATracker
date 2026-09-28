#include "tracker.hpp"
#include "hungarian.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <utility>

namespace ct {
namespace {

void eye7(double a[7][7], double s = 1.0) {
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j) a[i][j] = (i == j) ? s : 0.0;
}

void mul77(const double a[7][7], const double b[7][7], double o[7][7]) {
    double t[7][7]{};
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            for (int k = 0; k < 7; ++k) t[i][j] += a[i][k] * b[k][j];
    std::memcpy(o, t, sizeof(t));
}

void mul77T(const double a[7][7], const double b[7][7], double o[7][7]) {
    double t[7][7]{};
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j)
            for (int k = 0; k < 7; ++k) t[i][j] += a[i][k] * b[j][k];
    std::memcpy(o, t, sizeof(t));
}

void add77(double a[7][7], const double b[7][7]) {
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j) a[i][j] += b[i][j];
}

void mul7(const double a[7][7], const double x[7], double o[7]) {
    double t[7]{};
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j) t[i] += a[i][j] * x[j];
    std::memcpy(o, t, sizeof(t));
}

float iou_xyxy(float ax1, float ay1, float ax2, float ay2, float bx1, float by1, float bx2, float by2) {
    float ix1 = std::max(ax1, bx1), iy1 = std::max(ay1, by1);
    float ix2 = std::min(ax2, bx2), iy2 = std::min(ay2, by2);
    float iw = std::max(0.f, ix2 - ix1), ih = std::max(0.f, iy2 - iy1);
    float inter = iw * ih;
    float a = std::max(0.f, ax2 - ax1) * std::max(0.f, ay2 - ay1);
    float b = std::max(0.f, bx2 - bx1) * std::max(0.f, by2 - by1);
    return inter / (a + b - inter + 1e-9f);
}

void xywh_to_xysr(float x, float y, float w, float h, double& cx, double& cy, double& s, double& r) {
    cx = x;
    cy = y;
    s = std::max(static_cast<double>(w) * h, 1e-6);
    r = w / std::max(h, 1e-6f);
}

}  // namespace

SortTracker::Kalman::Kalman(double cx, double cy, double s, double r) {
    x[0] = cx;
    x[1] = cy;
    x[2] = s;
    x[3] = r;
    eye7(P, 10.0);
    for (int i = 4; i < 7; ++i) P[i][i] = 10000.0;
}

void SortTracker::Kalman::predict() {
    if (x[2] + x[6] <= 0) x[6] = 0;
    double F[7][7];
    eye7(F);
    F[0][4] = F[1][5] = F[2][6] = 1;
    double xn[7];
    mul7(F, x, xn);
    std::memcpy(x, xn, sizeof(x));
    double FP[7][7], FPF[7][7];
    mul77(F, P, FP);
    mul77T(FP, F, FPF);
    std::memcpy(P, FPF, sizeof(P));
    double Q[7][7];
    eye7(Q, 1.0);
    for (int i = 4; i < 7; ++i) Q[i][i] = 0.01;
    add77(P, Q);
}

void SortTracker::Kalman::correct(double cx, double cy, double s, double r) {
    // z = Hx, H is 4x7 identity on the first 4 states
    double z[4] = {cx, cy, s, r};
    double y[4] = {z[0] - x[0], z[1] - x[1], z[2] - x[2], z[3] - x[3]};
    double S[4][4]{};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) S[i][j] = P[i][j];
    S[0][0] += 1;
    S[1][1] += 1;
    S[2][2] += 10;
    S[3][3] += 10;
    // invert 4x4 S (Gauss-Jordan)
    double A[4][8]{};
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) A[i][j] = S[i][j];
        A[i][4 + i] = 1;
    }
    for (int i = 0; i < 4; ++i) {
        int piv = i;
        for (int r = i + 1; r < 4; ++r)
            if (std::fabs(A[r][i]) > std::fabs(A[piv][i])) piv = r;
        if (piv != i)
            for (int c = 0; c < 8; ++c) std::swap(A[i][c], A[piv][c]);
        const double d = A[i][i] != 0 ? A[i][i] : 1e-12;
        for (int c = 0; c < 8; ++c) A[i][c] /= d;
        for (int r = 0; r < 4; ++r) {
            if (r == i) continue;
            const double f = A[r][i];
            for (int c = 0; c < 8; ++c) A[r][c] -= f * A[i][c];
        }
    }
    double Sinv[4][4];
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 4; ++j) Sinv[i][j] = A[i][4 + j];
    // K = P H^T Sinv  -> 7x4, H^T is first 4 columns of I
    double K[7][4]{};
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 4; ++j)
            for (int k = 0; k < 4; ++k) K[i][j] += P[i][k] * Sinv[k][j];
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 4; ++j) x[i] += K[i][j] * y[j];
    // P = (I - K H) P
    double KH[7][7]{};
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 4; ++j) KH[i][j] = K[i][j];
    double IKH[7][7];
    eye7(IKH);
    for (int i = 0; i < 7; ++i)
        for (int j = 0; j < 7; ++j) IKH[i][j] -= KH[i][j];
    double Pn[7][7];
    mul77(IKH, P, Pn);
    std::memcpy(P, Pn, sizeof(P));
}

void SortTracker::Kalman::as_xywh(float& cx, float& cy, float& w, float& h) const {
    cx = static_cast<float>(x[0]);
    cy = static_cast<float>(x[1]);
    const double s = std::max(x[2], 1e-6);
    const double r = std::max(x[3], 1e-6);
    w = static_cast<float>(std::sqrt(s * r));
    h = static_cast<float>(s / std::max(static_cast<double>(w), 1e-6));
}

SortTracker::SortTracker(int max_age, int min_hits, float iou)
    : max_age_(max_age), min_hits_(min_hits), iou_(iou) {}

std::vector<Track> SortTracker::update(const std::vector<Detection>& dets) {
    ++frame_count_;
    for (auto& t : tracks_) {
        t.kf.predict();
        t.age += 1;
        t.time_since_update += 1;
        float cx, cy, w, h;
        t.kf.as_xywh(cx, cy, w, h);
        t.history.push_back({cx, cy});
        if (t.history.size() > 30) t.history.erase(t.history.begin());
    }

    std::vector<int> matched_t, matched_d;
    if (!tracks_.empty() && !dets.empty()) {
        std::vector<std::vector<double>> cost(tracks_.size(), std::vector<double>(dets.size(), 1.0));
        std::vector<std::vector<float>> ious(tracks_.size(), std::vector<float>(dets.size(), 0.f));
        for (int ti = 0; ti < static_cast<int>(tracks_.size()); ++ti) {
            float cx, cy, w, h;
            tracks_[ti].kf.as_xywh(cx, cy, w, h);
            const float ax1 = cx - w / 2, ay1 = cy - h / 2, ax2 = cx + w / 2, ay2 = cy + h / 2;
            for (int di = 0; di < static_cast<int>(dets.size()); ++di) {
                const auto& d = dets[di];
                const float iou =
                    iou_xyxy(ax1, ay1, ax2, ay2, d.x - d.w / 2, d.y - d.h / 2, d.x + d.w / 2, d.y + d.h / 2);
                ious[ti][di] = iou;
                cost[ti][di] = 1.0 - static_cast<double>(iou);
            }
        }
        auto [rows, cols] = linear_sum_assignment(cost);
        for (size_t k = 0; k < rows.size(); ++k) {
            if (ious[rows[k]][cols[k]] >= iou_) {
                matched_t.push_back(rows[k]);
                matched_d.push_back(cols[k]);
            }
        }
    }

    std::vector<char> used_t(tracks_.size(), 0), used_d(dets.size(), 0);
    for (size_t k = 0; k < matched_t.size(); ++k) {
        used_t[matched_t[k]] = 1;
        used_d[matched_d[k]] = 1;
        auto& t = tracks_[matched_t[k]];
        const auto& d = dets[matched_d[k]];
        double cx, cy, s, r;
        xywh_to_xysr(d.x, d.y, d.w, d.h, cx, cy, s, r);
        t.kf.correct(cx, cy, s, r);
        t.conf = d.confidence;
        t.class_id = d.class_id;
        t.class_name = d.class_name;
        t.time_since_update = 0;
        t.hits += 1;
        float pcx, pcy, pw, ph;
        t.kf.as_xywh(pcx, pcy, pw, ph);
        if (!t.history.empty()) t.history.back() = {pcx, pcy};
    }
    for (int di = 0; di < static_cast<int>(dets.size()); ++di) {
        if (used_d[di]) continue;
        Internal t;
        t.id = next_id_++;
        double cx, cy, s, r;
        xywh_to_xysr(dets[di].x, dets[di].y, dets[di].w, dets[di].h, cx, cy, s, r);
        t.kf = Kalman(cx, cy, s, r);
        t.conf = dets[di].confidence;
        t.class_id = dets[di].class_id;
        t.class_name = dets[di].class_name;
        t.hits = 1;
        t.age = 1;
        t.history.push_back({dets[di].x, dets[di].y});
        tracks_.push_back(std::move(t));
    }
    tracks_.erase(std::remove_if(tracks_.begin(), tracks_.end(),
                                 [&](const Internal& t) { return t.time_since_update > max_age_; }),
                  tracks_.end());
    std::vector<Track> out;
    for (const auto& t : tracks_) {
        if (t.time_since_update > 0) continue;
        // SORT: unconfirmed tracks only show during the stream's first min_hits frames.
        if (t.hits < min_hits_ && frame_count_ > min_hits_) continue;
        Track o;
        o.track_id = t.id;
        o.class_id = t.class_id;
        o.class_name = t.class_name;
        t.kf.as_xywh(o.x, o.y, o.w, o.h);
        o.confidence = t.conf;
        o.age = t.age;
        o.history = t.history;
        out.push_back(o);
    }
    return out;
}

}  // namespace ct
