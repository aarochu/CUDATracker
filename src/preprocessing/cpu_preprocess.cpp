#include "preprocess.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <opencv2/imgproc.hpp>

namespace ct {

LetterboxMeta letterbox_geometry(int src_w, int src_h, int imgsz, bool letterbox) {
    LetterboxMeta m;
    m.src_w = src_w;
    m.src_h = src_h;
    m.imgsz = imgsz;
    if (!letterbox) {
        m.new_w = imgsz;
        m.new_h = imgsz;
        m.scale_x = static_cast<float>(imgsz) / static_cast<float>(src_w);
        m.scale_y = static_cast<float>(imgsz) / static_cast<float>(src_h);
        m.scale = m.scale_x;
        return m;
    }
    m.scale = std::min(static_cast<float>(imgsz) / src_w, static_cast<float>(imgsz) / src_h);
    m.scale_x = m.scale;
    m.scale_y = m.scale;
    m.new_w = std::max(1, std::min(imgsz, static_cast<int>(std::round(src_w * m.scale))));
    m.new_h = std::max(1, std::min(imgsz, static_cast<int>(std::round(src_h * m.scale))));
    m.pad_x = (imgsz - m.new_w) / 2;
    m.pad_y = (imgsz - m.new_h) / 2;
    return m;
}

void cpu_preprocess(const cv::Mat& bgr, int imgsz, bool letterbox, int pad_value, cv::Mat& nchw,
                    LetterboxMeta& meta) {
    meta = letterbox_geometry(bgr.cols, bgr.rows, imgsz, letterbox);
    cv::Mat canvas(imgsz, imgsz, CV_8UC3, cv::Scalar(pad_value, pad_value, pad_value));
    cv::Mat resized;
    cv::resize(bgr, resized, cv::Size(meta.new_w, meta.new_h), 0, 0, cv::INTER_LINEAR);
    resized.copyTo(canvas(cv::Rect(meta.pad_x, meta.pad_y, meta.new_w, meta.new_h)));
    cv::Mat rgb;
    cv::cvtColor(canvas, rgb, cv::COLOR_BGR2RGB);
    rgb.convertTo(rgb, CV_32F, 1.0 / 255.0);
    std::vector<cv::Mat> ch(3);
    cv::split(rgb, ch);
    const int sizes[4] = {1, 3, imgsz, imgsz};
    nchw.create(4, sizes, CV_32F);
    int hw = imgsz * imgsz;
    for (int c = 0; c < 3; ++c) {
        std::memcpy(nchw.ptr<float>() + c * hw, ch[c].ptr<float>(), hw * sizeof(float));
    }
}

}  // namespace ct
