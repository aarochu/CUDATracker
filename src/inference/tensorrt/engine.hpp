#pragma once

#ifdef CT_WITH_TENSORRT

#include "detection/detector.hpp"
#include <memory>
#include <string>
#include <vector>

namespace ct {

class TrtBackend final : public IInferBackend {
public:
    TrtBackend(const std::string& engine_path, int imgsz);
    ~TrtBackend() override;
    TrtBackend(const TrtBackend&) = delete;
    TrtBackend& operator=(const TrtBackend&) = delete;
    std::vector<cv::Mat> infer(const cv::Mat& nchw) override;
    std::vector<cv::Mat> infer_device(float* device_nchw) override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ct

#endif
