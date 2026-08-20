#pragma once

#include "app/types.hpp"
#include <opencv2/dnn.hpp>
#include <stdexcept>
#include <vector>

namespace ct {

class IInferBackend {
public:
    virtual ~IInferBackend() = default;
    virtual std::vector<cv::Mat> infer(const cv::Mat& nchw) = 0;
    virtual std::vector<cv::Mat> infer_device(float*) {
        throw std::runtime_error("this backend has no device-buffer infer");
    }
};

class OpenCvDnnBackend final : public IInferBackend {
public:
    explicit OpenCvDnnBackend(const std::string& onnx, bool try_cuda);
    std::vector<cv::Mat> infer(const cv::Mat& nchw) override;

private:
    cv::dnn::Net net_;
};

std::vector<Detection> decode_yolo(const std::vector<cv::Mat>& outs, const LetterboxMeta& meta, float conf, float iou);

}  // namespace ct
