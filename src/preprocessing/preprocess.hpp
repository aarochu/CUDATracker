#pragma once

#include "app/types.hpp"
#include <opencv2/core.hpp>
#include <vector>

namespace ct {

LetterboxMeta letterbox_geometry(int src_w, int src_h, int imgsz, bool letterbox);
void cpu_preprocess(const cv::Mat& bgr, int imgsz, bool letterbox, int pad_value,
                    cv::Mat& nchw, LetterboxMeta& meta);

#ifdef CT_WITH_CUDA
class CudaPreprocessor {
public:
    explicit CudaPreprocessor(int imgsz, int pad_value, bool fused, bool letterbox);
    ~CudaPreprocessor();
    CudaPreprocessor(const CudaPreprocessor&) = delete;
    CudaPreprocessor& operator=(const CudaPreprocessor&) = delete;
    void run(const cv::Mat& bgr, float* host_nchw, LetterboxMeta& meta, bool copy_to_host,
             double& transfer_ms, double& kernel_ms);
    float* device_nchw() const { return static_cast<float*>(d_chw_); }

private:
    int imgsz_;
    int pad_value_;
    bool fused_;
    bool letterbox_;
    void* d_src_ = nullptr;
    void* d_canvas_ = nullptr;
    void* d_rgb_ = nullptr;
    void* d_hwc_ = nullptr;
    void* d_chw_ = nullptr;
    void* ev_start_ = nullptr;
    void* ev_stop_ = nullptr;
    size_t src_cap_ = 0;
};
#endif

}  // namespace ct
