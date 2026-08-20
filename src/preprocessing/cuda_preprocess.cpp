#include "preprocess.hpp"

#ifdef CT_WITH_CUDA

#include "app/types.hpp"
#include <cuda_runtime.h>
#include <chrono>
#include <cstring>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <vector>

extern "C" {
cudaError_t ct_letterbox_resize(const unsigned char*, int, int, int, unsigned char*, int, int, int, int, int, int,
                                unsigned char, cudaStream_t);
cudaError_t ct_bgr_to_rgb(const unsigned char*, unsigned char*, int, cudaStream_t);
cudaError_t ct_normalize_hwc(const unsigned char*, float*, int, float, cudaStream_t);
cudaError_t ct_hwc_to_chw(const float*, float*, int, int, cudaStream_t);
cudaError_t ct_fused_preprocess(const unsigned char*, int, int, int, float*, int, int, int, int, int, int, unsigned char,
                                float, cudaStream_t);
}

namespace ct {
namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    }
}

}  // namespace

CudaPreprocessor::CudaPreprocessor(int imgsz, int pad_value, bool fused, bool letterbox)
    : imgsz_(imgsz), pad_value_(pad_value), fused_(fused), letterbox_(letterbox) {
    const size_t n = static_cast<size_t>(imgsz) * imgsz;
    src_cap_ = 1920ull * 1080ull * 3ull;
    try {
        check(cudaMalloc(&d_src_, src_cap_), "cudaMalloc src");
        check(cudaMalloc(&d_canvas_, n * 3), "cudaMalloc canvas");
        check(cudaMalloc(&d_rgb_, n * 3), "cudaMalloc rgb");
        check(cudaMalloc(&d_hwc_, n * 3 * sizeof(float)), "cudaMalloc hwc");
        check(cudaMalloc(&d_chw_, n * 3 * sizeof(float)), "cudaMalloc chw");
        cudaEvent_t a = nullptr, b = nullptr;
        check(cudaEventCreate(&a), "cudaEventCreate start");
        check(cudaEventCreate(&b), "cudaEventCreate stop");
        ev_start_ = a;
        ev_stop_ = b;
    } catch (...) {
        cudaFree(d_src_);
        cudaFree(d_canvas_);
        cudaFree(d_rgb_);
        cudaFree(d_hwc_);
        cudaFree(d_chw_);
        if (ev_start_) cudaEventDestroy(static_cast<cudaEvent_t>(ev_start_));
        if (ev_stop_) cudaEventDestroy(static_cast<cudaEvent_t>(ev_stop_));
        d_src_ = d_canvas_ = d_rgb_ = d_hwc_ = d_chw_ = ev_start_ = ev_stop_ = nullptr;
        throw;
    }
}

CudaPreprocessor::~CudaPreprocessor() {
    cudaFree(d_src_);
    cudaFree(d_canvas_);
    cudaFree(d_rgb_);
    cudaFree(d_hwc_);
    cudaFree(d_chw_);
    if (ev_start_) cudaEventDestroy(static_cast<cudaEvent_t>(ev_start_));
    if (ev_stop_) cudaEventDestroy(static_cast<cudaEvent_t>(ev_stop_));
}

void CudaPreprocessor::run(const cv::Mat& bgr, float* out_nchw, LetterboxMeta& meta, bool copy_to_host,
                           double& transfer_ms, double& kernel_ms) {
    cv::Mat src = bgr;
    if (!src.isContinuous() || src.channels() != 3) {
        src = bgr.clone();
    }
    meta = letterbox_geometry(src.cols, src.rows, imgsz_, letterbox_);
    const size_t nbytes = static_cast<size_t>(src.rows) * static_cast<size_t>(src.step);
    if (nbytes > src_cap_) {
        cudaFree(d_src_);
        d_src_ = nullptr;
        src_cap_ = nbytes;
        check(cudaMalloc(&d_src_, src_cap_), "cudaMalloc src grow");
    }
    auto t0 = std::chrono::steady_clock::now();
    check(cudaMemcpy(d_src_, src.data, nbytes, cudaMemcpyHostToDevice), "H2D");
    transfer_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();

    auto a = static_cast<cudaEvent_t>(ev_start_);
    auto b = static_cast<cudaEvent_t>(ev_stop_);
    check(cudaEventRecord(a), "cudaEventRecord a");
    const int pitch = static_cast<int>(src.step);
    if (fused_) {
        check(ct_fused_preprocess(static_cast<unsigned char*>(d_src_), src.cols, src.rows, pitch,
                                  static_cast<float*>(d_chw_), imgsz_, imgsz_, meta.new_w, meta.new_h, meta.pad_x,
                                  meta.pad_y, static_cast<unsigned char>(pad_value_), 1.f / 255.f, 0),
              "fused");
    } else {
        check(ct_letterbox_resize(static_cast<unsigned char*>(d_src_), src.cols, src.rows, pitch,
                                  static_cast<unsigned char*>(d_canvas_), imgsz_, imgsz_, meta.new_w, meta.new_h,
                                  meta.pad_x, meta.pad_y, static_cast<unsigned char>(pad_value_), 0),
              "resize");
        const int pix = imgsz_ * imgsz_;
        check(ct_bgr_to_rgb(static_cast<unsigned char*>(d_canvas_), static_cast<unsigned char*>(d_rgb_), pix, 0), "bgr");
        check(ct_normalize_hwc(static_cast<unsigned char*>(d_rgb_), static_cast<float*>(d_hwc_), pix * 3, 1.f / 255.f, 0),
              "norm");
        check(ct_hwc_to_chw(static_cast<float*>(d_hwc_), static_cast<float*>(d_chw_), imgsz_, imgsz_, 0), "layout");
    }
    check(cudaEventRecord(b), "cudaEventRecord b");
    check(cudaEventSynchronize(b), "cudaEventSynchronize");
    float ms = 0;
    check(cudaEventElapsedTime(&ms, a, b), "cudaEventElapsedTime");
    kernel_ms = ms;
    if (copy_to_host) {
        if (!out_nchw) throw std::runtime_error("CUDA preprocess D2H requested with null host pointer");
        auto t1 = std::chrono::steady_clock::now();
        check(cudaMemcpy(out_nchw, d_chw_, static_cast<size_t>(imgsz_) * imgsz_ * 3 * sizeof(float),
                         cudaMemcpyDeviceToHost),
              "D2H");
        transfer_ms += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t1).count();
    }
}

}  // namespace ct

#endif
