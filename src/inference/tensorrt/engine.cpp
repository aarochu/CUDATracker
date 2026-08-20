#ifdef CT_WITH_TENSORRT

#include "engine.hpp"
#include "app/types.hpp"

#include <NvInfer.h>
#include <NvInferRuntime.h>
#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cuda_runtime.h>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <vector>

namespace ct {
namespace {

void ck_cuda(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

template <class T>
void trt_release(T* p) {
    if (!p) return;
#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 10
    delete p;
#else
    p->destroy();
#endif
}

class Logger final : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) log_line("WARN", "tensorrt", msg ? msg : "");
    }
};

size_t dtype_size(nvinfer1::DataType t) {
    switch (t) {
        case nvinfer1::DataType::kFLOAT:
            return 4;
        case nvinfer1::DataType::kHALF:
            return 2;
        case nvinfer1::DataType::kINT8:
            return 1;
        case nvinfer1::DataType::kINT32:
            return 4;
        default:
            return 4;
    }
}

int64_t volume(const nvinfer1::Dims& d) {
    int64_t n = 1;
    for (int i = 0; i < d.nbDims; ++i) n *= std::max(d.d[i], 1);
    return n;
}

}  // namespace

struct TrtBackend::Impl {
    Logger logger;
    nvinfer1::IRuntime* runtime = nullptr;
    nvinfer1::ICudaEngine* engine = nullptr;
    nvinfer1::IExecutionContext* context = nullptr;
    cudaStream_t stream = nullptr;
    std::string in_name;
    std::string out_name;
    nvinfer1::Dims in_dims{};
    nvinfer1::Dims out_dims{};
    nvinfer1::DataType out_type = nvinfer1::DataType::kFLOAT;
    void* d_in = nullptr;
    void* d_out = nullptr;
    size_t in_bytes = 0;
    size_t out_bytes = 0;

    ~Impl() {
        if (d_in) cudaFree(d_in);
        if (d_out) cudaFree(d_out);
        if (stream) cudaStreamDestroy(stream);
        trt_release(context);
        trt_release(engine);
        trt_release(runtime);
    }
};

TrtBackend::TrtBackend(const std::string& engine_path) : impl_(std::make_unique<Impl>()) {
    std::ifstream in(engine_path, std::ios::binary);
    if (!in) throw std::runtime_error("cannot open TensorRT engine " + engine_path + " — run python scripts/build_engine.py");
    std::vector<char> blob((std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>());
    if (blob.empty()) throw std::runtime_error("empty TensorRT engine " + engine_path);

    auto* rt = nvinfer1::createInferRuntime(impl_->logger);
    if (!rt) throw std::runtime_error("TensorRT createInferRuntime failed");
    impl_->runtime = rt;
    auto* eng = rt->deserializeCudaEngine(blob.data(), blob.size());
    if (!eng) {
        throw std::runtime_error("Failed to deserialize " + engine_path +
                                 ". Rebuild the engine with this TensorRT version.");
    }
    impl_->engine = eng;
    auto* ctx = eng->createExecutionContext();
    if (!ctx) throw std::runtime_error("TensorRT createExecutionContext failed");
    impl_->context = ctx;
    ck_cuda(cudaStreamCreate(&impl_->stream), "cudaStreamCreate");

#if defined(NV_TENSORRT_MAJOR) && NV_TENSORRT_MAJOR >= 8
    const int n = eng->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char* name = eng->getIOTensorName(i);
        const auto mode = eng->getTensorIOMode(name);
        if (mode == nvinfer1::TensorIOMode::kINPUT && impl_->in_name.empty()) {
            impl_->in_name = name;
            impl_->in_dims = ctx->getTensorShape(name);
        } else if (mode == nvinfer1::TensorIOMode::kOUTPUT && impl_->out_name.empty()) {
            impl_->out_name = name;
            impl_->out_dims = ctx->getTensorShape(name);
            impl_->out_type = eng->getTensorDataType(name);
        }
    }
#else
    throw std::runtime_error("CUDATracker needs TensorRT 8.5+ (named I/O tensors)");
#endif
    if (impl_->in_name.empty() || impl_->out_name.empty())
        throw std::runtime_error("TensorRT engine has no input/output tensor");
    for (int i = 0; i < impl_->in_dims.nbDims; ++i) {
        if (impl_->in_dims.d[i] < 0)
            throw std::runtime_error("dynamic-shape TensorRT engines are not supported in v1; re-export with dynamic=False");
    }
    impl_->in_bytes = static_cast<size_t>(volume(impl_->in_dims)) * sizeof(float);
    impl_->out_bytes = static_cast<size_t>(volume(impl_->out_dims)) * dtype_size(impl_->out_type);
    ck_cuda(cudaMalloc(&impl_->d_in, impl_->in_bytes), "cudaMalloc TRT input");
    ck_cuda(cudaMalloc(&impl_->d_out, impl_->out_bytes), "cudaMalloc TRT output");
    log_line("INFO", "infer", "TensorRT engine " + engine_path);
}

TrtBackend::~TrtBackend() = default;

std::vector<cv::Mat> TrtBackend::infer(const cv::Mat& nchw) {
    if (!nchw.isContinuous()) throw std::runtime_error("TRT input must be continuous");
    const size_t nbytes = nchw.total() * nchw.elemSize();
    if (nbytes > impl_->in_bytes) throw std::runtime_error("TRT input larger than engine binding");
    ck_cuda(cudaMemcpyAsync(impl_->d_in, nchw.ptr<float>(), nbytes, cudaMemcpyHostToDevice, impl_->stream),
            "H2D TRT input");
    return infer_device(static_cast<float*>(impl_->d_in));
}

std::vector<cv::Mat> TrtBackend::infer_device(float* device_nchw) {
    auto* ctx = impl_->context;
    if (!ctx->setTensorAddress(impl_->in_name.c_str(), device_nchw))
        throw std::runtime_error("setTensorAddress input failed");
    if (!ctx->setTensorAddress(impl_->out_name.c_str(), impl_->d_out))
        throw std::runtime_error("setTensorAddress output failed");
    if (!ctx->enqueueV3(impl_->stream)) throw std::runtime_error("TensorRT enqueueV3 failed");
    ck_cuda(cudaStreamSynchronize(impl_->stream), "TRT stream sync");

    std::vector<int> sizes;
    sizes.reserve(impl_->out_dims.nbDims);
    for (int i = 0; i < impl_->out_dims.nbDims; ++i) sizes.push_back(std::max(impl_->out_dims.d[i], 1));
    cv::Mat host(static_cast<int>(sizes.size()), sizes.data(), CV_32F);
    if (impl_->out_type == nvinfer1::DataType::kFLOAT) {
        ck_cuda(cudaMemcpy(host.ptr<float>(), impl_->d_out, impl_->out_bytes, cudaMemcpyDeviceToHost), "D2H TRT out");
    } else if (impl_->out_type == nvinfer1::DataType::kHALF) {
        std::vector<uint16_t> half(static_cast<size_t>(volume(impl_->out_dims)));
        ck_cuda(cudaMemcpy(half.data(), impl_->d_out, impl_->out_bytes, cudaMemcpyDeviceToHost), "D2H TRT half");
        // IEEE-754 binary16 → float32 (ignores denormals; good enough for boxes)
        auto* dst = host.ptr<float>();
        for (size_t i = 0; i < half.size(); ++i) {
            const uint16_t h = half[i];
            const uint32_t sign = (h & 0x8000u) << 16;
            const uint32_t exp = (h >> 10) & 0x1f;
            const uint32_t man = h & 0x3ff;
            uint32_t f;
            if (exp == 0) {
                f = sign;
            } else if (exp == 31) {
                f = sign | 0x7f800000u | (man << 13);
            } else {
                f = sign | ((exp + (127 - 15)) << 23) | (man << 13);
            }
            std::memcpy(dst + i, &f, 4);
        }
    } else {
        throw std::runtime_error("unsupported TensorRT output dtype");
    }
    return {host};
}

}  // namespace ct

#endif
