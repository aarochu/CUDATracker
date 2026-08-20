#ifndef CT_NVRTC
#include "kernels.cuh"
#include <cuda_runtime.h>
#endif

// Packed RGB uint8 HWC -> float32 HWC, multiplied by scale (1/255).
extern "C" __global__ void ct_normalize_hwc_kernel(
    const unsigned char* __restrict__ src,
    float* __restrict__ dst,
    int n_values,
    float scale) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n_values) {
        return;
    }
    dst[i] = (float)src[i] * scale;
}

#ifndef CT_NVRTC
extern "C" cudaError_t ct_normalize_hwc(
    const unsigned char* src,
    float* dst,
    int n_values,
    float scale,
    cudaStream_t stream) {
    const int block = CT_BLOCK_1D;
    const int grid = (n_values + block - 1) / block;
    ct_normalize_hwc_kernel<<<grid, block, 0, stream>>>(src, dst, n_values, scale);
    return cudaGetLastError();
}
#endif
