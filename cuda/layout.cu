#ifndef CT_NVRTC
#include "kernels.cuh"
#include <cuda_runtime.h>
#endif

// float32 HWC (H,W,3) -> float32 CHW (3,H,W).
extern "C" __global__ void ct_hwc_to_chw_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int h,
    int w) {
    const int x = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int y = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    if (x >= w || y >= h) {
        return;
    }
    const int hw = h * w;
    const int hwc = (y * w + x) * 3;
    const int idx = y * w + x;
    dst[0 * hw + idx] = src[hwc + 0];
    dst[1 * hw + idx] = src[hwc + 1];
    dst[2 * hw + idx] = src[hwc + 2];
}

#ifndef CT_NVRTC
extern "C" cudaError_t ct_hwc_to_chw(
    const float* src,
    float* dst,
    int h,
    int w,
    cudaStream_t stream) {
    dim3 block(CT_BLOCK_2D, CT_BLOCK_2D);
    dim3 grid((w + CT_BLOCK_2D - 1) / CT_BLOCK_2D, (h + CT_BLOCK_2D - 1) / CT_BLOCK_2D);
    ct_hwc_to_chw_kernel<<<grid, block, 0, stream>>>(src, dst, h, w);
    return cudaGetLastError();
}
#endif
