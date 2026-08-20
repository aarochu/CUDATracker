#ifndef CT_NVRTC
#include "kernels.cuh"
#include <cuda_runtime.h>
#endif

// Packed BGR uint8 -> packed RGB uint8 (in-place or out-of-place).
extern "C" __global__ void ct_bgr_to_rgb_kernel(
    const unsigned char* __restrict__ src,
    unsigned char* __restrict__ dst,
    int n_pixels) {
    const int i = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (i >= n_pixels) {
        return;
    }
    const int o = i * 3;
    const unsigned char b = src[o + 0];
    const unsigned char g = src[o + 1];
    const unsigned char r = src[o + 2];
    dst[o + 0] = r;
    dst[o + 1] = g;
    dst[o + 2] = b;
}

#ifndef CT_NVRTC
extern "C" cudaError_t ct_bgr_to_rgb(
    const unsigned char* src,
    unsigned char* dst,
    int n_pixels,
    cudaStream_t stream) {
    const int block = CT_BLOCK_1D;
    const int grid = (n_pixels + block - 1) / block;
    ct_bgr_to_rgb_kernel<<<grid, block, 0, stream>>>(src, dst, n_pixels);
    return cudaGetLastError();
}
#endif
