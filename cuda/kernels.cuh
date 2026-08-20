#pragma once

// Shared device helpers for CUDATracker preprocess kernels.
// Launch geometry: 16x16 threads for 2D (resize, fused), 256 threads for 1D (color, normalize, layout).

#ifndef CT_BLOCK_2D
#define CT_BLOCK_2D 16
#endif
#ifndef CT_BLOCK_1D
#define CT_BLOCK_1D 256
#endif

#ifdef __CUDACC__

__device__ __forceinline__ float ct_clampf(float x, float lo, float hi) {
    return fminf(hi, fmaxf(lo, x));
}

__device__ __forceinline__ unsigned char ct_u8_from_f(float x) {
    const int v = __float2int_rn(ct_clampf(x, 0.f, 255.f));
    return static_cast<unsigned char>(v);
}

// Matches OpenCV INTER_LINEAR mapping: src = (dst + 0.5) * scale - 0.5
// Coordinates are clamped (OpenCV saturates); returning 0 here broke CPU vs CUDA
// equivalence on upscales where (dst+0.5)*scale-0.5 goes slightly negative.
__device__ __forceinline__ void ct_bilinear_bgr(
    const unsigned char* src,
    int sw,
    int sh,
    int src_pitch,
    float fx,
    float fy,
    unsigned char* bgr) {
    fx = ct_clampf(fx, 0.f, (float)(sw - 1));
    fy = ct_clampf(fy, 0.f, (float)(sh - 1));
    const int x0 = (int)floorf(fx);
    const int y0 = (int)floorf(fy);
    const int x1 = (x0 + 1 < sw) ? x0 + 1 : (sw - 1);
    const int y1 = (y0 + 1 < sh) ? y0 + 1 : (sh - 1);
    const float ax = fx - (float)x0;
    const float ay = fy - (float)y0;
    const float w00 = (1.f - ax) * (1.f - ay);
    const float w10 = ax * (1.f - ay);
    const float w01 = (1.f - ax) * ay;
    const float w11 = ax * ay;
    const int i00 = y0 * src_pitch + x0 * 3;
    const int i10 = y0 * src_pitch + x1 * 3;
    const int i01 = y1 * src_pitch + x0 * 3;
    const int i11 = y1 * src_pitch + x1 * 3;
    for (int c = 0; c < 3; ++c) {
        const float v = w00 * src[i00 + c] + w10 * src[i10 + c] +
                        w01 * src[i01 + c] + w11 * src[i11 + c];
        bgr[c] = ct_u8_from_f(v);
    }
}

#endif
