#ifndef CT_NVRTC
#include "kernels.cuh"
#include <cuda_runtime.h>
#endif

// Phase 10 fused path: letterbox bilinear + BGR->RGB + /255 + HWC->CHW in one launch.
// Off by default. Turn on with preprocess.fused: true after you measure the four-kernel path.
extern "C" __global__ void ct_fused_preprocess_kernel(
    const unsigned char* __restrict__ src,
    int sw,
    int sh,
    int src_pitch,
    float* __restrict__ dst_chw,
    int dw,
    int dh,
    int new_w,
    int new_h,
    int pad_x,
    int pad_y,
    unsigned char pad_val,
    float scale) {
    const int x = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int y = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    if (x >= dw || y >= dh) {
        return;
    }
    unsigned char bgr[3];
    if (x < pad_x || y < pad_y || x >= pad_x + new_w || y >= pad_y + new_h) {
        bgr[0] = pad_val;
        bgr[1] = pad_val;
        bgr[2] = pad_val;
    } else {
        const float scale_x = (float)sw / (float)new_w;
        const float scale_y = (float)sh / (float)new_h;
        const float fx = ((float)(x - pad_x) + 0.5f) * scale_x - 0.5f;
        const float fy = ((float)(y - pad_y) + 0.5f) * scale_y - 0.5f;
        ct_bilinear_bgr(src, sw, sh, src_pitch, fx, fy, bgr);
    }
    const int hw = dh * dw;
    const int idx = y * dw + x;
    dst_chw[0 * hw + idx] = (float)bgr[2] * scale;  // R
    dst_chw[1 * hw + idx] = (float)bgr[1] * scale;  // G
    dst_chw[2 * hw + idx] = (float)bgr[0] * scale;  // B
}

#ifndef CT_NVRTC
extern "C" cudaError_t ct_fused_preprocess(
    const unsigned char* src,
    int sw,
    int sh,
    int src_pitch,
    float* dst_chw,
    int dw,
    int dh,
    int new_w,
    int new_h,
    int pad_x,
    int pad_y,
    unsigned char pad_val,
    float scale,
    cudaStream_t stream) {
    dim3 block(CT_BLOCK_2D, CT_BLOCK_2D);
    dim3 grid((dw + CT_BLOCK_2D - 1) / CT_BLOCK_2D, (dh + CT_BLOCK_2D - 1) / CT_BLOCK_2D);
    ct_fused_preprocess_kernel<<<grid, block, 0, stream>>>(
        src, sw, sh, src_pitch, dst_chw, dw, dh, new_w, new_h, pad_x, pad_y, pad_val, scale);
    return cudaGetLastError();
}
#endif
