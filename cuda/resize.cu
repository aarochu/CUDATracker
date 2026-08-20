#ifndef CT_NVRTC
#include "kernels.cuh"
#include <cuda_runtime.h>
#endif

// Letterbox + bilinear resize. Destination is a packed BGR imgsz x imgsz canvas.
// Content is placed at (pad_x, pad_y) with size (new_w, new_h). Outside pixels get pad_val.
extern "C" __global__ void ct_letterbox_resize_kernel(
    const unsigned char* __restrict__ src,
    int sw,
    int sh,
    int src_pitch,
    unsigned char* __restrict__ dst,
    int dw,
    int dh,
    int new_w,
    int new_h,
    int pad_x,
    int pad_y,
    unsigned char pad_val) {
    const int x = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    const int y = (int)blockIdx.y * (int)blockDim.y + (int)threadIdx.y;
    if (x >= dw || y >= dh) {
        return;
    }
    unsigned char* out = dst + (y * dw + x) * 3;
    if (x < pad_x || y < pad_y || x >= pad_x + new_w || y >= pad_y + new_h) {
        out[0] = pad_val;
        out[1] = pad_val;
        out[2] = pad_val;
        return;
    }
    const float scale_x = (float)sw / (float)new_w;
    const float scale_y = (float)sh / (float)new_h;
    const float fx = ((float)(x - pad_x) + 0.5f) * scale_x - 0.5f;
    const float fy = ((float)(y - pad_y) + 0.5f) * scale_y - 0.5f;
    unsigned char bgr[3];
    ct_bilinear_bgr(src, sw, sh, src_pitch, fx, fy, bgr);
    out[0] = bgr[0];
    out[1] = bgr[1];
    out[2] = bgr[2];
}

#ifndef CT_NVRTC
extern "C" cudaError_t ct_letterbox_resize(
    const unsigned char* src,
    int sw,
    int sh,
    int src_pitch,
    unsigned char* dst,
    int dw,
    int dh,
    int new_w,
    int new_h,
    int pad_x,
    int pad_y,
    unsigned char pad_val,
    cudaStream_t stream) {
    dim3 block(CT_BLOCK_2D, CT_BLOCK_2D);
    dim3 grid((dw + CT_BLOCK_2D - 1) / CT_BLOCK_2D, (dh + CT_BLOCK_2D - 1) / CT_BLOCK_2D);
    ct_letterbox_resize_kernel<<<grid, block, 0, stream>>>(
        src, sw, sh, src_pitch, dst, dw, dh, new_w, new_h, pad_x, pad_y, pad_val);
    return cudaGetLastError();
}
#endif
