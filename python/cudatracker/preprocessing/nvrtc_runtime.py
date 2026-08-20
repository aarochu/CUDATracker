from __future__ import annotations

import ctypes
import ctypes.util
import os
from pathlib import Path
from typing import Any

from cudatracker.logutil import log
from cudatracker.preprocessing.kernel_source import find_nvrtc_dll, kernel_source


NVRTC_SUCCESS = 0
CUDA_SUCCESS = 0


def _load_driver() -> ctypes.CDLL:
    names = ["nvcuda.dll", "libcuda.so.1", "libcuda.so"]
    last = None
    for name in names:
        try:
            lib = ctypes.CDLL(name)
            return lib
        except OSError as exc:
            last = exc
    path = ctypes.util.find_library("cuda")
    if path:
        return ctypes.CDLL(path)
    raise RuntimeError(f"Could not load CUDA driver library: {last}")


class Nvrtc:
    def __init__(self) -> None:
        self.lib = ctypes.CDLL(find_nvrtc_dll())
        self.lib.nvrtcCreateProgram.argtypes = [
            ctypes.c_void_p,
            ctypes.c_char_p,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        self.lib.nvrtcCreateProgram.restype = ctypes.c_int
        self.lib.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p]
        self.lib.nvrtcCompileProgram.restype = ctypes.c_int
        self.lib.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.nvrtcGetProgramLogSize.restype = ctypes.c_int
        self.lib.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.nvrtcGetProgramLog.restype = ctypes.c_int
        self.lib.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        self.lib.nvrtcGetPTXSize.restype = ctypes.c_int
        self.lib.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.nvrtcGetPTX.restype = ctypes.c_int
        self.lib.nvrtcDestroyProgram.argtypes = [ctypes.c_void_p]
        self.lib.nvrtcDestroyProgram.restype = ctypes.c_int

    def _check(self, err: int, prog: Any, what: str) -> None:
        if err == NVRTC_SUCCESS:
            return
        log_msg = ""
        if prog:
            sz = ctypes.c_size_t()
            self.lib.nvrtcGetProgramLogSize(prog, ctypes.byref(sz))
            buf = ctypes.create_string_buffer(sz.value)
            self.lib.nvrtcGetProgramLog(prog, buf)
            log_msg = buf.value.decode("utf-8", errors="replace")
        raise RuntimeError(f"NVRTC {what} failed ({err}): {log_msg}")

    def compile_ptx(self, source: str, arch: str) -> bytes:
        prog = ctypes.c_void_p()
        src = source.encode("utf-8")
        err = self.lib.nvrtcCreateProgram(
            ctypes.byref(prog), src, b"ct_kernels.cu", 0, None, None
        )
        self._check(err, prog, "create")
        opts = [
            b"--gpu-architecture=" + arch.encode("utf-8"),
            b"--std=c++17",
            b"-DCT_NVRTC=1",
            b"--use_fast_math",
        ]
        arr = (ctypes.c_char_p * len(opts))(*opts)
        err = self.lib.nvrtcCompileProgram(prog, len(opts), arr)
        if err != NVRTC_SUCCESS:
            self._check(err, prog, "compile")
        sz = ctypes.c_size_t()
        self.lib.nvrtcGetPTXSize(prog, ctypes.byref(sz))
        ptx = ctypes.create_string_buffer(sz.value)
        self.lib.nvrtcGetPTX(prog, ptx)
        self.lib.nvrtcDestroyProgram(ctypes.byref(prog))
        return bytes(ptx.raw[: sz.value])


class CudaKernels:
    """Compile project .cu kernels with NVRTC and launch them on the current device."""

    def __init__(self) -> None:
        import torch

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA preprocess needs a CUDA GPU.")
        torch.cuda.init()
        _ = torch.empty(1, device="cuda")
        major, minor = torch.cuda.get_device_capability()
        arch = f"compute_{major}{minor}"
        self.driver = _load_driver()
        self._bind_driver()
        self._check_cu(self.driver.cuInit(0), "cuInit")
        self._ensure_context()
        ptx = Nvrtc().compile_ptx(kernel_source(), arch)
        image = ctypes.create_string_buffer(ptx if ptx.endswith(b"\0") else ptx + b"\0")
        self.module = ctypes.c_void_p()
        self._check_cu(self.driver.cuModuleLoadData(ctypes.byref(self.module), image), "cuModuleLoadData")
        self.fn_resize = self._fn("ct_letterbox_resize_kernel")
        self.fn_bgr = self._fn("ct_bgr_to_rgb_kernel")
        self.fn_norm = self._fn("ct_normalize_hwc_kernel")
        self.fn_layout = self._fn("ct_hwc_to_chw_kernel")
        self.fn_fused = self._fn("ct_fused_preprocess_kernel")
        log("INFO", "cuda", f"NVRTC kernels loaded for {arch} ({torch.cuda.get_device_name(0)})")

    def _bind_driver(self) -> None:
        d = self.driver
        d.cuInit.argtypes = [ctypes.c_uint]
        d.cuInit.restype = ctypes.c_int
        d.cuModuleLoadData.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        d.cuModuleLoadData.restype = ctypes.c_int
        d.cuModuleGetFunction.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_char_p]
        d.cuModuleGetFunction.restype = ctypes.c_int
        d.cuLaunchKernel.argtypes = [
            ctypes.c_void_p,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_uint,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
        ]
        d.cuLaunchKernel.restype = ctypes.c_int
        d.cuCtxGetCurrent.argtypes = [ctypes.c_void_p]
        d.cuCtxGetCurrent.restype = ctypes.c_int
        d.cuDeviceGet.argtypes = [ctypes.c_void_p, ctypes.c_int]
        d.cuDeviceGet.restype = ctypes.c_int
        d.cuDevicePrimaryCtxRetain.argtypes = [ctypes.c_void_p, ctypes.c_int]
        d.cuDevicePrimaryCtxRetain.restype = ctypes.c_int
        d.cuCtxSetCurrent.argtypes = [ctypes.c_void_p]
        d.cuCtxSetCurrent.restype = ctypes.c_int

    def _ensure_context(self) -> None:
        ctx = ctypes.c_void_p()
        self._check_cu(self.driver.cuCtxGetCurrent(ctypes.byref(ctx)), "cuCtxGetCurrent")
        if ctx:
            return
        dev = ctypes.c_int()
        self._check_cu(self.driver.cuDeviceGet(ctypes.byref(dev), 0), "cuDeviceGet")
        self._check_cu(self.driver.cuDevicePrimaryCtxRetain(ctypes.byref(ctx), dev), "cuDevicePrimaryCtxRetain")
        self._check_cu(self.driver.cuCtxSetCurrent(ctx), "cuCtxSetCurrent")

    def _check_cu(self, err: int, what: str) -> None:
        if err != CUDA_SUCCESS:
            raise RuntimeError(f"{what} failed with CUDA driver error {err}")

    def _fn(self, name: str) -> ctypes.c_void_p:
        fn = ctypes.c_void_p()
        self._check_cu(
            self.driver.cuModuleGetFunction(ctypes.byref(fn), self.module, name.encode("utf-8")),
            f"cuModuleGetFunction({name})",
        )
        return fn

    def launch(self, fn: ctypes.c_void_p, grid: tuple[int, int, int], block: tuple[int, int, int], args: list[Any], stream: int) -> None:
        holders = []
        for a in args:
            if isinstance(a, ctypes._SimpleCData):
                holders.append(a)
            elif isinstance(a, float):
                holders.append(ctypes.c_float(a))
            else:
                holders.append(ctypes.c_int(int(a)))
        void_p_arr = (ctypes.c_void_p * len(holders))(
            *[ctypes.cast(ctypes.byref(h), ctypes.c_void_p) for h in holders]
        )
        err = self.driver.cuLaunchKernel(
            fn,
            grid[0],
            grid[1],
            grid[2],
            block[0],
            block[1],
            block[2],
            0,
            ctypes.c_void_p(stream),
            void_p_arr,
            None,
        )
        self._check_cu(err, "cuLaunchKernel")


def _c_ptr(tensor) -> ctypes.c_uint64:
    return ctypes.c_uint64(int(tensor.data_ptr()))


def _c_i(v: int) -> ctypes.c_int:
    return ctypes.c_int(int(v))


def _c_u8(v: int) -> ctypes.c_uint8:
    return ctypes.c_uint8(int(v))


def _c_f(v: float) -> ctypes.c_float:
    return ctypes.c_float(float(v))


def grid_1d(n: int, block: int = 256) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return ((n + block - 1) // block, 1, 1), (block, 1, 1)


def grid_2d(w: int, h: int, block: int = 16) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return ((w + block - 1) // block, (h + block - 1) // block, 1), (block, block, 1)
