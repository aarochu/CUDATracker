#include "app/bench_io.hpp"
#include "app/types.hpp"
#include "capture/camera.hpp"
#include "detection/detector.hpp"
#include "preprocessing/preprocess.hpp"
#include "telemetry/profiler.hpp"
#include "tracking/tracker.hpp"
#include "visualization/renderer.hpp"

#ifdef CT_WITH_TENSORRT
#include "inference/tensorrt/engine.hpp"
#endif

#include <algorithm>
#include <chrono>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
#include <opencv2/highgui.hpp>

using namespace ct;

int main(int argc, char** argv) {
    std::string cfg_path = "configs/default.yaml";
    std::string source_override;
    std::string pre_override;
    std::string backend_override;
    std::string precision_override;
    bool bench = false;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        auto next = [&]() {
            if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
            return std::string(argv[++i]);
        };
        if (a == "--config") cfg_path = next();
        else if (a == "--source") source_override = next();
        else if (a == "--preprocess") pre_override = next();
        else if (a == "--backend") backend_override = next();
        else if (a == "--precision") precision_override = next();
        else if (a == "--bench") bench = true;
        else if (a == "--help") {
            std::cout << "cudatracker --config configs/default.yaml [--source f] [--preprocess cpu|cuda] "
                         "[--backend onnx|tensorrt] [--precision fp32|fp16] [--bench]\n";
            return 0;
        }
    }
    AppConfig cfg;
    try {
        cfg = load_config(cfg_path);
    } catch (const std::exception& e) {
        log_line("ERROR", "app", e.what());
        return 2;
    }
    if (!source_override.empty()) cfg.source = source_override;
    if (!pre_override.empty()) cfg.preprocess = pre_override;
    if (!precision_override.empty()) cfg.precision = precision_override;
    if (!backend_override.empty()) cfg.backend = backend_override;
    if (bench) cfg.vis = false;

    if (cfg.backend == "pytorch") {
        log_line("WARN", "app", "C++ binary has no PyTorch. Using OpenCV DNN ONNX. python -m cudatracker for PyTorch.");
        cfg.backend = "onnx";
    }
    if (cfg.backend == "tensorrt") {
#ifndef CT_WITH_TENSORRT
        log_line("ERROR", "app", "binary built without TensorRT. cmake -DWITH_TENSORRT=ON or use python -m cudatracker");
        return 2;
#endif
    } else if (cfg.backend != "onnx") {
        log_line("ERROR", "app", "unknown C++ backend '" + cfg.backend + "'. Use onnx or tensorrt.");
        return 2;
    }

    Capture cap;
    if (!cap.open(cfg, bench)) return 2;

    std::unique_ptr<IInferBackend> infer;
    try {
#ifdef CT_WITH_TENSORRT
        if (cfg.backend == "tensorrt") {
            std::string eng = cfg.engine;
            if (cfg.precision == "fp32" && !cfg.engine_fp32.empty()) eng = cfg.engine_fp32;
            if (cfg.precision == "fp16" && !cfg.engine_fp16.empty()) eng = cfg.engine_fp16;
            infer = std::make_unique<TrtBackend>(eng, cfg.imgsz);
        } else
#endif
        {
            infer = std::make_unique<OpenCvDnnBackend>(cfg.onnx, cfg.device == "gpu");
        }
    } catch (const std::exception& e) {
        log_line("ERROR", "infer", e.what());
        return 2;
    }

    SortTracker tracker(cfg.max_age, cfg.min_hits, cfg.track_iou);
#ifdef CT_WITH_CUDA
    std::unique_ptr<CudaPreprocessor> cuda_pre;
    try {
        if (cfg.preprocess == "cuda") {
            cuda_pre = std::make_unique<CudaPreprocessor>(cfg.imgsz, cfg.pad_value, cfg.fused, cfg.letterbox);
        }
    } catch (const std::exception& e) {
        log_line("ERROR", "preprocess", e.what());
        return 2;
    }
#endif
    if (cfg.preprocess == "cuda") {
#ifndef CT_WITH_CUDA
        log_line("ERROR", "app", "binary built without CUDA");
        return 2;
#endif
    }

    const bool trt_device =
#ifdef CT_WITH_TENSORRT
        cfg.backend == "tensorrt";
#else
        false;
#endif

    int frames = 0;
    const int limit = bench ? (cfg.warmup_frames + cfg.measure_frames) : 1'000'000'000;
    std::vector<StageTimes> measured;
    try {
        while (frames < limit) {
            auto t0 = std::chrono::steady_clock::now();
            cv::Mat bgr;
            std::uint64_t ts = 0, idx = 0;
            auto t_cap0 = std::chrono::steady_clock::now();
            if (!cap.read(bgr, ts, idx)) break;
            double cap_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_cap0).count();

            LetterboxMeta meta;
            cv::Mat nchw;
            double pre_ms = 0, xfer_ms = 0;
            auto t_pre = std::chrono::steady_clock::now();
            std::vector<cv::Mat> raw;
#ifdef CT_WITH_CUDA
            if (cuda_pre) {
                const bool copy_host = !trt_device;
                if (copy_host) {
                    const int sizes[4] = {1, 3, cfg.imgsz, cfg.imgsz};
                    nchw.create(4, sizes, CV_32F);
                }
                double kms = 0;
                cuda_pre->run(bgr, copy_host ? nchw.ptr<float>() : nullptr, meta, copy_host, xfer_ms, kms);
                pre_ms = kms;
            } else
#endif
            {
                cpu_preprocess(bgr, cfg.imgsz, cfg.letterbox, cfg.pad_value, nchw, meta);
                pre_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_pre).count();
            }
            auto t_inf = std::chrono::steady_clock::now();
#ifdef CT_WITH_CUDA
            if (cuda_pre && trt_device) {
                raw = infer->infer_device(cuda_pre->device_nchw());
            } else
#endif
            {
                raw = infer->infer(nchw);
            }
            double inf_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_inf).count();
            auto t_post = std::chrono::steady_clock::now();
            auto dets = decode_yolo(raw, meta, cfg.conf, cfg.iou);
            double post_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_post).count();
            auto t_tr = std::chrono::steady_clock::now();
            auto tracks = tracker.update(dets);
            double tr_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_tr).count();
            StageTimes times;
            times.capture_ms = cap_ms;
            times.preprocess_ms = pre_ms;
            times.transfer_ms = xfer_ms;
            times.inference_ms = inf_ms;
            times.postprocess_ms = post_ms;
            times.tracking_ms = tr_ms;
            times.end_to_end_ms =
                std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
            if (cfg.vis) {
                auto t_r = std::chrono::steady_clock::now();
                auto vis = draw_overlay(bgr, tracks, times, 1000.0 / std::max(times.end_to_end_ms, 0.001), cfg, cfg.hud,
                                        cfg.trails, cfg.hardware_panel ? hardware_line() : "");
                cv::imshow("CUDATracker", vis);
                int key = cv::waitKey(1) & 0xFF;
                times.render_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_r).count();
                times.end_to_end_ms =
                    std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
                if (key == 'q') break;
                if (key == 'h') cfg.hardware_panel = !cfg.hardware_panel;
                if (key == 't') cfg.trails = !cfg.trails;
                if (key == 'd') cfg.hud = !cfg.hud;
                if (key == 'p' || key == '1' || key == '2') {
                    log_line("INFO", "app",
                             "C++ live reload of preprocess/precision is not wired; pass --preprocess / --precision");
                }
            }
            if (bench && frames >= cfg.warmup_frames) measured.push_back(times);
            if (++frames % 30 == 0) {
                log_line("INFO", "perf",
                         "e2e=" + std::to_string(times.end_to_end_ms) + "ms infer=" + std::to_string(inf_ms));
            }
        }
    } catch (const std::exception& e) {
        log_line("ERROR", "app", e.what());
        cap.close();
        return 2;
    }
    cap.close();
    if (bench) {
        if (measured.empty()) {
            log_line("ERROR", "bench", "no measure frames (source ended before warmup finished)");
            return 2;
        }
        write_bench_artifacts(cfg, measured, hardware_line());
    }
    return 0;
}
