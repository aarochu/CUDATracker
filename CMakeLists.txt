cmake_minimum_required(VERSION 3.18)
project(CUDATracker LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(WITH_CUDA "Build CUDA preprocess kernels" ON)
option(WITH_TENSORRT "Build native TensorRT backend" OFF)
option(WITH_ONNXRUNTIME "Build ONNX Runtime backend" OFF)

if(WITH_CUDA)
  include(CheckLanguage)
  check_language(CUDA)
  if(NOT CMAKE_CUDA_COMPILER)
    message(WARNING "CUDA compiler not found; building without CUDA kernels")
    set(WITH_CUDA OFF)
  endif()
endif()

find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs videoio dnn highgui)

add_library(ct_core
  src/app/config.cpp
  src/app/log.cpp
  src/capture/camera.cpp
  src/preprocessing/cpu_preprocess.cpp
  src/detection/detector.cpp
  src/tracking/tracker.cpp
  src/telemetry/profiler.cpp
  src/telemetry/gpu_monitor.cpp
  src/visualization/renderer.cpp
  src/app/bench_io.cpp
)
target_include_directories(ct_core PUBLIC src cuda)
target_link_libraries(ct_core PUBLIC ${OpenCV_LIBS})

if(WITH_CUDA)
  enable_language(CUDA)
  find_package(CUDAToolkit REQUIRED)
  target_sources(ct_core PRIVATE
    src/preprocessing/cuda_preprocess.cpp
    cuda/resize.cu
    cuda/colorspace.cu
    cuda/normalize.cu
    cuda/layout.cu
    cuda/fused_preprocess.cu
  )
  target_compile_definitions(ct_core PUBLIC CT_WITH_CUDA=1)
  target_link_libraries(ct_core PUBLIC CUDA::cudart)
  set_target_properties(ct_core PROPERTIES CUDA_ARCHITECTURES "75;80;86;87;89;90")
endif()

if(WITH_TENSORRT)
  find_package(CUDAToolkit REQUIRED)
  find_path(TENSORRT_INCLUDE NvInfer.h)
  find_library(TENSORRT_LIB nvinfer)
  if(NOT TENSORRT_INCLUDE OR NOT TENSORRT_LIB)
    message(FATAL_ERROR "WITH_TENSORRT=ON but TensorRT was not found")
  endif()
  target_sources(ct_core PRIVATE src/inference/tensorrt/engine.cpp)
  target_include_directories(ct_core PUBLIC ${TENSORRT_INCLUDE})
  target_link_libraries(ct_core PUBLIC ${TENSORRT_LIB} CUDA::cudart)
  target_compile_definitions(ct_core PUBLIC CT_WITH_TENSORRT=1)
endif()

add_executable(cudatracker src/main.cpp)
target_link_libraries(cudatracker PRIVATE ct_core)

install(TARGETS cudatracker RUNTIME DESTINATION bin)
