cmake_minimum_required(VERSION 3.18)
project(scindoPpDoclayoutV3PreprocessBackend LANGUAGES C CXX)

# Boilerplate for a Triton C++ backend, following
# triton-inference-server/backend's examples/backends/recommended/CMakeLists.txt.
#
# To copy this file for a new model's backend, change only:
#   1. BACKEND_NAME's default value below, and the ldscript filename in the
#      configure_file() call to match.
#   2. The add_library() source list.
#   3. The "model-specific dependencies" section (find_package/link calls) -
#      OpenCV/nvImageCodec/CUDA below are specific to this model's GPU image
#      decode; a different backend would list whatever it actually uses.
# Everything else (FetchContent block, ldscript wiring, install layout) is
# infrastructure shared by every backend in this repo.

set(BACKEND_NAME "pp_doclayout_v3_preprocess" CACHE STRING
    "Backend name Triton uses: install subdir under backends/, and the \
libtriton_<name>.so filename. Overridden by the Makefile's $(BACKEND).")

set(TRITON_REPO_ORGANIZATION "https://github.com/triton-inference-server" CACHE STRING
    "Git org to fetch triton-inference-server/{common,core,backend} from")

# Pinned to the branch matching this repo's base image
# (nvcr.io/nvidia/tritonserver:26.06-py3 -> "r26.06"). Intentionally not
# "main", which floats and would silently pick up ABI/API drift.
set(TRITON_COMMON_REPO_TAG "r26.06" CACHE STRING "Tag for triton-inference-server/common repo")
set(TRITON_CORE_REPO_TAG "r26.06" CACHE STRING "Tag for triton-inference-server/core repo")
set(TRITON_BACKEND_REPO_TAG "r26.06" CACHE STRING "Tag for triton-inference-server/backend repo")

set(TRITON_MIN_CXX_STANDARD 17 CACHE STRING "Minimum C++ standard")

# This backend's preprocess.cc manages nvImageCodec/CUDA resources directly
# (against the exact vendored libcudart below) and does not use
# BackendModelInstance's CUDA-stream helper. Turning this on would also make
# repo-backend enable_language(CUDA) to compile its own kernel.cu, requiring
# a fully working host nvcc toolchain unrelated to anything this backend
# actually needs. Keep it off.
option(TRITON_ENABLE_GPU "Enable GPU support in backend utilities" OFF)
option(TRITON_ENABLE_STATS "Include statistics collection in backend utilities" ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Proprietary NVIDIA binaries vendored from the scindo-models-triton image by
# `make vendor-from-image` (see Makefile). Not fetched by CMake: there is no
# buildable-from-source alternative, and pinning to the exact .so extracted
# from the deployed image (not a separately-resolved host copy) is the whole
# point of that mechanism.
set(NVIMGCODEC_ROOT "" CACHE PATH "Vendored nvImageCodec root (include/ + libnvimgcodec.so.0)")
set(CUDA_ROOT "" CACHE PATH "Vendored CUDA root (include/ + lib/libcudart.so.13)")

if(NOT EXISTS "${NVIMGCODEC_ROOT}/include/nvimgcodec.h")
  message(FATAL_ERROR
    "NVIMGCODEC_ROOT ('${NVIMGCODEC_ROOT}') missing include/nvimgcodec.h; "
    "run `make vendor-from-image` first (the Makefile's build target already depends on it).")
endif()
if(NOT EXISTS "${CUDA_ROOT}/include/cuda_runtime_api.h")
  message(FATAL_ERROR
    "CUDA_ROOT ('${CUDA_ROOT}') missing include/cuda_runtime_api.h; "
    "run `make vendor-from-image` first.")
endif()
set(NVIMGCODEC_LIB "${NVIMGCODEC_ROOT}/libnvimgcodec.so.0")
set(CUDART_LIB "${CUDA_ROOT}/lib/libcudart.so.13")

foreach(lib ${NVIMGCODEC_LIB} ${CUDART_LIB})
  if(NOT EXISTS "${lib}")
    message(FATAL_ERROR "Expected vendored library not found: ${lib}")
  endif()
endforeach()

# Triton headers/utilities. repo-core's fetched source supplies
# triton/core/tritonbackend.h + tritonserver.h (pinned to the exact release
# above, replacing the old TRITON_INCLUDE host hack); repo-backend's supplies
# triton/backend/backend_common.h and friends.
include(FetchContent)

# EXCLUDE_FROM_ALL keeps these fetched repos' own install()/test targets
# (headers, static libs, gtest binaries, pkgconfig files, ...) out of our
# `cmake --build`/`--install`, which would otherwise land inside
# ARTIFACT_ROOT alongside the one .so this project actually ships. Their
# libraries still get built because our target links them.
FetchContent_Declare(
  repo-common
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/common.git
  GIT_TAG ${TRITON_COMMON_REPO_TAG}
  GIT_SHALLOW ON
  EXCLUDE_FROM_ALL
)
FetchContent_Declare(
  repo-core
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/core.git
  GIT_TAG ${TRITON_CORE_REPO_TAG}
  GIT_SHALLOW ON
  EXCLUDE_FROM_ALL
)
FetchContent_Declare(
  repo-backend
  GIT_REPOSITORY ${TRITON_REPO_ORGANIZATION}/backend.git
  GIT_TAG ${TRITON_BACKEND_REPO_TAG}
  GIT_SHALLOW ON
  EXCLUDE_FROM_ALL
)
FetchContent_MakeAvailable(repo-common repo-core repo-backend)

# Model-specific dependencies: this model's preprocess.cc only needs these
# OpenCV modules; a different model's backend would list whatever it uses.
find_package(OpenCV REQUIRED COMPONENTS core imgproc imgcodecs)

# Shared library. The ldscript hides everything except the TRITONBACKEND_*
# ABI, same as every other Triton C++ backend.
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/src/libtriton_pp_doclayout_v3_preprocess.ldscript
  ${CMAKE_CURRENT_BINARY_DIR}/libtriton_${BACKEND_NAME}.ldscript
  COPYONLY
)

add_library(
  triton-pp-doclayout-v3-preprocess-backend SHARED
  src/backend.cc
  src/preprocess.cc
)

target_include_directories(
  triton-pp-doclayout-v3-preprocess-backend
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${NVIMGCODEC_ROOT}/include
    ${CUDA_ROOT}/include
)

target_compile_features(triton-pp-doclayout-v3-preprocess-backend PRIVATE cxx_std_${TRITON_MIN_CXX_STANDARD})
target_compile_options(
  triton-pp-doclayout-v3-preprocess-backend PRIVATE
  -O3 -Wall -Wextra -Wpedantic
)

set_target_properties(
  triton-pp-doclayout-v3-preprocess-backend PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  OUTPUT_NAME "triton_${BACKEND_NAME}"
  LINK_DEPENDS ${CMAKE_CURRENT_BINARY_DIR}/libtriton_${BACKEND_NAME}.ldscript
  LINK_FLAGS "-Wl,--version-script ${CMAKE_CURRENT_BINARY_DIR}/libtriton_${BACKEND_NAME}.ldscript"
)

target_link_libraries(
  triton-pp-doclayout-v3-preprocess-backend
  PRIVATE
    triton-core-serverapi   # from repo-core (header-only)
    triton-core-backendapi  # from repo-core (header-only)
    triton-core-serverstub  # from repo-core (link-time stub for libtritonserver.so)
    triton-backend-utils    # from repo-backend (backend_common/backend_model/backend_model_instance)
    opencv_core
    opencv_imgproc
    opencv_imgcodecs
    "${CUDART_LIB}"
    "${NVIMGCODEC_LIB}"
)

# Install. CMAKE_INSTALL_PREFIX is set by the Makefile to ARTIFACT_ROOT, so
# this lands exactly at
#   $(ARTIFACT_ROOT)/backends/$(BACKEND)/libtriton_$(BACKEND).so
# matching the pre-CMake raw-g++ Makefile's BACKEND_OUT convention exactly.
# CMAKE_INSTALL_PREFIX is baked into this DESTINATION at configure time, so
# `cmake --install --prefix X` cannot retarget it after the fact - always
# drive configure+build+install together (as the Makefile does).
install(
  TARGETS triton-pp-doclayout-v3-preprocess-backend
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/${BACKEND_NAME}
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/backends/${BACKEND_NAME}
)
