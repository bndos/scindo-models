MODEL_NAME ?= pp_doclayout_v3_preprocess
VERSION ?= 1
OUT ?= build/$(MODEL_NAME)
BACKEND ?= pp_doclayout_v3_preprocess
CXX ?= g++
TRITON_INCLUDE ?= /opt/tritonserver/include
PYTHON ?= python3

NVIMGCODEC_ROOT := $(shell $(PYTHON) -c "import nvidia.nvimgcodec, os; print(os.path.dirname(nvidia.nvimgcodec.__file__))" 2>/dev/null)
CUDA_ROOT := $(shell $(PYTHON) -c "import nvidia.cu13; print(nvidia.cu13.__path__[0])" 2>/dev/null)

OPENCV_CFLAGS := $(shell pkg-config --cflags opencv4)
OPENCV_LIBS := $(shell pkg-config --libs opencv4)
CXXFLAGS ?= -O3 -std=c++17 -Wall -Wextra -Wpedantic -fPIC
ARTIFACT_ROOT ?= $(abspath $(OUT)/../..)
MODEL_REPOSITORY_ROOT ?= $(abspath $(OUT)/..)
BACKEND_OUT := $(ARTIFACT_ROOT)/backends/$(BACKEND)

.PHONY: build clean

build:
	@test -n "$(OUT)"
	@test -n "$(MODEL_NAME)"
	@test -n "$(NVIMGCODEC_ROOT)" || { echo "nvidia-nvimgcodec-cu13 not found; pip install -e .[triton-gpu-decode]"; exit 1; }
	@test -n "$(CUDA_ROOT)" || { echo "nvidia-cuda-runtime not found; pip install -e .[triton-gpu-decode]"; exit 1; }
	@mkdir -p "$(OUT)/$(VERSION)"
	@mkdir -p "$(BACKEND_OUT)"
	@sed "s/@MODEL_NAME@/$(MODEL_NAME)/g" config.pbtxt.in > "$(OUT)/config.pbtxt"
	$(CXX) $(CXXFLAGS) -I"$(TRITON_INCLUDE)" -I"$(NVIMGCODEC_ROOT)/include" -I"$(CUDA_ROOT)/include" $(OPENCV_CFLAGS) \
		-shared src/backend.cc src/preprocess.cc -o "$(BACKEND_OUT)/libtriton_$(BACKEND).so" \
		-L"$(CUDA_ROOT)/lib" -l:libcudart.so.13 -L"$(NVIMGCODEC_ROOT)" -l:libnvimgcodec.so.0 \
		$(OPENCV_LIBS)

clean:
	rm -rf build
