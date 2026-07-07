MODEL_NAME ?= pp_doclayout_v3_preprocess
VERSION ?= 1
OUT ?= build/$(MODEL_NAME)
BACKEND ?= pp_doclayout_v3_preprocess
CMAKE ?= cmake
CMAKE_BUILD_DIR ?= build/cmake

TRITON_IMAGE ?= scindo-models-triton:26.06
FROM_IMAGE_CACHE ?= $(abspath $(CURDIR)/../../../../.cache/from-image)
NVIMGCODEC_ROOT ?= $(FROM_IMAGE_CACHE)/nvimgcodec
CUDA_ROOT ?= $(FROM_IMAGE_CACHE)/cuda

ARTIFACT_ROOT ?= $(abspath $(OUT)/../..)
MODEL_REPOSITORY_ROOT ?= $(abspath $(OUT)/..)
BACKEND_OUT := $(ARTIFACT_ROOT)/backends/$(BACKEND)

.PHONY: build clean vendor-from-image

vendor-from-image:
	@if [ ! -f "$(NVIMGCODEC_ROOT)/include/nvimgcodec.h" ]; then \
		docker image inspect "$(TRITON_IMAGE)" >/dev/null 2>&1 || { \
			echo "Image $(TRITON_IMAGE) not found; build it first:"; \
			echo "  docker build -f docker/Dockerfile.triton -t $(TRITON_IMAGE) ."; \
			exit 1; \
		}; \
		echo "Extracting nvImageCodec/CUDA from $(TRITON_IMAGE)..."; \
		mkdir -p "$(FROM_IMAGE_CACHE)"; \
		cid=$$(docker create "$(TRITON_IMAGE)"); \
		trap 'docker rm -f "$$cid" >/dev/null' EXIT; \
		rm -rf "$(NVIMGCODEC_ROOT)" "$(CUDA_ROOT)"; \
		docker cp "$$cid:/usr/local/lib/python3.12/dist-packages/nvidia/nvimgcodec" "$(NVIMGCODEC_ROOT)"; \
		mkdir -p "$(CUDA_ROOT)/lib"; \
		docker cp -L "$$cid:/usr/local/cuda/include" "$(CUDA_ROOT)/include"; \
		docker cp -L "$$cid:/usr/local/cuda/lib64/libcudart.so.13" "$(CUDA_ROOT)/lib/libcudart.so.13"; \
	fi

build: vendor-from-image
	@test -n "$(OUT)"
	@test -n "$(MODEL_NAME)"
	@mkdir -p "$(OUT)/$(VERSION)"
	@mkdir -p "$(BACKEND_OUT)"
	@sed "s/@MODEL_NAME@/$(MODEL_NAME)/g" config.pbtxt.in > "$(OUT)/config.pbtxt"
	$(CMAKE) -S . -B "$(CMAKE_BUILD_DIR)" \
		-DCMAKE_BUILD_TYPE=Release \
		-DCMAKE_INSTALL_PREFIX="$(ARTIFACT_ROOT)" \
		-DBACKEND_NAME="$(BACKEND)" \
		-DNVIMGCODEC_ROOT="$(NVIMGCODEC_ROOT)" \
		-DCUDA_ROOT="$(CUDA_ROOT)"
	$(CMAKE) --build "$(CMAKE_BUILD_DIR)" --parallel
	$(CMAKE) --install "$(CMAKE_BUILD_DIR)"

clean:
	rm -rf build
