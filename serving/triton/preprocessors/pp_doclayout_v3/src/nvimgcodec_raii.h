#ifndef PP_DOCLAYOUT_V3_NVIMGCODEC_RAII_H_
#define PP_DOCLAYOUT_V3_NVIMGCODEC_RAII_H_

#include <memory>
#include <type_traits>

#include <cuda_runtime_api.h>
#include <nvimgcodec.h>

namespace pp_doclayout_v3 {

namespace detail {

struct InstanceDeleter {
  void operator()(std::remove_pointer_t<nvimgcodecInstance_t> *p) const {
    nvimgcodecInstanceDestroy(p);
  }
};
struct DecoderDeleter {
  void operator()(std::remove_pointer_t<nvimgcodecDecoder_t> *p) const {
    nvimgcodecDecoderDestroy(p);
  }
};
struct CodeStreamDeleter {
  void operator()(std::remove_pointer_t<nvimgcodecCodeStream_t> *p) const {
    nvimgcodecCodeStreamDestroy(p);
  }
};
struct ImageDeleter {
  void operator()(std::remove_pointer_t<nvimgcodecImage_t> *p) const {
    nvimgcodecImageDestroy(p);
  }
};
struct FutureDeleter {
  void operator()(std::remove_pointer_t<nvimgcodecFuture_t> *p) const {
    nvimgcodecFutureDestroy(p);
  }
};
struct CudaDeviceDeleter {
  void operator()(void *p) const { cudaFree(p); }
};

} // namespace detail

using InstanceHandle =
    std::unique_ptr<std::remove_pointer_t<nvimgcodecInstance_t>,
                    detail::InstanceDeleter>;
using DecoderHandle =
    std::unique_ptr<std::remove_pointer_t<nvimgcodecDecoder_t>,
                    detail::DecoderDeleter>;
using CodeStreamHandle =
    std::unique_ptr<std::remove_pointer_t<nvimgcodecCodeStream_t>,
                    detail::CodeStreamDeleter>;
using ImageHandle = std::unique_ptr<std::remove_pointer_t<nvimgcodecImage_t>,
                                    detail::ImageDeleter>;
using FutureHandle = std::unique_ptr<std::remove_pointer_t<nvimgcodecFuture_t>,
                                     detail::FutureDeleter>;
using DeviceBufferHandle = std::unique_ptr<void, detail::CudaDeviceDeleter>;

inline DeviceBufferHandle AllocateDeviceBuffer(size_t size) {
  void *ptr = nullptr;
  if (cudaMalloc(&ptr, size) != cudaSuccess) {
    return DeviceBufferHandle();
  }
  return DeviceBufferHandle(ptr);
}

} // namespace pp_doclayout_v3

#endif // PP_DOCLAYOUT_V3_NVIMGCODEC_RAII_H_
