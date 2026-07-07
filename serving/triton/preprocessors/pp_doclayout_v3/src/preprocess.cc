#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>
#include <vector>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "constants.h"
#include "nvimgcodec_raii.h"
#include "preprocess.h"

namespace {

using pp_doclayout_v3::AllocateDeviceBuffer;
using pp_doclayout_v3::CodeStreamHandle;
using pp_doclayout_v3::DecoderHandle;
using pp_doclayout_v3::DeviceBufferHandle;
using pp_doclayout_v3::FutureHandle;
using pp_doclayout_v3::ImageHandle;
using pp_doclayout_v3::InstanceHandle;
using pp_doclayout_v3::kTargetSize;

struct DecoderState {
  InstanceHandle instance;
  DecoderHandle decoder;
};

nvimgcodecImageInfo_t DecodeTargetInfo(int width, int height,
                                       void *device_buffer) {
  nvimgcodecImageInfo_t info{};
  info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
  info.struct_size = sizeof(info);
  info.color_spec = NVIMGCODEC_COLORSPEC_SRGB;
  // BGR interleaved, matching cv::imdecode(..., IMREAD_COLOR)'s channel
  // order, so the resize/normalize code below stays unchanged.
  info.sample_format = NVIMGCODEC_SAMPLEFORMAT_I_BGR;
  info.num_planes = 1;
  info.plane_info[0].struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_PLANE_INFO;
  info.plane_info[0].struct_size = sizeof(info.plane_info[0]);
  info.plane_info[0].width = static_cast<uint32_t>(width);
  info.plane_info[0].height = static_cast<uint32_t>(height);
  info.plane_info[0].row_stride = static_cast<size_t>(width) * 3;
  info.plane_info[0].num_channels = 3;
  info.plane_info[0].sample_type = NVIMGCODEC_SAMPLE_DATA_TYPE_UINT8;
  info.buffer = device_buffer;
  info.buffer_kind = NVIMGCODEC_IMAGE_BUFFER_KIND_STRIDED_DEVICE;
  return info;
}

void WriteOutputTensors(const cv::Mat &decoded, float *image_output,
                        float *im_shape_output, float *scale_factor_output) {
  cv::Mat resized;
  cv::resize(decoded, resized, cv::Size(kTargetSize, kTargetSize), 0.0, 0.0,
             cv::INTER_CUBIC);

  im_shape_output[0] = static_cast<float>(kTargetSize);
  im_shape_output[1] = static_cast<float>(kTargetSize);
  scale_factor_output[0] =
      static_cast<float>(kTargetSize) / static_cast<float>(decoded.rows);
  scale_factor_output[1] =
      static_cast<float>(kTargetSize) / static_cast<float>(decoded.cols);

  const int plane_size = kTargetSize * kTargetSize;
  cv::Mat channels[3] = {
      cv::Mat(kTargetSize, kTargetSize, CV_32F, image_output),
      cv::Mat(kTargetSize, kTargetSize, CV_32F, image_output + plane_size),
      cv::Mat(kTargetSize, kTargetSize, CV_32F, image_output + 2 * plane_size),
  };
  cv::Mat float_image;
  resized.convertTo(float_image, CV_32F, 1.0 / 255.0);
  cv::split(float_image, channels);
}

} // namespace

extern "C" void *pp_doclayout_v3_decoder_create(int device_id) {
  const char *extensions_path = getenv("NVIMGCODEC_EXTENSIONS_PATH");
  if (extensions_path == nullptr || extensions_path[0] == '\0') {
    fprintf(stderr, "pp_doclayout_v3: NVIMGCODEC_EXTENSIONS_PATH is not set; "
                    "nvImageCodec extensions (nvJPEG, OpenCV) will not load\n");
    extensions_path = nullptr;
  }

  nvimgcodecInstanceCreateInfo_t instance_info{};
  instance_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;
  instance_info.struct_size = sizeof(instance_info);
  instance_info.load_builtin_modules = 1;
  instance_info.load_extension_modules = 1;
  instance_info.extension_modules_path = extensions_path;

  nvimgcodecInstance_t raw_instance = nullptr;
  if (nvimgcodecInstanceCreate(&raw_instance, &instance_info) !=
      NVIMGCODEC_STATUS_SUCCESS) {
    return nullptr;
  }
  InstanceHandle instance(raw_instance);

  nvimgcodecExecutionParams_t exec_params{};
  exec_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_EXECUTION_PARAMS;
  exec_params.struct_size = sizeof(exec_params);
  exec_params.device_id = device_id;

  nvimgcodecDecoder_t raw_decoder = nullptr;
  if (nvimgcodecDecoderCreate(instance.get(), &raw_decoder, &exec_params,
                              nullptr) != NVIMGCODEC_STATUS_SUCCESS) {
    return nullptr;
  }

  return new DecoderState{std::move(instance), DecoderHandle(raw_decoder)};
}

extern "C" void pp_doclayout_v3_decoder_destroy(void *decoder_state) {
  delete static_cast<DecoderState *>(decoder_state);
}

void pp_doclayout_v3_preprocess_batch(
    void *decoder_state, const std::vector<std::vector<std::uint8_t>> &images,
    float *image_output, float *im_shape_output, float *scale_factor_output,
    std::vector<std::string> *errors) {
  const auto count = static_cast<int>(images.size());
  errors->assign(count, "");

  auto *state = static_cast<DecoderState *>(decoder_state);
  if (state == nullptr || image_output == nullptr ||
      im_shape_output == nullptr || scale_factor_output == nullptr) {
    errors->assign(count, "invalid arguments to preprocess_batch");
    return;
  }

  // Per-image setup state. All handles are unique_ptr (move-only), so this
  // struct is move-constructible and safe to store in a vector.
  struct PerImage {
    int idx;
    CodeStreamHandle code_stream;
    DeviceBufferHandle device_buffer;
    ImageHandle image;
    int src_width, src_height;
  };

  // create per-image nvImageCodec resources (sequential).
  std::vector<PerImage> valid;
  valid.reserve(count);

  for (int i = 0; i < count; ++i) {
    const auto &img = images[i];
    if (img.empty()) {
      (*errors)[i] = "empty image input";
      continue;
    }

    nvimgcodecCodeStream_t raw_stream = nullptr;
    if (nvimgcodecCodeStreamCreateFromHostMem(
            state->instance.get(), &raw_stream, img.data(), img.size(),
            nullptr) != NVIMGCODEC_STATUS_SUCCESS) {
      (*errors)[i] = "failed to parse image code stream";
      continue;
    }
    CodeStreamHandle code_stream(raw_stream);

    nvimgcodecImageInfo_t source_info{};
    source_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
    source_info.struct_size = sizeof(source_info);
    if (nvimgcodecCodeStreamGetImageInfo(code_stream.get(), &source_info) !=
        NVIMGCODEC_STATUS_SUCCESS) {
      (*errors)[i] = "failed to read image info";
      continue;
    }

    const int w = static_cast<int>(source_info.plane_info[0].width);
    const int h = static_cast<int>(source_info.plane_info[0].height);
    if (w <= 0 || h <= 0) {
      (*errors)[i] = "image has invalid dimensions";
      continue;
    }

    DeviceBufferHandle device_buffer =
        AllocateDeviceBuffer(static_cast<size_t>(w) * h * 3);
    if (!device_buffer) {
      (*errors)[i] = "cudaMalloc failed for decode buffer";
      continue;
    }

    const nvimgcodecImageInfo_t target_info =
        DecodeTargetInfo(w, h, device_buffer.get());
    nvimgcodecImage_t raw_image = nullptr;
    if (nvimgcodecImageCreate(state->instance.get(), &raw_image,
                              &target_info) != NVIMGCODEC_STATUS_SUCCESS) {
      (*errors)[i] = "failed to create decode target image";
      continue;
    }

    valid.push_back({i, std::move(code_stream), std::move(device_buffer),
                     ImageHandle(raw_image), w, h});
  }

  if (valid.empty()) {
    return;
  }

  // batch-decode all valid images in one nvImageCodec call.
  std::vector<nvimgcodecCodeStream_t> raw_streams(valid.size());
  std::vector<nvimgcodecImage_t> raw_images(valid.size());
  for (size_t j = 0; j < valid.size(); ++j) {
    raw_streams[j] = valid[j].code_stream.get();
    raw_images[j] = valid[j].image.get();
  }

  nvimgcodecDecodeParams_t decode_params{};
  decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
  decode_params.struct_size = sizeof(decode_params);
  decode_params.apply_exif_orientation = 1;

  nvimgcodecFuture_t raw_future = nullptr;
  if (nvimgcodecDecoderDecode(state->decoder.get(), raw_streams.data(),
                              raw_images.data(), static_cast<int>(valid.size()),
                              &decode_params,
                              &raw_future) != NVIMGCODEC_STATUS_SUCCESS) {
    for (const auto &s : valid) {
      (*errors)[s.idx] = "failed to submit batch decode";
    }
    return;
  }
  FutureHandle future(raw_future);
  nvimgcodecFutureWaitForAll(future.get());

  std::vector<nvimgcodecProcessingStatus_t> statuses(
      valid.size(), NVIMGCODEC_PROCESSING_STATUS_UNKNOWN);
  size_t status_count = valid.size();
  nvimgcodecFutureGetProcessingStatus(future.get(), statuses.data(),
                                      &status_count);

  // copy each decoded image to host and write output tensors.
  constexpr size_t kImageFloats =
      3 * static_cast<size_t>(kTargetSize) * kTargetSize;

  for (size_t j = 0; j < valid.size(); ++j) {
    const int i = valid[j].idx;
    if (statuses[j] != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
      char msg[64];
      snprintf(msg, sizeof(msg), "decode failed, nvImageCodec status 0x%x",
               statuses[j]);
      (*errors)[i] = msg;
      continue;
    }

    const int w = valid[j].src_width;
    const int h = valid[j].src_height;
    const size_t decoded_size = static_cast<size_t>(w) * h * 3;

    std::vector<std::uint8_t> host_buffer(decoded_size);
    if (cudaMemcpy(host_buffer.data(), valid[j].device_buffer.get(),
                   decoded_size, cudaMemcpyDeviceToHost) != cudaSuccess) {
      (*errors)[i] = "cudaMemcpy device to host failed";
      continue;
    }

    const cv::Mat decoded(h, w, CV_8UC3, host_buffer.data());
    WriteOutputTensors(decoded,
                       image_output + static_cast<size_t>(i) * kImageFloats,
                       im_shape_output + i * 2, scale_factor_output + i * 2);
  }
}
