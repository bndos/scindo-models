#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include "constants.h"
#include "nvimgcodec_raii.h"

namespace {

using pp_doclayout_v3::AllocateDeviceBuffer;
using pp_doclayout_v3::CodeStreamHandle;
using pp_doclayout_v3::DecoderHandle;
using pp_doclayout_v3::DeviceBufferHandle;
using pp_doclayout_v3::FutureHandle;
using pp_doclayout_v3::ImageHandle;
using pp_doclayout_v3::InstanceHandle;
using pp_doclayout_v3::kTargetSize;

thread_local std::string g_last_error;

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

const char *Fail(const std::string &message) {
  g_last_error = message;
  return g_last_error.c_str();
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

extern "C" const char *
pp_doclayout_v3_preprocess(void *decoder_state, const std::uint8_t *input,
                           int64_t byte_size, float *image_output,
                           float *im_shape_output, float *scale_factor_output) {
  auto *state = static_cast<DecoderState *>(decoder_state);
  if (state == nullptr || input == nullptr || image_output == nullptr ||
      im_shape_output == nullptr || scale_factor_output == nullptr ||
      byte_size <= 0) {
    return Fail("invalid arguments");
  }

  try {
    nvimgcodecCodeStream_t raw_stream = nullptr;
    if (nvimgcodecCodeStreamCreateFromHostMem(
            state->instance.get(), &raw_stream, input,
            static_cast<size_t>(byte_size),
            nullptr) != NVIMGCODEC_STATUS_SUCCESS) {
      return Fail("failed to parse image code stream");
    }
    CodeStreamHandle code_stream(raw_stream);

    nvimgcodecImageInfo_t source_info{};
    source_info.struct_type = NVIMGCODEC_STRUCTURE_TYPE_IMAGE_INFO;
    source_info.struct_size = sizeof(source_info);
    if (nvimgcodecCodeStreamGetImageInfo(code_stream.get(), &source_info) !=
        NVIMGCODEC_STATUS_SUCCESS) {
      return Fail("failed to read image info");
    }

    const int width = static_cast<int>(source_info.plane_info[0].width);
    const int height = static_cast<int>(source_info.plane_info[0].height);
    if (width <= 0 || height <= 0) {
      return Fail("image has invalid dimensions");
    }

    const size_t decoded_size = static_cast<size_t>(width) * height * 3;
    DeviceBufferHandle device_buffer = AllocateDeviceBuffer(decoded_size);
    if (!device_buffer) {
      return Fail("cudaMalloc failed for decode buffer");
    }

    const nvimgcodecImageInfo_t target_info =
        DecodeTargetInfo(width, height, device_buffer.get());
    nvimgcodecImage_t raw_image = nullptr;
    if (nvimgcodecImageCreate(state->instance.get(), &raw_image,
                              &target_info) != NVIMGCODEC_STATUS_SUCCESS) {
      return Fail("failed to create decode target image");
    }
    ImageHandle image(raw_image);

    nvimgcodecDecodeParams_t decode_params{};
    decode_params.struct_type = NVIMGCODEC_STRUCTURE_TYPE_DECODE_PARAMS;
    decode_params.struct_size = sizeof(decode_params);
    decode_params.apply_exif_orientation = 1;

    nvimgcodecCodeStream_t stream_handle = code_stream.get();
    nvimgcodecImage_t image_handle = image.get();
    nvimgcodecFuture_t raw_future = nullptr;
    if (nvimgcodecDecoderDecode(state->decoder.get(), &stream_handle,
                                &image_handle, 1, &decode_params,
                                &raw_future) != NVIMGCODEC_STATUS_SUCCESS) {
      return Fail("failed to submit decode");
    }
    FutureHandle future(raw_future);
    nvimgcodecFutureWaitForAll(future.get());

    nvimgcodecProcessingStatus_t decode_status =
        NVIMGCODEC_PROCESSING_STATUS_UNKNOWN;
    size_t decode_status_count = 1;
    nvimgcodecFutureGetProcessingStatus(future.get(), &decode_status,
                                        &decode_status_count);
    if (decode_status != NVIMGCODEC_PROCESSING_STATUS_SUCCESS) {
      char message[64];
      snprintf(message, sizeof(message),
               "decode failed, nvImageCodec status 0x%x", decode_status);
      return Fail(message);
    }

    std::vector<std::uint8_t> host_buffer(decoded_size);
    if (cudaMemcpy(host_buffer.data(), device_buffer.get(), decoded_size,
                   cudaMemcpyDeviceToHost) != cudaSuccess) {
      return Fail("cudaMemcpy device to host failed");
    }

    const cv::Mat decoded(height, width, CV_8UC3, host_buffer.data());
    WriteOutputTensors(decoded, image_output, im_shape_output,
                       scale_factor_output);
    return nullptr;
  } catch (const std::exception &e) {
    return Fail(std::string("exception: ") + e.what());
  }
}
