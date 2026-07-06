#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "triton/core/tritonbackend.h"

#include "constants.h"

extern "C" void *pp_doclayout_v3_decoder_create(int device_id);
extern "C" void pp_doclayout_v3_decoder_destroy(void *decoder_state);
extern "C" const char *
pp_doclayout_v3_preprocess(void *decoder_state, const std::uint8_t *input,
                           int64_t byte_size, float *image_output,
                           float *im_shape_output, float *scale_factor_output);

namespace {

TRITONSERVER_Error *Error(const std::string &message) {
  return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG, message.c_str());
}

#define RETURN_IF_ERROR(X)                                                     \
  do {                                                                         \
    TRITONSERVER_Error *err__ = (X);                                           \
    if (err__ != nullptr) {                                                    \
      return err__;                                                            \
    }                                                                          \
  } while (false)

bool IsRequested(TRITONBACKEND_Request *request, const char *name) {
  uint32_t count = 0;
  if (TRITONBACKEND_RequestOutputCount(request, &count) != nullptr) {
    return true;
  }
  if (count == 0) {
    return true;
  }

  for (uint32_t idx = 0; idx < count; ++idx) {
    const char *output_name = nullptr;
    if (TRITONBACKEND_RequestOutputName(request, idx, &output_name) !=
        nullptr) {
      return true;
    }
    if (std::strcmp(output_name, name) == 0) {
      return true;
    }
  }
  return false;
}

TRITONSERVER_Error *ReadImage(TRITONBACKEND_Request *request,
                              std::vector<std::uint8_t> *image) {
  TRITONBACKEND_Input *input = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_RequestInput(request, "image", &input));

  const char *name = nullptr;
  TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_INVALID;
  const int64_t *shape = nullptr;
  uint32_t dims_count = 0;
  uint64_t byte_size = 0;
  uint32_t buffer_count = 0;
  RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
      input, &name, &datatype, &shape, &dims_count, &byte_size, &buffer_count));

  if (datatype != TRITONSERVER_TYPE_BYTES) {
    return Error("image must be TYPE_BYTES");
  }

  // Collect all buffers into a contiguous staging area.
  std::vector<std::uint8_t> raw(byte_size);
  uint64_t copied = 0;
  for (uint32_t idx = 0; idx < buffer_count; ++idx) {
    const void *buffer = nullptr;
    uint64_t buffer_byte_size = 0;
    TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
    int64_t memory_type_id = 0;
    RETURN_IF_ERROR(TRITONBACKEND_InputBuffer(
        input, idx, &buffer, &buffer_byte_size, &memory_type, &memory_type_id));
    if (memory_type != TRITONSERVER_MEMORY_CPU &&
        memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
      return Error("image input must be in CPU memory");
    }
    if (copied + buffer_byte_size > raw.size()) {
      return Error("image input buffers exceed expected byte size");
    }
    std::memcpy(raw.data() + copied, buffer,
                static_cast<size_t>(buffer_byte_size));
    copied += buffer_byte_size;
  }
  if (copied != raw.size()) {
    return Error("image input buffers are incomplete");
  }

  // TYPE_BYTES: each element is prefixed with a 4-byte little-endian length.
  if (raw.size() < 4) {
    return Error("image TYPE_BYTES payload too small");
  }
  std::uint32_t str_len = 0;
  std::memcpy(&str_len, raw.data(), sizeof(str_len));
  if (static_cast<uint64_t>(str_len) + 4 > raw.size()) {
    return Error("image TYPE_BYTES length prefix exceeds payload");
  }

  image->assign(raw.data() + 4, raw.data() + 4 + str_len);
  return nullptr;
}

TRITONSERVER_Error *WriteOutput(TRITONBACKEND_Response *response,
                                const char *name,
                                TRITONSERVER_DataType datatype,
                                const int64_t *shape, uint32_t dims_count,
                                const void *source, uint64_t byte_size) {
  TRITONBACKEND_Output *output = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ResponseOutput(response, &output, name,
                                               datatype, shape, dims_count));
  void *buffer = nullptr;
  TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
  int64_t memory_type_id = 0;
  RETURN_IF_ERROR(TRITONBACKEND_OutputBuffer(output, &buffer, byte_size,
                                             &memory_type, &memory_type_id));
  if (memory_type != TRITONSERVER_MEMORY_CPU &&
      memory_type != TRITONSERVER_MEMORY_CPU_PINNED) {
    return Error("preprocess output must be allocated in CPU memory");
  }
  std::memcpy(buffer, source, static_cast<size_t>(byte_size));
  return nullptr;
}

TRITONSERVER_Error *CleanupResponse(TRITONBACKEND_Response **response,
                                    TRITONSERVER_Error *error) {
  if (*response != nullptr) {
    TRITONSERVER_Error *delete_error = TRITONBACKEND_ResponseDelete(*response);
    *response = nullptr;
    if (delete_error != nullptr) {
      TRITONSERVER_ErrorDelete(delete_error);
    }
  }
  return error;
}

#define RETURN_IF_RESPONSE_ERROR(X, RESPONSE)                                  \
  do {                                                                         \
    TRITONSERVER_Error *err__ = (X);                                           \
    if (err__ != nullptr) {                                                    \
      return CleanupResponse(&(RESPONSE), err__);                              \
    }                                                                          \
  } while (false)

TRITONSERVER_Error *ExecuteRequest(TRITONBACKEND_Request *request,
                                   void *decoder_state) {
  std::vector<std::uint8_t> image;
  RETURN_IF_ERROR(ReadImage(request, &image));

  constexpr int target_size = pp_doclayout_v3::kTargetSize;
  std::vector<float> output_image(3 * target_size * target_size);
  float im_shape[2] = {};
  float scale_factor[2] = {};
  const char *preprocess_error = pp_doclayout_v3_preprocess(
      decoder_state, image.data(), static_cast<int64_t>(image.size()),
      output_image.data(), im_shape, scale_factor);
  if (preprocess_error != nullptr) {
    return Error(std::string("preprocessing failed: ") + preprocess_error);
  }

  TRITONBACKEND_Response *response = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));

  if (IsRequested(request, "im_shape")) {
    const int64_t shape[] = {1, 2};
    RETURN_IF_RESPONSE_ERROR(WriteOutput(response, "im_shape",
                                         TRITONSERVER_TYPE_FP32, shape, 2,
                                         im_shape, sizeof(im_shape)),
                             response);
  }
  if (IsRequested(request, "image")) {
    const int64_t shape[] = {1, 3, target_size, target_size};
    RETURN_IF_RESPONSE_ERROR(
        WriteOutput(response, "image", TRITONSERVER_TYPE_FP32, shape, 4,
                    output_image.data(),
                    static_cast<uint64_t>(output_image.size() * sizeof(float))),
        response);
  }
  if (IsRequested(request, "scale_factor")) {
    const int64_t shape[] = {1, 2};
    RETURN_IF_RESPONSE_ERROR(WriteOutput(response, "scale_factor",
                                         TRITONSERVER_TYPE_FP32, shape, 2,
                                         scale_factor, sizeof(scale_factor)),
                             response);
  }

  TRITONSERVER_Error *send_error = TRITONBACKEND_ResponseSend(
      response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr);
  response = nullptr;
  return send_error;
}

TRITONSERVER_Error *SendError(TRITONBACKEND_Request *request,
                              TRITONSERVER_Error *error) {
  TRITONBACKEND_Response *response = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
  return TRITONBACKEND_ResponseSend(
      response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, error);
}

} // namespace

extern "C" {

TRITONBACKEND_ISPEC TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *instance) {
  int32_t device_id = 0;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceDeviceId(instance, &device_id));

  void *decoder_state = pp_doclayout_v3_decoder_create(device_id);
  if (decoder_state == nullptr) {
    return Error("failed to create nvImageCodec decoder");
  }
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(instance, decoder_state));
  return nullptr;
}

TRITONBACKEND_ISPEC TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *instance) {
  void *decoder_state = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &decoder_state));
  pp_doclayout_v3_decoder_destroy(decoder_state);
  return nullptr;
}

TRITONBACKEND_ISPEC TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance *instance,
                                   TRITONBACKEND_Request **requests,
                                   const uint32_t request_count) {
  void *decoder_state = nullptr;
  TRITONSERVER_Error *state_error =
      TRITONBACKEND_ModelInstanceState(instance, &decoder_state);
  if (state_error != nullptr) {
    for (uint32_t idx = 0; idx < request_count; ++idx) {
      TRITONSERVER_Error *send_error = SendError(requests[idx], state_error);
      if (send_error != nullptr) {
        TRITONSERVER_ErrorDelete(send_error);
      }
      TRITONBACKEND_RequestRelease(requests[idx],
                                   TRITONSERVER_REQUEST_RELEASE_ALL);
    }
    TRITONSERVER_ErrorDelete(state_error);
    return nullptr;
  }

  for (uint32_t idx = 0; idx < request_count; ++idx) {
    TRITONBACKEND_Request *request = requests[idx];
    TRITONSERVER_Error *error = ExecuteRequest(request, decoder_state);
    if (error != nullptr) {
      TRITONSERVER_Error *send_error = SendError(request, error);
      if (send_error != nullptr) {
        TRITONSERVER_ErrorDelete(send_error);
      }
      TRITONSERVER_ErrorDelete(error);
    }
    TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL);
  }
  return nullptr;
}

} // extern "C"
