#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "triton/backend/backend_model_instance.h"
#include "triton/core/tritonbackend.h"

#include "constants.h"
#include "preprocess.h"

namespace triton {
namespace backend {
namespace pp_doclayout_v3_preprocess {

namespace {

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

  TRITONSERVER_DataType datatype = TRITONSERVER_TYPE_INVALID;
  uint64_t byte_size = 0;
  uint32_t buffer_count = 0;
  RETURN_IF_ERROR(TRITONBACKEND_InputProperties(
      input, nullptr, &datatype, nullptr, nullptr, &byte_size, &buffer_count));

  if (datatype != TRITONSERVER_TYPE_UINT8) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                                 "image must be TYPE_UINT8");
  }

  image->resize(byte_size);
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
      return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                                   "image input must be in CPU memory");
    }
    if (copied + buffer_byte_size > image->size()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          "image input buffers exceed expected byte size");
    }
    std::memcpy(image->data() + copied, buffer,
                static_cast<size_t>(buffer_byte_size));
    copied += buffer_byte_size;
  }
  if (copied != image->size()) {
    return TRITONSERVER_ErrorNew(TRITONSERVER_ERROR_INVALID_ARG,
                                 "image input buffers are incomplete");
  }

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
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INVALID_ARG,
        "preprocess output must be allocated in CPU memory");
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

// Write all requested outputs for one request into its response, using the
// i-th slice of the batch output arrays. On error, *response is deleted and
// set to nullptr via CleanupResponse.
TRITONSERVER_Error *WriteRequestOutputs(TRITONBACKEND_Request *request,
                                        TRITONBACKEND_Response **response,
                                        const float *image_slice,
                                        const float *im_shape_slice,
                                        const float *scale_factor_slice) {
  constexpr int target_size = ::pp_doclayout_v3::kTargetSize;
  if (IsRequested(request, "im_shape")) {
    const int64_t shape[] = {1, 2};
    TRITONSERVER_Error *err =
        WriteOutput(*response, "im_shape", TRITONSERVER_TYPE_FP32, shape, 2,
                    im_shape_slice, 2 * sizeof(float));
    if (err != nullptr) {
      return CleanupResponse(response, err);
    }
  }
  if (IsRequested(request, "image")) {
    const int64_t shape[] = {1, 3, target_size, target_size};
    TRITONSERVER_Error *err = WriteOutput(
        *response, "image", TRITONSERVER_TYPE_FP32, shape, 4, image_slice,
        3 * static_cast<size_t>(target_size) * target_size * sizeof(float));
    if (err != nullptr) {
      return CleanupResponse(response, err);
    }
  }
  if (IsRequested(request, "scale_factor")) {
    const int64_t shape[] = {1, 2};
    TRITONSERVER_Error *err =
        WriteOutput(*response, "scale_factor", TRITONSERVER_TYPE_FP32, shape, 2,
                    scale_factor_slice, 2 * sizeof(float));
    if (err != nullptr) {
      return CleanupResponse(response, err);
    }
  }
  return nullptr;
}

TRITONSERVER_Error *SendError(TRITONBACKEND_Request *request,
                              TRITONSERVER_Error *error) {
  TRITONBACKEND_Response *response = nullptr;
  RETURN_IF_ERROR(TRITONBACKEND_ResponseNew(&response, request));
  return TRITONBACKEND_ResponseSend(
      response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, error);
}

} // namespace

/////////////

//
// ModelState
//
// State associated with a model that is using this backend. An object
// of this class is created and associated with each
// TRITONBACKEND_Model. ModelState is derived from BackendModel class
// provided in the backend utilities that provides many common
// functions.
//
class ModelState : public BackendModel {
public:
  static TRITONSERVER_Error *Create(TRITONBACKEND_Model *triton_model,
                                    ModelState **state);
  virtual ~ModelState() = default;

private:
  explicit ModelState(TRITONBACKEND_Model *triton_model);

  // Validate that this model is supported by this backend.
  TRITONSERVER_Error *ValidateModelConfig();
  TRITONSERVER_Error *CheckTensor(common::TritonJson::Value &io,
                                  const std::string &expected_name,
                                  const std::string &expected_dtype,
                                  const std::vector<int64_t> &expected_dims);
};

ModelState::ModelState(TRITONBACKEND_Model *triton_model)
    : BackendModel(triton_model) {
  THROW_IF_BACKEND_MODEL_ERROR(ValidateModelConfig());
}

TRITONSERVER_Error *ModelState::Create(TRITONBACKEND_Model *triton_model,
                                       ModelState **state) {
  try {
    *state = new ModelState(triton_model);
  } catch (const BackendModelException &ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }
  return nullptr; // success
}

TRITONSERVER_Error *
ModelState::CheckTensor(common::TritonJson::Value &io,
                        const std::string &expected_name,
                        const std::string &expected_dtype,
                        const std::vector<int64_t> &expected_dims) {
  const char *name = nullptr;
  size_t name_len = 0;
  RETURN_IF_ERROR(io.MemberAsString("name", &name, &name_len));
  RETURN_ERROR_IF_FALSE(expected_name == name, TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("expected tensor named '") + expected_name +
                            "', got '" + name + "'");

  std::string dtype;
  RETURN_IF_ERROR(io.MemberAsString("data_type", &dtype));
  RETURN_ERROR_IF_FALSE(expected_dtype == dtype, TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("'") + expected_name +
                            "': expected data_type " + expected_dtype +
                            ", got " + dtype);

  std::vector<int64_t> dims;
  RETURN_IF_ERROR(backend::ParseShape(io, "dims", &dims));
  RETURN_ERROR_IF_FALSE(dims == expected_dims, TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("'") + expected_name + "': expected dims " +
                            backend::ShapeToString(expected_dims) + ", got " +
                            backend::ShapeToString(dims));
  return nullptr;
}

TRITONSERVER_Error *ModelState::ValidateModelConfig() {
  // If verbose logging is enabled, dump the model's configuration as
  // JSON into the console output.
  if (TRITONSERVER_LogIsEnabled(TRITONSERVER_LOG_VERBOSE)) {
    common::TritonJson::WriteBuffer buffer;
    RETURN_IF_ERROR(ModelConfig().PrettyWrite(&buffer));
    LOG_MESSAGE(
        TRITONSERVER_LOG_VERBOSE,
        (std::string("model configuration:\n") + buffer.Contents()).c_str());
  }

  int64_t max_batch_size = 0;
  RETURN_IF_ERROR(ModelConfig().MemberAsInt("max_batch_size", &max_batch_size));
  RETURN_ERROR_IF_FALSE(
      max_batch_size > 0, TRITONSERVER_ERROR_INVALID_ARG,
      std::string("pp_doclayout_v3_preprocess requires max_batch_size > 0"));

  common::TritonJson::Value inputs, outputs;
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &inputs));
  RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &outputs));
  RETURN_ERROR_IF_FALSE(inputs.ArraySize() == 1, TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("must have exactly 1 input, got ") +
                            std::to_string(inputs.ArraySize()));
  RETURN_ERROR_IF_FALSE(outputs.ArraySize() == 3,
                        TRITONSERVER_ERROR_INVALID_ARG,
                        std::string("must have exactly 3 outputs, got ") +
                            std::to_string(outputs.ArraySize()));

  common::TritonJson::Value input;
  RETURN_IF_ERROR(inputs.IndexAsObject(0, &input));
  RETURN_IF_ERROR(CheckTensor(input, "image", "TYPE_UINT8", {-1}));

  struct ExpectedOutput {
    const char *name;
    std::vector<int64_t> dims;
  };
  const ExpectedOutput expected_outputs[] = {
      {"im_shape", {2}},
      {"image",
       {3, ::pp_doclayout_v3::kTargetSize, ::pp_doclayout_v3::kTargetSize}},
      {"scale_factor", {2}},
  };
  for (size_t idx = 0; idx < 3; ++idx) {
    common::TritonJson::Value output;
    RETURN_IF_ERROR(outputs.IndexAsObject(idx, &output));
    RETURN_IF_ERROR(CheckTensor(output, expected_outputs[idx].name, "TYPE_FP32",
                                expected_outputs[idx].dims));
  }
  return nullptr;
}

extern "C" {

// Triton calls TRITONBACKEND_Initialize when a backend is loaded into
// Triton to allow the backend to create and initialize any state that
// is intended to be shared across all models and model instances that
// use the backend. The backend should also verify version
// compatibility with Triton in this function.
//
TRITONSERVER_Error *TRITONBACKEND_Initialize(TRITONBACKEND_Backend *backend) {
  const char *cname;
  RETURN_IF_ERROR(TRITONBACKEND_BackendName(backend, &cname));
  std::string name(cname);

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("TRITONBACKEND_Initialize: ") + name).c_str());

  // Check the backend API version that Triton supports vs. what this
  // backend was compiled against. Make sure that the Triton major
  // version is the same and the minor version is >= what this backend
  // uses.
  uint32_t api_version_major, api_version_minor;
  RETURN_IF_ERROR(
      TRITONBACKEND_ApiVersion(&api_version_major, &api_version_minor));

  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("Triton TRITONBACKEND API version: ") +
               std::to_string(api_version_major) + "." +
               std::to_string(api_version_minor))
                  .c_str());
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("'") + name + "' TRITONBACKEND API version: " +
               std::to_string(TRITONBACKEND_API_VERSION_MAJOR) + "." +
               std::to_string(TRITONBACKEND_API_VERSION_MINOR))
                  .c_str());

  if ((api_version_major != TRITONBACKEND_API_VERSION_MAJOR) ||
      (api_version_minor < TRITONBACKEND_API_VERSION_MINOR)) {
    return TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_UNSUPPORTED,
        "triton backend API version does not support this backend");
  }

  // The backend configuration may contain information needed by the
  // backend, such as tritonserver command-line arguments. This
  // backend doesn't use any such configuration but for this example
  // print whatever is available.
  TRITONSERVER_Message *backend_config_message;
  RETURN_IF_ERROR(
      TRITONBACKEND_BackendConfig(backend, &backend_config_message));

  const char *buffer;
  size_t byte_size;
  RETURN_IF_ERROR(TRITONSERVER_MessageSerializeToJson(backend_config_message,
                                                      &buffer, &byte_size));
  LOG_MESSAGE(TRITONSERVER_LOG_INFO,
              (std::string("backend configuration:\n") + buffer).c_str());

  return nullptr; // success
}

// Triton calls TRITONBACKEND_Finalize when a backend is no longer
// needed.
//
TRITONSERVER_Error *TRITONBACKEND_Finalize(TRITONBACKEND_Backend *backend) {
  return nullptr; // success
}

} // extern "C"

/////////////

//
// ModelInstanceState
//
// State associated with a model instance. An object of this class is
// created and associated with each
// TRITONBACKEND_ModelInstance. ModelInstanceState is derived from
// BackendModelInstance class provided in the backend utilities that
// provides many common functions.
//
class ModelInstanceState : public BackendModelInstance {
public:
  static TRITONSERVER_Error *
  Create(ModelState *model_state,
         TRITONBACKEND_ModelInstance *triton_model_instance,
         ModelInstanceState **state);
  virtual ~ModelInstanceState() {
    pp_doclayout_v3_decoder_destroy(decoder_state_);
  }

  // Get the state of the model that corresponds to this instance.
  ModelState *StateForModel() const { return model_state_; }

  void *DecoderState() const { return decoder_state_; }

private:
  ModelInstanceState(ModelState *model_state,
                     TRITONBACKEND_ModelInstance *triton_model_instance);

  ModelState *model_state_;
  void *decoder_state_;
};

ModelInstanceState::ModelInstanceState(
    ModelState *model_state, TRITONBACKEND_ModelInstance *triton_model_instance)
    : BackendModelInstance(model_state, triton_model_instance),
      model_state_(model_state),
      decoder_state_(pp_doclayout_v3_decoder_create(DeviceId())) {
  if (decoder_state_ == nullptr) {
    throw BackendModelInstanceException(TRITONSERVER_ErrorNew(
        TRITONSERVER_ERROR_INTERNAL, "failed to create nvImageCodec decoder"));
  }
}

TRITONSERVER_Error *
ModelInstanceState::Create(ModelState *model_state,
                           TRITONBACKEND_ModelInstance *triton_model_instance,
                           ModelInstanceState **state) {
  try {
    *state = new ModelInstanceState(model_state, triton_model_instance);
  } catch (const BackendModelInstanceException &ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  return nullptr; // success
}

extern "C" {

// Triton calls TRITONBACKEND_ModelInitialize when a model is loaded
// to allow the backend to create any state associated with the model,
// and to also examine the model configuration to determine if the
// configuration is suitable for the backend. Any errors reported by
// this function will prevent the model from loading.
//
TRITONSERVER_Error *TRITONBACKEND_ModelInitialize(TRITONBACKEND_Model *model) {
  ModelState *model_state;
  RETURN_IF_ERROR(ModelState::Create(model, &model_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelSetState(
      model, reinterpret_cast<void *>(model_state)));
  return nullptr; // success
}

// Triton calls TRITONBACKEND_ModelFinalize when a model is no longer
// needed. The backend should cleanup any state associated with the
// model. This function will not be called until all model instances
// of the model have been finalized.
//
TRITONSERVER_Error *TRITONBACKEND_ModelFinalize(TRITONBACKEND_Model *model) {
  void *vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vstate));
  delete reinterpret_cast<ModelState *>(vstate);
  return nullptr; // success
}

} // extern "C"

extern "C" {

// Triton calls TRITONBACKEND_ModelInstanceInitialize when a model
// instance is created to allow the backend to initialize any state
// associated with the instance.
//
TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceInitialize(TRITONBACKEND_ModelInstance *instance) {
  // Get the model state associated with this instance's model.
  TRITONBACKEND_Model *model;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceModel(instance, &model));

  void *vmodelstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelState(model, &vmodelstate));
  ModelState *model_state = reinterpret_cast<ModelState *>(vmodelstate);

  // Create a ModelInstanceState object and associate it with the
  // TRITONBACKEND_ModelInstance.
  ModelInstanceState *instance_state;
  RETURN_IF_ERROR(
      ModelInstanceState::Create(model_state, instance, &instance_state));
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceSetState(
      instance, reinterpret_cast<void *>(instance_state)));
  return nullptr; // success
}

// Triton calls TRITONBACKEND_ModelInstanceFinalize when a model
// instance is no longer needed. The backend should cleanup any state
// associated with the model instance.
//
TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceFinalize(TRITONBACKEND_ModelInstance *instance) {
  void *vstate;
  RETURN_IF_ERROR(TRITONBACKEND_ModelInstanceState(instance, &vstate));
  delete reinterpret_cast<ModelInstanceState *>(vstate);
  return nullptr; // success
}

} // extern "C"

/////////////

extern "C" {

// When Triton calls TRITONBACKEND_ModelInstanceExecute it is required
// that a backend create a response for each request in the batch. A
// response may be the output tensors required for that request or may
// be an error that is returned in the response.
//
TRITONSERVER_Error *
TRITONBACKEND_ModelInstanceExecute(TRITONBACKEND_ModelInstance *instance,
                                   TRITONBACKEND_Request **requests,
                                   const uint32_t request_count) {
  // Triton will not call this function simultaneously for the same
  // 'instance'. But since this backend could be used by multiple
  // instances from multiple models the implementation needs to handle
  // multiple calls to this function at the same time (with different
  // 'instance' objects). Best practice for a high-performance
  // implementation is to avoid introducing mutex/lock and instead use
  // only function-local and model-instance-specific state.
  void *vstate = nullptr;
  TRITONSERVER_Error *state_error =
      TRITONBACKEND_ModelInstanceState(instance, &vstate);
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
  auto *instance_state = reinterpret_cast<ModelInstanceState *>(vstate);
  void *decoder_state = instance_state->DecoderState();

  // At this point, the backend takes ownership of 'requests', which
  // means that it is responsible for sending a response for every
  // request. From here, even if something goes wrong in processing,
  // the backend must return 'nullptr' from this function to indicate
  // success. Any errors and failures must be communicated via the
  // response objects.

  uint64_t exec_start_ns = 0;
  SET_TIMESTAMP(exec_start_ns);

  // read all images. Failed reads leave images[i] empty.
  std::vector<std::vector<std::uint8_t>> images(request_count);
  std::vector<TRITONSERVER_Error *> read_errors(request_count, nullptr);
  for (uint32_t i = 0; i < request_count; ++i) {
    read_errors[i] = ReadImage(requests[i], &images[i]);
  }

  // batch preprocess (GPU decode + resize/normalize).
  constexpr int target_size = ::pp_doclayout_v3::kTargetSize;
  constexpr size_t kImageFloats =
      3 * static_cast<size_t>(target_size) * target_size;
  std::vector<float> batch_image(request_count * kImageFloats);
  std::vector<float> batch_im_shape(request_count * 2);
  std::vector<float> batch_scale_factor(request_count * 2);
  std::vector<std::string> preprocess_errors;
  pp_doclayout_v3_preprocess_batch(
      decoder_state, images, batch_image.data(), batch_im_shape.data(),
      batch_scale_factor.data(), &preprocess_errors);

  uint64_t compute_end_ns = 0;
  SET_TIMESTAMP(compute_end_ns);

  // send per-request responses.
  for (uint32_t i = 0; i < request_count; ++i) {
    TRITONSERVER_Error *error = nullptr;

    if (read_errors[i] != nullptr) {
      error = read_errors[i];
      read_errors[i] = nullptr;
    } else if (!preprocess_errors[i].empty()) {
      error = TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INTERNAL,
          (std::string("preprocessing failed: ") + preprocess_errors[i])
              .c_str());
    }

    bool request_success = (error == nullptr);
    if (error == nullptr) {
      TRITONBACKEND_Response *response = nullptr;
      error = TRITONBACKEND_ResponseNew(&response, requests[i]);
      if (error == nullptr) {
        error = WriteRequestOutputs(
            requests[i], &response, batch_image.data() + i * kImageFloats,
            batch_im_shape.data() + i * 2, batch_scale_factor.data() + i * 2);
      }
      if (error == nullptr) {
        // ResponseSend always consumes the response object.
        LOG_IF_ERROR(
            TRITONBACKEND_ResponseSend(
                response, TRITONSERVER_RESPONSE_COMPLETE_FINAL, nullptr),
            "failed sending response");
      } else {
        request_success = false;
        // response was deleted and nulled by WriteRequestOutputs; send error.
        TRITONSERVER_Error *send_error = SendError(requests[i], error);
        TRITONSERVER_ErrorDelete(error);
        error = nullptr;
        if (send_error != nullptr) {
          TRITONSERVER_ErrorDelete(send_error);
        }
      }
    } else {
      TRITONSERVER_Error *send_error = SendError(requests[i], error);
      TRITONSERVER_ErrorDelete(error);
      error = nullptr;
      if (send_error != nullptr) {
        TRITONSERVER_ErrorDelete(send_error);
      }
    }

#ifdef TRITON_ENABLE_STATS
    LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportStatistics(
                     instance_state->TritonModelInstance(), requests[i],
                     request_success, exec_start_ns, exec_start_ns,
                     compute_end_ns, compute_end_ns),
                 "failed reporting request statistics");
#endif

    TRITONBACKEND_RequestRelease(requests[i], TRITONSERVER_REQUEST_RELEASE_ALL);
  }

#ifdef TRITON_ENABLE_STATS
  LOG_IF_ERROR(TRITONBACKEND_ModelInstanceReportBatchStatistics(
                   instance_state->TritonModelInstance(), request_count,
                   exec_start_ns, exec_start_ns, compute_end_ns,
                   compute_end_ns),
               "failed reporting batch statistics");
#else
  (void)exec_start_ns;
  (void)compute_end_ns;
#endif

  return nullptr;
}

} // extern "C"

} // namespace pp_doclayout_v3_preprocess
} // namespace backend
} // namespace triton
