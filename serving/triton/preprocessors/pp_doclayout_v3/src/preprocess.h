#ifndef PP_DOCLAYOUT_V3_PREPROCESS_H_
#define PP_DOCLAYOUT_V3_PREPROCESS_H_

#include <cstdint>
#include <string>
#include <vector>

extern "C" void *pp_doclayout_v3_decoder_create(int device_id);
extern "C" void pp_doclayout_v3_decoder_destroy(void *decoder_state);

// Preprocesses a batch of encoded images (JPEG/PNG) in parallel using
// nvImageCodec GPU decoding. Each image is resized to 800x800 and normalized
// to [0,1] CHW float format.
//
// image_output        : [count * 3 * 800 * 800] floats, CHW per image
// im_shape_output     : [count * 2] floats
// scale_factor_output : [count * 2] floats
// errors              : output, size count; errors[i] is empty on success
void pp_doclayout_v3_preprocess_batch(
    void *decoder_state, const std::vector<std::vector<std::uint8_t>> &images,
    float *image_output, float *im_shape_output, float *scale_factor_output,
    std::vector<std::string> *errors);

#endif // PP_DOCLAYOUT_V3_PREPROCESS_H_
