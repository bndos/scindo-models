from __future__ import annotations

import cv2
import numpy as np
import triton_python_backend_utils as pb_utils


class TritonPythonModel:
    def execute(self, requests):
        responses = []
        for request in requests:
            image_bytes = (
                pb_utils.get_input_tensor_by_name(request, "image").as_numpy().flat[0]
            )
            image = cv2.imdecode(
                np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
            )
            if image is None:
                raise pb_utils.TritonModelException("Failed to decode image bytes")

            original_h, original_w = image.shape[:2]
            resized = cv2.resize(
                image,
                (800, 800),
                interpolation=cv2.INTER_CUBIC,
            ).astype(np.float32)
            output_image = np.transpose(resized / 255.0, (2, 0, 1))[None, ...]
            im_shape = np.array([[800, 800]], dtype=np.float32)
            scale_factor = np.array(
                [[800 / original_h, 800 / original_w]],
                dtype=np.float32,
            )

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("im_shape", im_shape),
                        pb_utils.Tensor("image", output_image),
                        pb_utils.Tensor("scale_factor", scale_factor),
                    ]
                )
            )
        return responses
