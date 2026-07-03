from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from scindo_models.inference_engine.base import InferSession, TensorMap
from scindo_models.models.base import ScindoModel


@dataclass(frozen=True)
class PPDocLayoutV3Config:
    input_size: int = 800


class PPDocLayoutV3(ScindoModel):
    def __init__(
        self,
        session: InferSession,
        config: PPDocLayoutV3Config | None = None,
    ):
        self.session = session
        self.config = config or PPDocLayoutV3Config()

    def __call__(self, image: np.ndarray) -> TensorMap:
        return self.session(self.preprocess(image))

    def preprocess(self, image: np.ndarray) -> TensorMap:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("Expected image with shape HxWx3")

        original_h, original_w = image.shape[:2]
        tensor = cv2.resize(
            image,
            (self.config.input_size, self.config.input_size),
            interpolation=cv2.INTER_CUBIC,
        ).astype(np.float32)
        image_tensor = np.transpose(tensor, (2, 0, 1))[None, ...]

        return {
            "im_shape": np.array(
                [[self.config.input_size, self.config.input_size]],
                dtype=np.float32,
            ),
            "image": image_tensor,
            "scale_factor": np.array(
                [
                    [
                        self.config.input_size / original_h,
                        self.config.input_size / original_w,
                    ]
                ],
                dtype=np.float32,
            ),
        }
