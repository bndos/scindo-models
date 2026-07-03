from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Protocol

import numpy as np

from scindo_models.inference_engine.base import InferSession, TensorMap


class ModelType(Enum):
    PP_DOCLAYOUT_V3 = "pp_doclayout_v3"


class ScindoModel(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> TensorMap:
        pass


class ScindoModelClass(Protocol):
    def __call__(self, session: InferSession) -> ScindoModel:
        pass


def get_model(model_type: ModelType) -> ScindoModelClass:
    if model_type == ModelType.PP_DOCLAYOUT_V3:
        from scindo_models.models.pp_doclayout_v3 import PPDocLayoutV3

        return PPDocLayoutV3

    raise ValueError(f"Unsupported model: {model_type.value}")


def parse_model_type(value: str) -> ModelType:
    try:
        return ModelType(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported model: {value}") from exc
