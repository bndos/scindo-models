from __future__ import annotations

from abc import ABC, abstractmethod
from enum import Enum
from typing import Protocol

import numpy as np

from scindo_models.inference_engine.base import InferSession, TensorMap


class ModelType(str, Enum):
    PP_DOCLAYOUT_V3 = "pp_doclayout_v3"


class ScindoModel(ABC):
    @classmethod
    @abstractmethod
    def optimization_sample(cls) -> np.ndarray:
        pass

    @abstractmethod
    def __call__(self, image: np.ndarray) -> TensorMap:
        pass


class ScindoModelClass(Protocol):
    @classmethod
    def optimization_sample(cls) -> np.ndarray:
        pass

    def __call__(self, session: InferSession) -> ScindoModel:
        pass
