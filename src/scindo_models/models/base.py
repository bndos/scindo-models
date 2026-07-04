from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

import numpy as np

from scindo_models.inference_engine.base import InferSession, TensorMap


class ScindoModel(ABC):
    @abstractmethod
    def __call__(self, image: np.ndarray) -> TensorMap:
        pass


class ScindoModelClass(Protocol):
    def __call__(self, session: InferSession) -> ScindoModel:
        pass
