from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np


TensorMap = dict[str, np.ndarray]


class EngineType(Enum):
    ONNXRUNTIME = "onnxruntime"


class InferSession(ABC):
    @abstractmethod
    def __init__(self, artifact_path: Path):
        pass

    @property
    @abstractmethod
    def input_name(self) -> str:
        pass

    @abstractmethod
    def __call__(self, inputs: TensorMap) -> TensorMap:
        pass


def get_engine(engine_type: EngineType) -> type[InferSession]:
    if engine_type == EngineType.ONNXRUNTIME:
        from scindo_models.inference_engine.onnxruntime import OrtInferSession

        return OrtInferSession

    raise ValueError(f"Unsupported engine: {engine_type.value}")


def parse_engine_type(value: str) -> EngineType:
    try:
        return EngineType(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported engine: {value}") from exc


def require_dense_outputs(
    output_names: list[str],
    outputs: Sequence[Any],
) -> TensorMap:
    dense_outputs: TensorMap = {}
    for name, output in zip(output_names, outputs):
        if not isinstance(output, np.ndarray):
            raise TypeError(f"Expected dense ndarray output for {name}")
        dense_outputs[name] = output
    return dense_outputs
