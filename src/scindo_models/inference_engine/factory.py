from __future__ import annotations

from enum import Enum

from scindo_models.inference_engine.base import InferSession
from scindo_models.inference_engine.onnxruntime import OrtInferSession


class EngineType(Enum):
    ONNXRUNTIME = "onnxruntime"


def get_engine(engine_type: EngineType) -> type[InferSession]:
    if engine_type == EngineType.ONNXRUNTIME:
        return OrtInferSession

    raise ValueError(f"Unsupported engine: {engine_type.value}")


def parse_engine_type(value: str) -> EngineType:
    try:
        return EngineType(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported engine: {value}") from exc
