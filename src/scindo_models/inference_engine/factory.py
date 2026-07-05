from __future__ import annotations

from enum import Enum
from pathlib import Path

from scindo_models.artifacts import OnnxRuntimeBundleManifest, read_manifest_as
from scindo_models.inference_engine.base import InferSession
from scindo_models.inference_engine.onnxruntime import OrtInferSession


class EngineType(Enum):
    ONNXRUNTIME = "onnxruntime"


def get_engine(engine_type: EngineType) -> type[InferSession]:
    if engine_type == EngineType.ONNXRUNTIME:
        return OrtInferSession

    raise ValueError(f"Unsupported engine: {engine_type.value}")


def load_engine(artifact_path: Path) -> InferSession:
    manifest = read_manifest_as(artifact_path, OnnxRuntimeBundleManifest)
    engine_class = get_engine(parse_engine_type(manifest.engine))
    return engine_class(artifact_path)


def parse_engine_type(value: str) -> EngineType:
    try:
        return EngineType(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported engine: {value}") from exc
