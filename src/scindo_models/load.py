from __future__ import annotations

from pathlib import Path

from scindo_models.artifacts import ensure_artifact, read_onnxruntime_manifest
from scindo_models.inference_engine.base import get_engine, parse_engine_type
from scindo_models.inference_engine.base import InferSession
from scindo_models.models.base import (
    ScindoModel,
    get_model,
    parse_model_type,
)
from scindo_models.registry import DEFAULT_REGISTRY_PATH, load_registry


def load_model(
    model_name: str,
    artifact_name: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> ScindoModel:
    registry = load_registry(registry_path)
    model = registry.model(model_name)
    artifact = model.artifact(artifact_name)
    artifact_path = ensure_artifact(model, artifact)

    manifest = read_onnxruntime_manifest(artifact_path)
    engine_class = get_engine(parse_engine_type(manifest.engine))
    session: InferSession = engine_class(artifact_path)

    model_class = get_model(parse_model_type(model.model_type))
    return model_class(session)
