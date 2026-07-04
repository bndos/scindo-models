from __future__ import annotations

from pathlib import Path

from scindo_models.inference_engine.factory import load_engine
from scindo_models.models.base import ScindoModel
from scindo_models.models.factory import (
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
    artifact_path = model.ensure_artifact(artifact_name)
    session = load_engine(artifact_path)

    model_class = get_model(parse_model_type(model.model_type))
    return model_class(session)
