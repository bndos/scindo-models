from __future__ import annotations

from pathlib import Path

import numpy as np

from scindo_models.inference_engine.base import TensorMap
from scindo_models.load import load_model
from scindo_models.models.base import ModelType, parse_model_type
from scindo_models.models.pp_doclayout_v3.model import PPDocLayoutV3Config
from scindo_models.registry import DEFAULT_REGISTRY_PATH, load_registry


def optimize_artifact(
    model_name: str,
    artifact_name: str,
    registry_path: str | Path = DEFAULT_REGISTRY_PATH,
) -> TensorMap:
    registry = load_registry(registry_path)
    model_spec = registry.model(model_name)
    model = load_model(model_name, artifact_name, registry_path)
    sample = optimization_sample(parse_model_type(model_spec.model_type))
    return model(sample)


def optimization_sample(model_type: ModelType) -> np.ndarray:
    match model_type:
        case ModelType.PP_DOCLAYOUT_V3:
            size = PPDocLayoutV3Config().input_size
            return np.zeros((size, size, 3), dtype=np.uint8)
