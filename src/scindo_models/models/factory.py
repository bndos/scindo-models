from __future__ import annotations

from enum import Enum

from scindo_models.models.base import ScindoModelClass
from scindo_models.models.pp_doclayout_v3 import PPDocLayoutV3


class ModelType(Enum):
    PP_DOCLAYOUT_V3 = "pp_doclayout_v3"


def get_model(model_type: ModelType) -> ScindoModelClass:
    if model_type == ModelType.PP_DOCLAYOUT_V3:
        return PPDocLayoutV3

    raise ValueError(f"Unsupported model: {model_type.value}")


def parse_model_type(value: str) -> ModelType:
    try:
        return ModelType(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported model: {value}") from exc
