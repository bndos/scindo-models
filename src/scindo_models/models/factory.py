from __future__ import annotations

from scindo_models.models.base import ModelType, ScindoModelClass
from scindo_models.models.pp_doclayout_v3 import PPDocLayoutV3


def get_model(model_type: ModelType) -> ScindoModelClass:
    if model_type == ModelType.PP_DOCLAYOUT_V3:
        return PPDocLayoutV3

    raise ValueError(f"Unsupported model: {model_type.value}")
