from __future__ import annotations

from pathlib import Path
from typing import Literal

from scindo_models.artifacts.base import (
    ArtifactManifestBase,
    ArtifactModel,
    ArtifactType,
    StrictModel,
)


class OnnxModelFiles(StrictModel):
    model: Path


class OnnxModelManifest(ArtifactManifestBase):
    kind: Literal[ArtifactType.ONNX_MODEL]
    model: ArtifactModel
    files: OnnxModelFiles

    @property
    def model_path(self) -> Path:
        return self.files.model

    @property
    def model_type(self) -> str:
        return self.model.type
