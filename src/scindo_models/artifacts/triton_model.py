from __future__ import annotations

from pathlib import Path
from typing import Literal

from scindo_models.artifacts.base import (
    ArtifactManifestBase,
    ArtifactModel,
    ArtifactType,
    StrictModel,
)


class TritonTensorSpec(StrictModel):
    name: str
    dtype: str
    dims: tuple[int, ...]


class TritonModelFiles(StrictModel):
    config: Path
    model: Path


class TritonExecutionAccelerator(StrictModel):
    name: str
    parameters: dict[str, str]


class TritonModelConfig(StrictModel):
    name: str
    platform: Literal["onnxruntime_onnx"]
    default_model_filename: str
    version: int
    max_batch_size: int
    inputs: tuple[TritonTensorSpec, ...]
    outputs: tuple[TritonTensorSpec, ...]
    accelerators: tuple[TritonExecutionAccelerator, ...] = ()


class TritonModelManifest(ArtifactManifestBase):
    kind: Literal[ArtifactType.TRITON_MODEL]
    model: ArtifactModel
    files: TritonModelFiles
    triton: TritonModelConfig

    @property
    def model_path(self) -> Path:
        return self.files.model

    @property
    def model_type(self) -> str:
        return self.model.type
