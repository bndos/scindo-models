from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field

from scindo_models.artifacts.base import (
    ArtifactManifestBase,
    ArtifactModel,
    ArtifactType,
    StrictModel,
)


class OnnxRuntimeConfig(StrictModel):
    engine: Literal["onnxruntime"]
    model_file: Path
    providers: tuple[str, ...]
    provider_options: dict[str, dict[str, object]] = Field(default_factory=dict)
    outputs: tuple[str, ...] | None = None


class OnnxRuntimeBundleManifest(ArtifactManifestBase):
    kind: Literal[ArtifactType.ONNXRUNTIME_BUNDLE]
    model: ArtifactModel
    runtime: OnnxRuntimeConfig

    @property
    def model_path(self) -> Path:
        return self.runtime.model_file

    @property
    def model_type(self) -> str:
        return self.model.type

    @property
    def engine(self) -> Literal["onnxruntime"]:
        return self.runtime.engine

    @property
    def providers(self) -> tuple[str, ...]:
        return self.runtime.providers

    @property
    def provider_options(self) -> dict[str, dict[str, object]]:
        return self.runtime.provider_options

    @property
    def outputs(self) -> tuple[str, ...] | None:
        return self.runtime.outputs
