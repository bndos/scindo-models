from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from scindo_models.artifacts import ArtifactType
from scindo_models.models.base import ModelType


class BuildType(str, Enum):
    FETCH_HUGGINGFACE = "fetch-huggingface"
    ONNXRUNTIME_BUNDLE = "onnxruntime-bundle"


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    kind: ArtifactType
    path: Path
    build: str


@dataclass(frozen=True)
class FetchHuggingFaceBuildSpec:
    name: str
    builder: Literal[BuildType.FETCH_HUGGINGFACE]
    repo_id: str
    revision: str
    output: str
    file: str


@dataclass(frozen=True)
class OnnxRuntimeBundleBuildSpec:
    name: str
    builder: Literal[BuildType.ONNXRUNTIME_BUNDLE]
    input: str
    output: str
    providers: tuple[str, ...]
    outputs: tuple[str, ...] | None
    provider_options: dict[str, dict[str, object]]


BuildProfileSpec = FetchHuggingFaceBuildSpec | OnnxRuntimeBundleBuildSpec


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: ModelType
    root: Path
    artifacts: dict[str, ArtifactSpec]
    build_profiles: dict[str, BuildProfileSpec]

    def artifact(self, name: str) -> ArtifactSpec:
        try:
            return self.artifacts[name]
        except KeyError as exc:
            raise KeyError(f"Unknown artifact for {self.name}: {name}") from exc

    def build_profile(self, name: str) -> BuildProfileSpec:
        try:
            return self.build_profiles[name]
        except KeyError as exc:
            raise KeyError(f"Unknown build profile for {self.name}: {name}") from exc
