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
    TRITON_ONNX = "triton-onnx"
    TRITON_REPO = "triton-repo"


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
class OnnxTransformSpec:
    precision: Literal["fp16"]
    keep_io_fp32: bool


@dataclass(frozen=True)
class OnnxRuntimeBundleBuildSpec:
    name: str
    builder: Literal[BuildType.ONNXRUNTIME_BUNDLE]
    input: str
    output: str
    providers: tuple[str, ...]
    outputs: tuple[str, ...] | None
    provider_options: dict[str, dict[str, object]]
    onnx_transform: OnnxTransformSpec | None


@dataclass(frozen=True)
class TritonOnnxBuildSpec:
    name: str
    builder: Literal[BuildType.TRITON_ONNX]
    input: str
    output: str
    model_name: str
    version: int
    max_batch_size: int


@dataclass(frozen=True)
class TritonRepoPreprocessSpec:
    source: Path
    model_name: str
    version: int
    input_map: dict[str, str]
    output_map: dict[str, str]


@dataclass(frozen=True)
class TritonRepoInferSpec:
    input: str
    model_name: str
    version: int
    max_batch_size: int
    input_map: dict[str, str]
    output_map: dict[str, str]


@dataclass(frozen=True)
class TritonRepoBuildSpec:
    name: str
    builder: Literal[BuildType.TRITON_REPO]
    output: str
    model_name: str
    version: int
    max_batch_size: int
    preprocess: TritonRepoPreprocessSpec
    infer: TritonRepoInferSpec


BuildProfileSpec = (
    FetchHuggingFaceBuildSpec
    | OnnxRuntimeBundleBuildSpec
    | TritonOnnxBuildSpec
    | TritonRepoBuildSpec
)


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
