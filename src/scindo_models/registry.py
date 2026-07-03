from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

import tomli

from scindo_models.registry_schema import (
    ArtifactConfig,
    FetchBuildConfig,
    ModelConfig,
    ModelFileConfig,
    OnnxRuntimeBundleBuildConfig,
    RegistryConfig,
    SourceConfig,
)


DEFAULT_REGISTRY_PATH = Path("models/registry.toml")


class BuildType(Enum):
    FETCH = "fetch"
    ONNXRUNTIME_BUNDLE = "onnxruntime-bundle"


class SourceType(Enum):
    HUGGINGFACE = "huggingface"


class ArtifactType(Enum):
    ONNX_MODEL = "onnx_model"
    ONNXRUNTIME_BUNDLE = "onnxruntime_bundle"


@dataclass(frozen=True)
class ArtifactSourceSpec:
    kind: SourceType
    repo_id: str
    revision: str


@dataclass(frozen=True)
class ArtifactSpec:
    name: str
    kind: ArtifactType
    path: Path
    build: str


@dataclass(frozen=True)
class FetchBuildSpec:
    name: str
    builder: Literal[BuildType.FETCH]
    source: ArtifactSourceSpec
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


BuildProfileSpec = FetchBuildSpec | OnnxRuntimeBundleBuildSpec


@dataclass(frozen=True)
class ModelSpec:
    name: str
    model_type: str
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


@dataclass(frozen=True)
class ModelRegistry:
    models: dict[str, ModelSpec]

    def model(self, name: str) -> ModelSpec:
        try:
            return self.models[name]
        except KeyError as exc:
            raise KeyError(f"Unknown model: {name}") from exc


def load_registry(path: str | Path = DEFAULT_REGISTRY_PATH) -> ModelRegistry:
    registry_path = Path(path)
    with registry_path.open("rb") as f:
        config = RegistryConfig.model_validate(tomli.load(f))

    registry_root = registry_path.parent
    models = {}
    for model_file in config.model_files:
        model_config = _load_model_config(registry_root / model_file)
        if model_config.name in models:
            raise ValueError(f"Duplicate model name: {model_config.name}")
        models[model_config.name] = _model_spec(model_config)

    return ModelRegistry(models=models)


def _load_model_config(path: Path) -> ModelConfig:
    with path.open("rb") as f:
        config = ModelFileConfig.model_validate(tomli.load(f))
    return config.model


def _model_spec(config: ModelConfig) -> ModelSpec:
    root = Path(config.root)
    artifacts = {
        artifact_name: _artifact_spec(
            artifact_name,
            artifact_config,
            root=root,
        )
        for artifact_name, artifact_config in config.artifacts.items()
    }
    build_profiles = {
        profile_name: _build_profile_spec(
            profile_name,
            profile_config,
            sources=config.sources,
        )
        for profile_name, profile_config in config.build_profiles.items()
    }

    return ModelSpec(
        name=config.name,
        model_type=config.model_type,
        root=root,
        artifacts=artifacts,
        build_profiles=build_profiles,
    )


def _artifact_spec(
    name: str,
    config: ArtifactConfig,
    root: Path,
) -> ArtifactSpec:
    return ArtifactSpec(
        name=name,
        kind=parse_artifact_type(config.kind),
        path=root / config.path,
        build=config.build,
    )


def parse_artifact_type(value: str) -> ArtifactType:
    try:
        return ArtifactType(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported artifact type: {value}") from exc


def _source_spec(config: SourceConfig) -> ArtifactSourceSpec:
    match config.kind:
        case "huggingface":
            return ArtifactSourceSpec(
                kind=SourceType.HUGGINGFACE,
                repo_id=config.repo_id,
                revision=config.revision,
            )


def _build_profile_spec(
    name: str,
    config: FetchBuildConfig | OnnxRuntimeBundleBuildConfig,
    sources: dict[str, SourceConfig],
) -> BuildProfileSpec:
    match config.builder:
        case "fetch":
            return FetchBuildSpec(
                name=name,
                builder=BuildType.FETCH,
                source=_source_spec(sources[config.source]),
                output=config.output,
                file=config.file,
            )
        case "onnxruntime-bundle":
            return OnnxRuntimeBundleBuildSpec(
                name=name,
                builder=BuildType.ONNXRUNTIME_BUNDLE,
                input=config.input,
                output=config.output,
                providers=tuple(config.providers),
                outputs=tuple(config.outputs) if config.outputs is not None else None,
                provider_options=config.provider_options,
            )
