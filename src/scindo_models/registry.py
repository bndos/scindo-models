from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import tomli

from scindo_models.model_spec import (
    ArtifactSpec,
    BuildProfileSpec,
    BuildType,
    FetchHuggingFaceBuildSpec,
    ModelSpec,
    OnnxTransformSpec,
    OnnxRuntimeBundleBuildSpec,
    TritonOnnxBuildSpec,
)
from scindo_models.registry_schema import (
    FetchHuggingFaceBuildConfig,
    ModelConfig,
    ModelFileConfig,
    OnnxRuntimeBundleBuildConfig,
    OnnxTransformConfig,
    RegistryConfig,
    TritonOnnxBuildConfig,
)


DEFAULT_REGISTRY_PATH = Path("models/registry.toml")


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
        artifact_name: ArtifactSpec(
            name=artifact_name,
            kind=artifact_config.kind,
            path=root / artifact_config.path,
            build=artifact_config.build,
        )
        for artifact_name, artifact_config in config.artifacts.items()
    }
    build_profiles = {
        profile_name: _build_profile_spec(profile_name, profile_config)
        for profile_name, profile_config in config.build_profiles.items()
    }

    return ModelSpec(
        name=config.name,
        model_type=config.model_type,
        root=root,
        artifacts=artifacts,
        build_profiles=build_profiles,
    )


def _build_profile_spec(
    name: str,
    config: FetchHuggingFaceBuildConfig
    | OnnxRuntimeBundleBuildConfig
    | TritonOnnxBuildConfig,
) -> BuildProfileSpec:
    match config:
        case FetchHuggingFaceBuildConfig():
            return FetchHuggingFaceBuildSpec(
                name=name,
                builder=BuildType.FETCH_HUGGINGFACE,
                repo_id=config.repo_id,
                revision=config.revision,
                output=config.output,
                file=config.file,
            )
        case OnnxRuntimeBundleBuildConfig():
            return OnnxRuntimeBundleBuildSpec(
                name=name,
                builder=BuildType.ONNXRUNTIME_BUNDLE,
                input=config.input,
                output=config.output,
                providers=tuple(config.providers),
                outputs=tuple(config.outputs) if config.outputs is not None else None,
                provider_options=config.provider_options,
                onnx_transform=_onnx_transform_spec(config.onnx_transform),
            )
        case TritonOnnxBuildConfig():
            return TritonOnnxBuildSpec(
                name=name,
                builder=BuildType.TRITON_ONNX,
                input=config.input,
                output=config.output,
                model_name=config.model_name,
                version=config.version,
                max_batch_size=config.max_batch_size,
            )


def _onnx_transform_spec(
    config: OnnxTransformConfig | None,
) -> OnnxTransformSpec | None:
    if config is None:
        return None

    return OnnxTransformSpec(
        precision=config.precision,
        keep_io_fp32=config.keep_io_fp32,
    )
