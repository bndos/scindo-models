from __future__ import annotations

import shutil
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from scindo_models.artifacts import (
    ArtifactModel,
    ArtifactType,
    MANIFEST_FILE,
    OnnxModelFiles,
    OnnxModelManifest,
    OnnxRuntimeBundleManifest,
    OnnxRuntimeConfig,
    read_manifest,
)


class BuildType(Enum):
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

    def ensure_artifact(self, name: str) -> Path:
        artifact = self.artifact(name)
        if self._is_materialized(artifact):
            return artifact.path

        self._materialize(self.build_profile(artifact.build))
        return artifact.path

    def _materialize(self, build_profile: BuildProfileSpec) -> None:
        match build_profile.builder:
            case BuildType.FETCH_HUGGINGFACE:
                self._materialize_fetch_huggingface(build_profile)
            case BuildType.ONNXRUNTIME_BUNDLE:
                self._materialize_onnxruntime_bundle(build_profile)

    def _materialize_fetch_huggingface(
        self,
        build_profile: FetchHuggingFaceBuildSpec,
    ) -> None:
        output_artifact = self.artifact(build_profile.output)
        fetched_path = _fetch_huggingface(
            repo_id=build_profile.repo_id,
            revision=build_profile.revision,
            filename=build_profile.file,
            local_dir=output_artifact.path,
        )
        model_path = output_artifact.path / build_profile.file
        if fetched_path.resolve() != model_path.resolve():
            shutil.copy2(fetched_path, model_path)
        OnnxModelManifest(
            kind=ArtifactType.ONNX_MODEL,
            model=ArtifactModel(type=self.model_type),
            files=OnnxModelFiles(model=Path(build_profile.file)),
        ).write(output_artifact.path)

    def _materialize_onnxruntime_bundle(
        self,
        build_profile: OnnxRuntimeBundleBuildSpec,
    ) -> None:
        input_path = self.ensure_artifact(build_profile.input)
        input_manifest = read_manifest(input_path)
        output_artifact = self.artifact(build_profile.output)

        output_artifact.path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            input_path / input_manifest.model_path,
            output_artifact.path / input_manifest.model_path,
        )
        _create_provider_paths(output_artifact.path, build_profile.provider_options)
        OnnxRuntimeBundleManifest(
            kind=ArtifactType.ONNXRUNTIME_BUNDLE,
            model=ArtifactModel(type=self.model_type),
            runtime=OnnxRuntimeConfig(
                engine="onnxruntime",
                model_file=input_manifest.model_path,
                providers=build_profile.providers,
                provider_options=build_profile.provider_options,
                outputs=build_profile.outputs,
            ),
        ).write(output_artifact.path)

    def _is_materialized(self, artifact: ArtifactSpec) -> bool:
        if artifact.kind == ArtifactType.ONNX_MODEL:
            if not (artifact.path / MANIFEST_FILE).is_file():
                return False
            manifest = read_manifest(artifact.path)
            return (artifact.path / manifest.model_path).is_file()
        return (artifact.path / MANIFEST_FILE).is_file()


def parse_artifact_type(value: str) -> ArtifactType:
    try:
        return ArtifactType(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported artifact type: {value}") from exc


def _fetch_huggingface(
    repo_id: str,
    revision: str,
    filename: str,
    local_dir: Path,
) -> Path:
    from huggingface_hub import hf_hub_download

    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=local_dir,
    )
    return Path(downloaded_path)


def _create_provider_paths(
    artifact_path: Path,
    provider_options: dict[str, dict[str, object]],
) -> None:
    for options in provider_options.values():
        cache_path = options.get("trt_engine_cache_path")
        if isinstance(cache_path, str):
            (artifact_path / cache_path).mkdir(parents=True, exist_ok=True)
