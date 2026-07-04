from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from scindo_models.registry import (
    ArtifactSourceSpec,
    ArtifactSpec,
    ArtifactType,
    BuildType,
    BuildProfileSpec,
    FetchBuildSpec,
    ModelSpec,
    OnnxRuntimeBundleBuildSpec,
    SourceType,
)


MANIFEST_FILE = "artifact.json"


@dataclass(frozen=True)
class OnnxModelManifest:
    kind: Literal[ArtifactType.ONNX_MODEL]
    model_type: str
    model_file: Path

    @property
    def model_path(self) -> Path:
        return self.model_file


@dataclass(frozen=True)
class OnnxRuntimeBundleManifest:
    kind: Literal[ArtifactType.ONNXRUNTIME_BUNDLE]
    model_type: str
    model_file: Path
    engine: Literal["onnxruntime"]
    providers: tuple[str, ...]
    provider_options: dict[str, dict[str, object]]
    outputs: tuple[str, ...] | None

    @property
    def model_path(self) -> Path:
        return self.model_file


ArtifactManifest = OnnxModelManifest | OnnxRuntimeBundleManifest


class ArtifactFetcher(ABC):
    @abstractmethod
    def fetch(self, source: ArtifactSourceSpec, filename: str, local_dir: Path) -> Path:
        pass


class HuggingFaceArtifactFetcher(ArtifactFetcher):
    def fetch(self, source: ArtifactSourceSpec, filename: str, local_dir: Path) -> Path:
        from huggingface_hub import hf_hub_download

        local_dir.mkdir(parents=True, exist_ok=True)
        downloaded_path = hf_hub_download(
            repo_id=source.repo_id,
            filename=filename,
            revision=source.revision,
            local_dir=local_dir,
        )
        return Path(downloaded_path)


def ensure_artifact(model: ModelSpec, artifact: ArtifactSpec) -> Path:
    if _is_materialized(artifact):
        return artifact.path

    _materialize(model, model.build_profile(artifact.build))
    return artifact.path


def read_manifest(artifact_path: Path) -> ArtifactManifest:
    with (artifact_path / MANIFEST_FILE).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    kind = ArtifactType(raw["kind"])
    match kind:
        case ArtifactType.ONNX_MODEL:
            model_file = raw.get("files", {}).get("model")
            if not isinstance(model_file, str):
                raise ValueError(
                    f"Missing model file in artifact manifest: {artifact_path}"
                )
            return OnnxModelManifest(
                kind=ArtifactType.ONNX_MODEL,
                model_type=raw["model"]["type"],
                model_file=Path(model_file),
            )
        case ArtifactType.ONNXRUNTIME_BUNDLE:
            runtime = raw["runtime"]
            return OnnxRuntimeBundleManifest(
                kind=ArtifactType.ONNXRUNTIME_BUNDLE,
                model_type=raw["model"]["type"],
                model_file=Path(runtime["model_file"]),
                engine="onnxruntime",
                providers=tuple(runtime["providers"]),
                provider_options=dict(runtime.get("provider_options", {})),
                outputs=tuple(runtime["outputs"])
                if runtime.get("outputs") is not None
                else None,
            )


def read_onnxruntime_manifest(artifact_path: Path) -> OnnxRuntimeBundleManifest:
    manifest = read_manifest(artifact_path)
    match manifest.kind:
        case ArtifactType.ONNXRUNTIME_BUNDLE:
            return manifest
        case ArtifactType.ONNX_MODEL:
            raise ValueError(f"Artifact is not an ONNX Runtime bundle: {artifact_path}")


def get_fetcher(source_type: SourceType) -> ArtifactFetcher:
    match source_type:
        case SourceType.HUGGINGFACE:
            return HuggingFaceArtifactFetcher()


def _is_materialized(artifact: ArtifactSpec) -> bool:
    if artifact.kind == ArtifactType.ONNX_MODEL:
        if not (artifact.path / MANIFEST_FILE).is_file():
            return False
        manifest = read_manifest(artifact.path)
        return (artifact.path / manifest.model_path).is_file()
    return (artifact.path / MANIFEST_FILE).is_file()


def _materialize(model: ModelSpec, build_profile: BuildProfileSpec) -> None:
    match build_profile.builder:
        case BuildType.FETCH:
            _materialize_fetch(model, build_profile)
        case BuildType.ONNXRUNTIME_BUNDLE:
            _materialize_onnxruntime_bundle(model, build_profile)


def _materialize_fetch(model: ModelSpec, build_profile: FetchBuildSpec) -> None:
    output_artifact = model.artifact(build_profile.output)
    fetched_path = get_fetcher(build_profile.source.kind).fetch(
        build_profile.source,
        build_profile.file,
        output_artifact.path,
    )
    model_path = output_artifact.path / build_profile.file
    if fetched_path.resolve() != model_path.resolve():
        shutil.copy2(fetched_path, model_path)
    _write_manifest(
        output_artifact.path,
        {
            "kind": output_artifact.kind.value,
            "model": {"type": model.model_type},
            "files": {"model": build_profile.file},
        },
    )


def _materialize_onnxruntime_bundle(
    model: ModelSpec,
    build_profile: OnnxRuntimeBundleBuildSpec,
) -> None:
    input_artifact = model.artifact(build_profile.input)
    output_artifact = model.artifact(build_profile.output)
    input_path = ensure_artifact(model, input_artifact)
    input_manifest = read_manifest(input_path)

    output_artifact.path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(
        input_path / input_manifest.model_path,
        output_artifact.path / input_manifest.model_path,
    )
    _create_provider_paths(output_artifact.path, build_profile.provider_options)

    _write_manifest(
        output_artifact.path,
        {
            "kind": output_artifact.kind.value,
            "model": {"type": model.model_type},
            "runtime": {
                "engine": "onnxruntime",
                "model_file": str(input_manifest.model_path),
                "providers": list(build_profile.providers),
                "provider_options": build_profile.provider_options,
                "outputs": list(build_profile.outputs)
                if build_profile.outputs
                else None,
            },
        },
    )


def _create_provider_paths(
    artifact_path: Path,
    provider_options: dict[str, dict[str, object]],
) -> None:
    for options in provider_options.values():
        cache_path = options.get("trt_engine_cache_path")
        if isinstance(cache_path, str):
            (artifact_path / cache_path).mkdir(parents=True, exist_ok=True)


def _write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / MANIFEST_FILE).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
