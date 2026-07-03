from __future__ import annotations

import json
import shutil
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
MODEL_FILE = "inference.onnx"


@dataclass(frozen=True)
class ArtifactManifest:
    kind: ArtifactType
    model_type: str
    engine: str
    model_file: str
    providers: tuple[str, ...]
    provider_options: dict[str, dict[str, object]]
    outputs: tuple[str, ...] | None

    @property
    def model_path(self) -> Path:
        return Path(self.model_file)


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

    runtime = raw["runtime"]
    return ArtifactManifest(
        kind=ArtifactType(raw["kind"]),
        model_type=raw["model"]["type"],
        engine=runtime["engine"],
        model_file=runtime["model_file"],
        providers=tuple(runtime["providers"]),
        provider_options=dict(runtime.get("provider_options", {})),
        outputs=tuple(runtime["outputs"])
        if runtime.get("outputs") is not None
        else None,
    )


def get_fetcher(source_type: SourceType) -> ArtifactFetcher:
    match source_type:
        case SourceType.HUGGINGFACE:
            return HuggingFaceArtifactFetcher()


def _is_materialized(artifact: ArtifactSpec) -> bool:
    if artifact.kind == ArtifactType.ONNX_MODEL:
        return (artifact.path / MODEL_FILE).is_file()
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
    model_path = output_artifact.path / MODEL_FILE
    if fetched_path.resolve() != model_path.resolve():
        shutil.copy2(fetched_path, model_path)


def _materialize_onnxruntime_bundle(
    model: ModelSpec,
    build_profile: OnnxRuntimeBundleBuildSpec,
) -> None:
    input_artifact = model.artifact(build_profile.input)
    output_artifact = model.artifact(build_profile.output)
    input_path = ensure_artifact(model, input_artifact)

    output_artifact.path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path / MODEL_FILE, output_artifact.path / MODEL_FILE)
    _create_provider_paths(output_artifact.path, build_profile.provider_options)

    _write_manifest(
        output_artifact.path,
        {
            "kind": output_artifact.kind.value,
            "model": {"type": model.model_type},
            "runtime": {
                "engine": "onnxruntime",
                "model_file": MODEL_FILE,
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
