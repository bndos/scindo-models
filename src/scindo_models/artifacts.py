from __future__ import annotations

import json
from enum import Enum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

MANIFEST_FILE = "artifact.json"


class ArtifactType(str, Enum):
    ONNX_MODEL = "onnx_model"
    ONNXRUNTIME_BUNDLE = "onnxruntime_bundle"


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactManifestBase(StrictModel):
    def write(self, artifact_path: Path) -> None:
        artifact_path.mkdir(parents=True, exist_ok=True)
        with (artifact_path / MANIFEST_FILE).open("w", encoding="utf-8") as f:
            json.dump(self.model_dump(mode="json"), f, indent=2, sort_keys=True)
            f.write("\n")


class ArtifactModel(StrictModel):
    type: str


class OnnxModelFiles(StrictModel):
    model: Path


class OnnxRuntimeConfig(StrictModel):
    engine: Literal["onnxruntime"]
    model_file: Path
    providers: tuple[str, ...]
    provider_options: dict[str, dict[str, object]] = Field(default_factory=dict)
    outputs: tuple[str, ...] | None = None


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


ArtifactManifest = Annotated[
    OnnxModelManifest | OnnxRuntimeBundleManifest,
    Field(discriminator="kind"),
]
_ARTIFACT_MANIFEST_ADAPTER: TypeAdapter[ArtifactManifest] = TypeAdapter(
    ArtifactManifest
)


def read_manifest(artifact_path: Path) -> ArtifactManifest:
    with (artifact_path / MANIFEST_FILE).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    try:
        return _ARTIFACT_MANIFEST_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid artifact manifest: {artifact_path}") from exc


def read_onnxruntime_manifest(artifact_path: Path) -> OnnxRuntimeBundleManifest:
    manifest = read_manifest(artifact_path)
    match manifest.kind:
        case ArtifactType.ONNXRUNTIME_BUNDLE:
            return manifest
        case ArtifactType.ONNX_MODEL:
            raise ValueError(f"Artifact is not an ONNX Runtime bundle: {artifact_path}")
