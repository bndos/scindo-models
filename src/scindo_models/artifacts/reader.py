from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, TypeVar

from pydantic import Field, TypeAdapter, ValidationError

from scindo_models.artifacts.base import ArtifactManifestBase, MANIFEST_FILE
from scindo_models.artifacts.onnx_model import OnnxModelManifest
from scindo_models.artifacts.onnxruntime_bundle import OnnxRuntimeBundleManifest


ArtifactManifest = Annotated[
    OnnxModelManifest | OnnxRuntimeBundleManifest,
    Field(discriminator="kind"),
]
_ARTIFACT_MANIFEST_ADAPTER: TypeAdapter[ArtifactManifest] = TypeAdapter(
    ArtifactManifest
)
TArtifactManifest = TypeVar("TArtifactManifest", bound=ArtifactManifestBase)


def read_manifest(artifact_path: Path) -> ArtifactManifest:
    with (artifact_path / MANIFEST_FILE).open("r", encoding="utf-8") as f:
        raw = json.load(f)

    try:
        return _ARTIFACT_MANIFEST_ADAPTER.validate_python(raw)
    except ValidationError as exc:
        raise ValueError(f"Invalid artifact manifest: {artifact_path}") from exc


def read_manifest_as(
    artifact_path: Path,
    manifest_type: type[TArtifactManifest],
) -> TArtifactManifest:
    manifest = read_manifest(artifact_path)
    if not isinstance(manifest, manifest_type):
        raise ValueError(
            f"Expected {manifest_type.__name__} for artifact: {artifact_path}"
        )
    return manifest
