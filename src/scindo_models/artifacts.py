from __future__ import annotations

import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Literal


MANIFEST_FILE = "artifact.json"


class ArtifactType(Enum):
    ONNX_MODEL = "onnx_model"
    ONNXRUNTIME_BUNDLE = "onnxruntime_bundle"


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


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.mkdir(parents=True, exist_ok=True)
    with (path / MANIFEST_FILE).open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write("\n")
