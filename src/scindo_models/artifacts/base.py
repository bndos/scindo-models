from __future__ import annotations

import json
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict


MANIFEST_FILE = "artifact.json"


class ArtifactType(str, Enum):
    ONNX_MODEL = "onnx_model"
    ONNXRUNTIME_BUNDLE = "onnxruntime_bundle"
    TRITON_MODEL = "triton_model"


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
