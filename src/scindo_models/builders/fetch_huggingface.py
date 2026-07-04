from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download

from scindo_models.artifacts import (
    ArtifactModel,
    ArtifactType,
    MANIFEST_FILE,
    OnnxModelFiles,
    OnnxModelManifest,
    read_manifest_as,
)
from scindo_models.builders.base import ArtifactBuilder
from scindo_models.model_spec import ArtifactSpec, FetchHuggingFaceBuildSpec, ModelSpec


@dataclass(frozen=True)
class FetchHuggingFaceBuilder(ArtifactBuilder):
    model: ModelSpec
    artifact: ArtifactSpec
    profile: FetchHuggingFaceBuildSpec

    @property
    def input_artifact(self) -> None:
        return None

    def build(self) -> None:
        fetched_path = _fetch_huggingface(
            repo_id=self.profile.repo_id,
            revision=self.profile.revision,
            filename=self.profile.file,
            local_dir=self.artifact.path,
        )
        model_path = self.artifact.path / self.profile.file
        if fetched_path.resolve() != model_path.resolve():
            shutil.copy2(fetched_path, model_path)

        OnnxModelManifest(
            kind=ArtifactType.ONNX_MODEL,
            model=ArtifactModel(type=self.model.model_type.value),
            files=OnnxModelFiles(model=Path(self.profile.file)),
        ).write(self.artifact.path)

    def is_materialized(self) -> bool:
        if not (self.artifact.path / MANIFEST_FILE).is_file():
            return False

        manifest = read_manifest_as(self.artifact.path, OnnxModelManifest)
        return (self.artifact.path / manifest.model_path).is_file()


def _fetch_huggingface(
    repo_id: str,
    revision: str,
    filename: str,
    local_dir: Path,
) -> Path:
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        revision=revision,
        local_dir=local_dir,
    )
    return Path(downloaded_path)
