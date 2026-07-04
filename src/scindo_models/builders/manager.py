from __future__ import annotations

from dataclasses import dataclass

from scindo_models.builders.base import ArtifactBuilder, MaterializedArtifact
from scindo_models.builders.fetch_huggingface import FetchHuggingFaceBuilder
from scindo_models.builders.onnxruntime_bundle import OnnxRuntimeBundleBuilder
from scindo_models.model_spec import (
    ArtifactSpec,
    BuildProfileSpec,
    FetchHuggingFaceBuildSpec,
    ModelSpec,
    OnnxRuntimeBundleBuildSpec,
)


@dataclass(frozen=True)
class ArtifactMaterializer:
    model: ModelSpec

    def ensure(self, artifact_name: str) -> MaterializedArtifact:
        builder = self._builder_for_artifact(artifact_name)
        if builder.is_materialized():
            return MaterializedArtifact(path=self.model.artifact(artifact_name).path)

        if builder.input_artifact is not None:
            self.ensure(builder.input_artifact)

        builder.build()
        if not builder.is_materialized():
            raise RuntimeError(f"Artifact was not materialized: {artifact_name}")

        return MaterializedArtifact(path=self.model.artifact(artifact_name).path)

    def _builder_for_artifact(self, artifact_name: str) -> ArtifactBuilder:
        artifact = self.model.artifact(artifact_name)
        profile = self.model.build_profile(artifact.build)
        return _create_builder(self.model, artifact, profile)


def _create_builder(
    model: ModelSpec,
    artifact: ArtifactSpec,
    profile: BuildProfileSpec,
) -> ArtifactBuilder:
    match profile:
        case FetchHuggingFaceBuildSpec():
            return FetchHuggingFaceBuilder(
                model=model,
                artifact=artifact,
                profile=profile,
            )
        case OnnxRuntimeBundleBuildSpec():
            return OnnxRuntimeBundleBuilder(
                model=model,
                artifact=artifact,
                profile=profile,
            )
