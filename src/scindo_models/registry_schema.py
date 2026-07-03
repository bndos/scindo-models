from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class HuggingFaceSourceConfig(StrictModel):
    kind: Literal["huggingface"] = Field(
        description="Artifact source backed by the Hugging Face Hub."
    )
    repo_id: str = Field(description="Hugging Face repository id.")
    revision: str = Field(default="main", description="Repository revision to fetch.")


SourceConfig = Annotated[HuggingFaceSourceConfig, Field(discriminator="kind")]


class OnnxModelArtifactConfig(StrictModel):
    kind: Literal["onnx_model"] = Field(description="Raw ONNX model artifact.")
    path: str = Field(description="Artifact directory relative to the model root.")
    build: str = Field(description="Build profile used to materialize this artifact.")


class OnnxRuntimeBundleArtifactConfig(StrictModel):
    kind: Literal["onnxruntime_bundle"] = Field(
        description="Self-contained ONNX Runtime artifact bundle."
    )
    path: str = Field(description="Bundle directory relative to the model root.")
    build: str = Field(description="Build profile used to materialize this bundle.")


ArtifactConfig = Annotated[
    OnnxModelArtifactConfig | OnnxRuntimeBundleArtifactConfig,
    Field(discriminator="kind"),
]


class FetchBuildConfig(StrictModel):
    builder: Literal["fetch"] = Field(
        description="Fetches a source file into a raw artifact."
    )
    source: str = Field(description="Input source name.")
    output: str = Field(description="Output artifact name.")
    file: str = Field(description="Source filename copied into the artifact.")


class OnnxRuntimeBundleBuildConfig(StrictModel):
    builder: Literal["onnxruntime-bundle"] = Field(
        description="Builds an ONNX Runtime artifact bundle from an ONNX artifact."
    )
    input: str = Field(description="Input ONNX artifact name.")
    output: str = Field(description="Output artifact name.")
    providers: list[str] = Field(
        min_length=1,
        description="ONNX Runtime execution providers in priority order.",
    )
    outputs: list[str] | None = Field(
        default=None,
        description="Optional output names to request from the runtime.",
    )
    provider_options: dict[str, dict[str, object]] = Field(
        default_factory=dict,
        description="Provider-specific runtime options.",
    )


BuildProfileConfig = Annotated[
    FetchBuildConfig | OnnxRuntimeBundleBuildConfig,
    Field(discriminator="builder"),
]


class ModelConfig(StrictModel):
    name: str = Field(description="Model registry name.")
    model_type: str = Field(description="Model implementation type.")
    root: str = Field(description="Local root directory for model artifacts.")
    sources: dict[str, SourceConfig] = Field(
        default_factory=dict,
        description="External artifact sources keyed by source name.",
    )
    artifacts: dict[str, ArtifactConfig] = Field(
        description="Deployable artifacts keyed by artifact name."
    )
    build_profiles: dict[str, BuildProfileConfig] = Field(
        default_factory=dict,
        description="Artifact build profiles keyed by name.",
    )

    @model_validator(mode="after")
    def validate_references(self) -> ModelConfig:
        for artifact_name, artifact in self.artifacts.items():
            if artifact.build not in self.build_profiles:
                raise ValueError(
                    f"artifacts.{artifact_name}.build references unknown build: "
                    f"{artifact.build}"
                )

        for profile_name, profile in self.build_profiles.items():
            match profile.builder:
                case "fetch":
                    if profile.source not in self.sources:
                        raise ValueError(
                            f"build_profiles.{profile_name}.source references unknown "
                            f"source: {profile.source}"
                        )
                case "onnxruntime-bundle":
                    if profile.input not in self.artifacts:
                        raise ValueError(
                            f"build_profiles.{profile_name}.input references unknown "
                            f"artifact: {profile.input}"
                        )

            if profile.output not in self.artifacts:
                raise ValueError(
                    f"build_profiles.{profile_name}.output references unknown "
                    f"artifact: {profile.output}"
                )
            output = self.artifacts[profile.output]
            if output.build != profile_name:
                raise ValueError(
                    f"artifacts.{profile.output}.build must reference "
                    f"build_profiles.{profile_name}"
                )

        return self


class RegistryConfig(StrictModel):
    schema_version: Literal[1] = Field(description="Registry schema version.")
    model_files: list[str] = Field(
        min_length=1,
        description="Model config files relative to this registry index.",
    )


class ModelFileConfig(StrictModel):
    schema_version: Literal[1] = Field(description="Model file schema version.")
    model: ModelConfig = Field(description="Single model config.")
