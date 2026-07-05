from __future__ import annotations

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from scindo_models.artifacts import ArtifactType
from scindo_models.model_spec import BuildType
from scindo_models.models.base import ModelType


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OnnxModelArtifactConfig(StrictModel):
    kind: Literal[ArtifactType.ONNX_MODEL] = Field(
        description="Raw ONNX model artifact."
    )
    path: str = Field(description="Artifact directory relative to the model root.")
    build: str = Field(description="Build profile used to materialize this artifact.")


class OnnxRuntimeBundleArtifactConfig(StrictModel):
    kind: Literal[ArtifactType.ONNXRUNTIME_BUNDLE] = Field(
        description="Self-contained ONNX Runtime artifact bundle."
    )
    path: str = Field(description="Bundle directory relative to the model root.")
    build: str = Field(description="Build profile used to materialize this bundle.")


class TritonModelArtifactConfig(StrictModel):
    kind: Literal[ArtifactType.TRITON_MODEL] = Field(
        description="Triton model repository artifact."
    )
    path: str = Field(description="Artifact directory relative to the model root.")
    build: str = Field(description="Build profile used to materialize this artifact.")


ArtifactConfig = Annotated[
    OnnxModelArtifactConfig
    | OnnxRuntimeBundleArtifactConfig
    | TritonModelArtifactConfig,
    Field(discriminator="kind"),
]


class FetchHuggingFaceBuildConfig(StrictModel):
    builder: Literal[BuildType.FETCH_HUGGINGFACE] = Field(
        description="Fetches a Hugging Face Hub file into a raw artifact."
    )
    repo_id: str = Field(description="Hugging Face repository id.")
    revision: str = Field(default="main", description="Repository revision to fetch.")
    output: str = Field(description="Output artifact name.")
    file: str = Field(description="Source filename copied into the artifact.")


class OnnxTransformConfig(StrictModel):
    precision: Literal["fp16"] = Field(
        description="Precision transform applied to the ONNX graph before bundling."
    )
    keep_io_fp32: bool = Field(
        default=True,
        description="Keep public model inputs and outputs as FP32 when converting.",
    )


class OnnxRuntimeBundleBuildConfig(StrictModel):
    builder: Literal[BuildType.ONNXRUNTIME_BUNDLE] = Field(
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
    onnx_transform: OnnxTransformConfig | None = Field(
        default=None,
        description="Optional ONNX graph transform applied inside the bundle.",
    )

    @model_validator(mode="after")
    def validate_provider_options(self) -> OnnxRuntimeBundleBuildConfig:
        tensorrt_options = self.provider_options.get("TensorrtExecutionProvider")
        if tensorrt_options is None:
            return self

        unsupported_precision_flags = {
            "trt_bf16_enable",
            "trt_fp16_enable",
            "trt_int8_enable",
        }
        unsupported = unsupported_precision_flags & tensorrt_options.keys()
        if unsupported:
            names = ", ".join(sorted(unsupported))
            raise ValueError(
                "TensorRT 11 uses strongly typed networks; precision must be "
                f"encoded in the model artifact, not provider options: {names}"
            )

        return self


class TritonOnnxBuildConfig(StrictModel):
    builder: Literal[BuildType.TRITON_ONNX] = Field(
        description=(
            "Builds a Triton ONNX Runtime model repository from an ONNX Runtime "
            "bundle artifact."
        )
    )
    input: str = Field(description="Input ONNX Runtime bundle artifact name.")
    output: str = Field(description="Output Triton artifact name.")
    model_name: str = Field(description="Triton model name.")
    version: int = Field(default=1, ge=1, description="Triton model version.")
    max_batch_size: int = Field(
        default=0,
        ge=0,
        description="Triton max_batch_size. Use 0 when model tensors include batch.",
    )


BuildProfileConfig = Annotated[
    FetchHuggingFaceBuildConfig | OnnxRuntimeBundleBuildConfig | TritonOnnxBuildConfig,
    Field(discriminator="builder"),
]


class ModelConfig(StrictModel):
    name: str = Field(description="Model registry name.")
    model_type: ModelType = Field(description="Model implementation type.")
    root: str = Field(description="Local root directory for model artifacts.")
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
            if profile.output not in self.artifacts:
                raise ValueError(
                    f"build_profiles.{profile_name}.output references unknown "
                    f"artifact: {profile.output}"
                )

            match profile:
                case FetchHuggingFaceBuildConfig():
                    if self.artifacts[profile.output].kind != ArtifactType.ONNX_MODEL:
                        raise ValueError(
                            f"build_profiles.{profile_name}.output must reference "
                            "an onnx_model artifact"
                        )
                case OnnxRuntimeBundleBuildConfig():
                    if profile.input not in self.artifacts:
                        raise ValueError(
                            f"build_profiles.{profile_name}.input references unknown "
                            f"artifact: {profile.input}"
                        )
                    if self.artifacts[profile.input].kind != ArtifactType.ONNX_MODEL:
                        raise ValueError(
                            f"build_profiles.{profile_name}.input must reference an "
                            "onnx_model artifact"
                        )
                    if (
                        self.artifacts[profile.output].kind
                        != ArtifactType.ONNXRUNTIME_BUNDLE
                    ):
                        raise ValueError(
                            f"build_profiles.{profile_name}.output must reference "
                            "an onnxruntime_bundle artifact"
                        )
                case TritonOnnxBuildConfig():
                    if profile.input not in self.artifacts:
                        raise ValueError(
                            f"build_profiles.{profile_name}.input references unknown "
                            f"artifact: {profile.input}"
                        )
                    if (
                        self.artifacts[profile.input].kind
                        != ArtifactType.ONNXRUNTIME_BUNDLE
                    ):
                        raise ValueError(
                            f"build_profiles.{profile_name}.input must reference an "
                            "onnxruntime_bundle artifact"
                        )
                    if self.artifacts[profile.output].kind != ArtifactType.TRITON_MODEL:
                        raise ValueError(
                            f"build_profiles.{profile_name}.output must reference "
                            "a triton_model artifact"
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
