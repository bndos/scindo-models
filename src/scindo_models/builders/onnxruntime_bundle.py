from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

from scindo_models.artifacts import (
    ArtifactModel,
    ArtifactType,
    MANIFEST_FILE,
    OnnxModelManifest,
    OnnxRuntimeBundleManifest,
    OnnxRuntimeConfig,
    read_manifest_as,
)
from scindo_models.builders.base import ArtifactBuilder
from scindo_models.inference_engine.base import TensorMap
from scindo_models.inference_engine.factory import load_engine
from scindo_models.model_spec import (
    ArtifactSpec,
    ModelSpec,
    OnnxRuntimeBundleBuildSpec,
)
from scindo_models.models.factory import get_model


@dataclass(frozen=True)
class OnnxRuntimeBundleBuilder(ArtifactBuilder):
    model: ModelSpec
    artifact: ArtifactSpec
    profile: OnnxRuntimeBundleBuildSpec

    @property
    def input_artifact(self) -> str:
        return self.profile.input

    def build(self) -> None:
        input_artifact = self.model.artifact(self.profile.input)
        input_manifest = read_manifest_as(input_artifact.path, OnnxModelManifest)

        self.artifact.path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(
            input_artifact.path / input_manifest.model_path,
            self.artifact.path / input_manifest.model_path,
        )
        self._create_provider_paths()

        OnnxRuntimeBundleManifest(
            kind=ArtifactType.ONNXRUNTIME_BUNDLE,
            model=ArtifactModel(type=self.model.model_type.value),
            runtime=OnnxRuntimeConfig(
                engine="onnxruntime",
                model_file=input_manifest.model_path,
                providers=self.profile.providers,
                provider_options=self.profile.provider_options,
                outputs=self.profile.outputs,
            ),
        ).write(self.artifact.path)

        if self._trt_cache_path() is not None:
            self._materialize_runtime_cache()

    def _materialize_runtime_cache(self) -> TensorMap:
        session = load_engine(self.artifact.path)
        model_class = get_model(self.model.model_type)
        return model_class(session)(model_class.optimization_sample())

    def is_materialized(self) -> bool:
        if not (self.artifact.path / MANIFEST_FILE).is_file():
            return False

        manifest = read_manifest_as(self.artifact.path, OnnxRuntimeBundleManifest)
        if not (self.artifact.path / manifest.model_path).is_file():
            return False

        cache_path = self._trt_cache_path()
        if cache_path is None:
            return True

        return any((self.artifact.path / cache_path).glob("*.engine"))

    def _create_provider_paths(self) -> None:
        cache_path = self._trt_cache_path()
        if cache_path is not None:
            (self.artifact.path / cache_path).mkdir(parents=True, exist_ok=True)

    def _trt_cache_path(self) -> Path | None:
        options = self.profile.provider_options.get("TensorrtExecutionProvider")
        if options is None:
            return None

        if options.get("trt_engine_cache_enable") is not True:
            return None

        cache_path = options.get("trt_engine_cache_path")
        if not isinstance(cache_path, str):
            return None

        return Path(cache_path)
