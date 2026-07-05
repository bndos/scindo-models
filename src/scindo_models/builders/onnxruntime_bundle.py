from __future__ import annotations

from importlib import import_module
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
from scindo_models.inference_engine.factory import load_engine
from scindo_models.model_spec import (
    ArtifactSpec,
    ModelSpec,
    OnnxRuntimeBundleBuildSpec,
    OnnxTransformSpec,
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

        if self.artifact.path.exists():
            shutil.rmtree(self.artifact.path)
        self.artifact.path.mkdir(parents=True, exist_ok=True)
        self._write_model(
            input_artifact.path / input_manifest.model_path,
            self.artifact.path / input_manifest.model_path,
        )
        self._create_provider_paths()

        self._manifest(input_manifest).write(self.artifact.path)

        if self._trt_cache_path() is not None:
            self._materialize_runtime_cache()

    def _materialize_runtime_cache(self) -> None:
        session = load_engine(self.artifact.path)
        model_class = get_model(self.model.model_type)
        model_class(session)(model_class.optimization_sample())

    def is_materialized(self) -> bool:
        if not (self.artifact.path / MANIFEST_FILE).is_file():
            return False

        input_artifact = self.model.artifact(self.profile.input)
        input_manifest = read_manifest_as(input_artifact.path, OnnxModelManifest)
        manifest = read_manifest_as(self.artifact.path, OnnxRuntimeBundleManifest)
        if manifest != self._manifest(input_manifest):
            return False

        if not (self.artifact.path / manifest.model_path).is_file():
            return False

        cache_path = self._trt_cache_path()
        if cache_path is None:
            return True

        return any((self.artifact.path / cache_path).glob("*.engine"))

    def _manifest(
        self,
        input_manifest: OnnxModelManifest,
    ) -> OnnxRuntimeBundleManifest:
        return OnnxRuntimeBundleManifest(
            kind=ArtifactType.ONNXRUNTIME_BUNDLE,
            model=ArtifactModel(type=self.model.model_type.value),
            runtime=OnnxRuntimeConfig(
                engine="onnxruntime",
                model_file=input_manifest.model_path,
                providers=self.profile.providers,
                provider_options=self.profile.provider_options,
                outputs=self.profile.outputs,
            ),
        )

    def _create_provider_paths(self) -> None:
        cache_path = self._trt_cache_path()
        if cache_path is not None:
            (self.artifact.path / cache_path).mkdir(parents=True, exist_ok=True)

    def _write_model(self, input_path: Path, output_path: Path) -> None:
        transform = self.profile.onnx_transform
        if transform is None:
            shutil.copy2(input_path, output_path)
            return

        _transform_onnx(input_path, output_path, transform)

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


def _transform_onnx(
    input_path: Path,
    output_path: Path,
    transform: OnnxTransformSpec,
) -> None:
    match transform.precision:
        case "fp16":
            _convert_onnx_to_fp16(
                input_path,
                output_path,
                keep_io_fp32=transform.keep_io_fp32,
            )


def _convert_onnx_to_fp16(
    input_path: Path,
    output_path: Path,
    keep_io_fp32: bool,
) -> None:
    try:
        onnx = import_module("onnx")
        float16 = import_module("onnxconverter_common.float16")
    except ImportError as exc:
        raise RuntimeError(
            "ONNX FP16 transform requires onnx and onnxconverter-common. "
            "Install them in the optimizer image."
        ) from exc

    model = onnx.load(input_path)
    converted = float16.convert_float_to_float16(
        model,
        keep_io_types=keep_io_fp32,
        disable_shape_infer=True,
    )
    _clean_fp16_graph(converted, onnx.TensorProto)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(converted, output_path)


def _clean_fp16_graph(model: Any, tensor_proto: Any) -> None:
    graph_outputs = {output.name for output in model.graph.output}
    for node in model.graph.node:
        if node.op_type != "Cast" or graph_outputs.intersection(node.output):
            continue
        for attr in node.attribute:
            if attr.name == "to" and attr.i == tensor_proto.FLOAT:
                attr.i = tensor_proto.FLOAT16

    del model.graph.value_info[:]
