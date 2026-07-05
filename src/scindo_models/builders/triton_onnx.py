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
    OnnxRuntimeBundleManifest,
    TritonExecutionAccelerator,
    TritonModelConfig,
    TritonModelFiles,
    TritonModelManifest,
    TritonTensorSpec,
    read_manifest_as,
)
from scindo_models.builders.base import ArtifactBuilder
from scindo_models.model_spec import ArtifactSpec, ModelSpec, TritonOnnxBuildSpec


CONFIG_FILE = "config.pbtxt"


@dataclass(frozen=True)
class OnnxTensorMetadata:
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    specs: dict[str, TritonTensorSpec]


def build_onnxruntime_model_dir(
    input_path: Path,
    input_manifest: OnnxRuntimeBundleManifest,
    model_path: Path,
    model_name: str,
    version: int,
    max_batch_size: int,
    repository_path: str = "/models",
) -> TritonModelConfig:
    if model_path.exists():
        shutil.rmtree(model_path)

    model_file = input_manifest.model_path.name
    model_dir = model_path / str(version)
    model_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(input_path / input_manifest.model_path, model_dir / model_file)
    copy_provider_cache_dirs(input_path, input_manifest, model_dir)

    config = onnxruntime_model_config(
        input_path=input_path,
        input_manifest=input_manifest,
        model_name=model_name,
        version=version,
        max_batch_size=max_batch_size,
        repository_path=repository_path,
    )
    write_config(model_path / CONFIG_FILE, config)
    return config


def onnxruntime_model_config(
    input_path: Path,
    input_manifest: OnnxRuntimeBundleManifest,
    model_name: str,
    version: int,
    max_batch_size: int,
    repository_path: str = "/models",
) -> TritonModelConfig:
    tensors = _onnx_tensor_metadata(
        input_path / input_manifest.model_path,
        max_batch_size=max_batch_size,
    )
    output_names = input_manifest.outputs or tensors.outputs
    inputs = tuple(_tensor_spec(tensors.specs, name) for name in tensors.inputs)
    outputs = tuple(_tensor_spec(tensors.specs, name) for name in output_names)
    return TritonModelConfig(
        name=model_name,
        platform="onnxruntime_onnx",
        default_model_filename=input_manifest.model_path.name,
        version=version,
        max_batch_size=max_batch_size,
        inputs=inputs,
        outputs=outputs,
        accelerators=_accelerators(
            input_manifest,
            model_name=model_name,
            version=version,
            repository_path=repository_path,
        ),
    )


def copy_provider_cache_dirs(
    input_path: Path,
    input_manifest: OnnxRuntimeBundleManifest,
    model_dir: Path,
) -> None:
    for options in input_manifest.provider_options.values():
        cache_path = options.get("trt_engine_cache_path")
        if not isinstance(cache_path, str):
            continue

        src = input_path / cache_path
        dst = model_dir / cache_path
        if dst.exists():
            shutil.rmtree(dst)
        if src.is_dir():
            shutil.copytree(src, dst)
        else:
            dst.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class TritonOnnxBuilder(ArtifactBuilder):
    model: ModelSpec
    artifact: ArtifactSpec
    profile: TritonOnnxBuildSpec

    @property
    def input_artifact(self) -> str:
        return self.profile.input

    def build(self) -> None:
        input_artifact = self.model.artifact(self.profile.input)
        input_manifest = read_manifest_as(
            input_artifact.path, OnnxRuntimeBundleManifest
        )

        if self.artifact.path.exists():
            shutil.rmtree(self.artifact.path)

        config = build_onnxruntime_model_dir(
            input_path=input_artifact.path,
            input_manifest=input_manifest,
            model_path=self.artifact.path,
            model_name=self.profile.model_name,
            version=self.profile.version,
            max_batch_size=self.profile.max_batch_size,
        )

        TritonModelManifest(
            kind=ArtifactType.TRITON_MODEL,
            model=ArtifactModel(type=self.model.model_type.value),
            files=TritonModelFiles(
                config=Path(CONFIG_FILE),
                model=Path(str(self.profile.version)) / input_manifest.model_path.name,
            ),
            triton=config,
        ).write(self.artifact.path)

    def is_materialized(self) -> bool:
        if not (self.artifact.path / MANIFEST_FILE).is_file():
            return False

        input_artifact = self.model.artifact(self.profile.input)
        input_manifest = read_manifest_as(
            input_artifact.path, OnnxRuntimeBundleManifest
        )
        try:
            manifest = read_manifest_as(self.artifact.path, TritonModelManifest)
        except ValueError:
            return False
        if manifest.triton != self._config(input_manifest):
            return False

        return (self.artifact.path / manifest.files.config).is_file() and (
            self.artifact.path / manifest.files.model
        ).is_file()

    def _config(self, input_manifest: OnnxRuntimeBundleManifest) -> TritonModelConfig:
        return onnxruntime_model_config(
            input_path=self.model.artifact(self.profile.input).path,
            input_manifest=input_manifest,
            model_name=self.profile.model_name,
            version=self.profile.version,
            max_batch_size=self.profile.max_batch_size,
        )


def _accelerators(
    manifest: OnnxRuntimeBundleManifest,
    model_name: str,
    version: int,
    repository_path: str,
) -> tuple[TritonExecutionAccelerator, ...]:
    accelerators = []
    for provider in manifest.providers:
        match provider:
            case "TensorrtExecutionProvider":
                accelerators.append(
                    TritonExecutionAccelerator(
                        name="tensorrt",
                        parameters=_tensorrt_parameters(
                            manifest.provider_options.get(provider, {}),
                            model_name=model_name,
                            version=version,
                            repository_path=repository_path,
                        ),
                    )
                )
            case "CUDAExecutionProvider":
                accelerators.append(
                    TritonExecutionAccelerator(
                        name="cuda",
                        parameters=_string_parameters(
                            manifest.provider_options.get(provider, {})
                        ),
                    )
                )

    return tuple(accelerators)


def _tensorrt_parameters(
    options: dict[str, object],
    model_name: str,
    version: int,
    repository_path: str,
) -> dict[str, str]:
    parameters = _string_parameters(options)
    cache_path = parameters.get("trt_engine_cache_path")
    if cache_path is not None and not cache_path.startswith("/"):
        parameters["trt_engine_cache_path"] = (
            f"{repository_path}/{model_name}/{version}/{cache_path}"
        )
    return parameters


def _string_parameters(options: dict[str, object]) -> dict[str, str]:
    return {key: _string_parameter(value) for key, value in options.items()}


def _string_parameter(value: object) -> str:
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def _onnx_tensor_metadata(
    model_path: Path,
    max_batch_size: int,
) -> OnnxTensorMetadata:
    try:
        onnx = import_module("onnx")
    except ImportError as exc:
        raise RuntimeError(
            "Triton ONNX config generation requires onnx. Install it in the "
            "artifact build image."
        ) from exc

    model = onnx.load(model_path)
    initializer_names = {initializer.name for initializer in model.graph.initializer}
    input_names = tuple(
        value.name for value in model.graph.input if value.name not in initializer_names
    )
    output_names = tuple(value.name for value in model.graph.output)
    specs: dict[str, TritonTensorSpec] = {}
    for value in (*model.graph.input, *model.graph.output):
        if not value.type.HasField("tensor_type"):
            continue

        tensor = value.type.tensor_type
        dims = _triton_dims(tensor.shape.dim, max_batch_size=max_batch_size)
        specs[value.name] = TritonTensorSpec(
            name=value.name,
            dtype=_triton_dtype(tensor.elem_type, onnx.TensorProto),
            dims=dims,
        )

    return OnnxTensorMetadata(
        inputs=input_names,
        outputs=output_names,
        specs=specs,
    )


def _tensor_spec(
    tensor_specs: dict[str, TritonTensorSpec],
    name: str,
) -> TritonTensorSpec:
    try:
        return tensor_specs[name]
    except KeyError as exc:
        available = ", ".join(sorted(tensor_specs))
        raise ValueError(
            f"Tensor is not declared by the ONNX model: {name}. "
            f"Available tensors: {available}"
        ) from exc


def _triton_dims(
    dims: Any,
    max_batch_size: int,
) -> tuple[int, ...]:
    values = tuple(_triton_dim(dim) for dim in dims)
    if max_batch_size > 0:
        if not values:
            raise ValueError("Batched Triton models require rank >= 1 tensors")
        return values[1:]
    return values


def _triton_dim(dim: Any) -> int:
    if dim.HasField("dim_value"):
        return int(dim.dim_value)
    return -1


def _triton_dtype(elem_type: int, tensor_proto: Any) -> str:
    mapping = {
        tensor_proto.BOOL: "TYPE_BOOL",
        tensor_proto.UINT8: "TYPE_UINT8",
        tensor_proto.UINT16: "TYPE_UINT16",
        tensor_proto.UINT32: "TYPE_UINT32",
        tensor_proto.UINT64: "TYPE_UINT64",
        tensor_proto.INT8: "TYPE_INT8",
        tensor_proto.INT16: "TYPE_INT16",
        tensor_proto.INT32: "TYPE_INT32",
        tensor_proto.INT64: "TYPE_INT64",
        tensor_proto.FLOAT16: "TYPE_FP16",
        tensor_proto.FLOAT: "TYPE_FP32",
        tensor_proto.DOUBLE: "TYPE_FP64",
        tensor_proto.STRING: "TYPE_STRING",
    }
    try:
        return mapping[elem_type]
    except KeyError as exc:
        raise ValueError(f"Unsupported ONNX tensor dtype: {elem_type}") from exc


def write_config(path: Path, config: TritonModelConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_config_pbtxt(config), encoding="utf-8")


def _config_pbtxt(config: TritonModelConfig) -> str:
    lines = [
        f'name: "{config.name}"',
        f"max_batch_size: {config.max_batch_size}",
    ]
    if config.platform is not None:
        lines.append(f'platform: "{config.platform}"')
    if config.backend is not None:
        lines.append(f'backend: "{config.backend}"')
    if config.default_model_filename:
        lines.append(f'default_model_filename: "{config.default_model_filename}"')
    for tensor in config.inputs:
        lines.extend(_tensor_block("input", tensor))
    for tensor in config.outputs:
        lines.extend(_tensor_block("output", tensor))
    if config.accelerators:
        lines.extend(["optimization {", "  execution_accelerators {"])
        for accelerator in config.accelerators:
            lines.extend(_accelerator_block(accelerator))
        lines.extend(["  }", "}"])
    return "\n".join(lines) + "\n"


def _accelerator_block(accelerator: TritonExecutionAccelerator) -> list[str]:
    lines = [
        "    gpu_execution_accelerator: [",
        "      {",
        f'        name: "{accelerator.name}"',
    ]
    for key, value in accelerator.parameters.items():
        lines.extend(
            [
                "        parameters {",
                f'          key: "{key}"',
                f'          value: "{value}"',
                "        }",
            ]
        )
    lines.extend(["      }", "    ]"])
    return lines


def _tensor_block(kind: str, tensor: TritonTensorSpec) -> list[str]:
    dims = ", ".join(str(dim) for dim in tensor.dims)
    return [
        f"{kind} [",
        "  {",
        f'    name: "{tensor.name}"',
        f"    data_type: {tensor.dtype}",
        f"    dims: [ {dims} ]",
        "  }",
        "]",
    ]
