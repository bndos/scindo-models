from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import tomli
from pydantic import Field

from scindo_models.artifacts import (
    ArtifactModel,
    ArtifactType,
    MANIFEST_FILE,
    OnnxRuntimeBundleManifest,
    TritonModelConfig,
    TritonRepoFiles,
    TritonRepoManifest,
    TritonTensorSpec,
    read_manifest_as,
)
from scindo_models.artifacts.base import StrictModel
from scindo_models.builders.base import ArtifactBuilder
from scindo_models.builders.triton_onnx import (
    build_onnxruntime_model_dir,
)
from scindo_models.model_spec import ArtifactSpec, ModelSpec, TritonRepoBuildSpec


CONFIG_FILE = "config.pbtxt"
MODEL_REPOSITORY_DIR = "model_repository"
BACKENDS_DIR = "backends"


class StageContractConfig(StrictModel):
    backend: str
    max_batch_size: int = Field(ge=0)
    inputs: tuple[TritonTensorSpec, ...]
    outputs: tuple[TritonTensorSpec, ...]
    required_paths: tuple[Path, ...] = ()


@dataclass(frozen=True)
class TritonRepoBuilder(ArtifactBuilder):
    model: ModelSpec
    artifact: ArtifactSpec
    profile: TritonRepoBuildSpec

    @property
    def input_artifact(self) -> str:
        return self.profile.infer.input

    def build(self) -> None:
        input_artifact = self.model.artifact(self.profile.infer.input)
        input_manifest = read_manifest_as(
            input_artifact.path,
            OnnxRuntimeBundleManifest,
        )

        if self.artifact.path.exists():
            shutil.rmtree(self.artifact.path)
        self.artifact.path.mkdir(parents=True, exist_ok=True)
        model_repository_path = self.artifact.path / MODEL_REPOSITORY_DIR

        preprocess_config = self._build_preprocess_stage()
        infer_config = build_onnxruntime_model_dir(
            input_path=input_artifact.path,
            input_manifest=input_manifest,
            model_path=model_repository_path / self.profile.infer.model_name,
            model_name=self.profile.infer.model_name,
            version=self.profile.infer.version,
            max_batch_size=self.profile.infer.max_batch_size,
        )
        ensemble_config = self._ensemble_config(preprocess_config, infer_config)
        ensemble_path = model_repository_path / self.profile.model_name
        (ensemble_path / str(self.profile.version)).mkdir(parents=True, exist_ok=True)
        _write_ensemble_config(
            ensemble_path / CONFIG_FILE,
            ensemble_config,
            preprocess=preprocess_config,
            infer=infer_config,
            preprocess_input_map=self._preprocess_input_map(preprocess_config),
            preprocess_output_map=self._preprocess_output_map(preprocess_config),
            infer_input_map=self._infer_input_map(infer_config),
            infer_output_map=self._infer_output_map(infer_config),
        )

        TritonRepoManifest(
            kind=ArtifactType.TRITON_REPO,
            model=ArtifactModel(type=self.model.model_type.value),
            files=TritonRepoFiles(
                root=Path(MODEL_REPOSITORY_DIR),
                backends=Path(BACKENDS_DIR),
            ),
            models=(preprocess_config, infer_config, ensemble_config),
        ).write(self.artifact.path)

    def is_materialized(self) -> bool:
        if not (self.artifact.path / MANIFEST_FILE).is_file():
            return False
        try:
            manifest = read_manifest_as(self.artifact.path, TritonRepoManifest)
        except ValueError:
            return False

        return all(
            (
                self.artifact.path / manifest.files.root / model.name / CONFIG_FILE
            ).is_file()
            and (
                self.artifact.path
                / manifest.files.root
                / model.name
                / str(model.version)
            ).is_dir()
            for model in manifest.models
        )

    def _build_preprocess_stage(self) -> TritonModelConfig:
        source = self.profile.preprocess.source
        contract = _read_stage_contract(source / "contract.toml")
        output_path = (
            self.artifact.path
            / MODEL_REPOSITORY_DIR
            / self.profile.preprocess.model_name
        )
        subprocess.run(
            [
                "make",
                "-C",
                str(source),
                "build",
                f"OUT={output_path.resolve()}",
                f"MODEL_NAME={self.profile.preprocess.model_name}",
                f"VERSION={self.profile.preprocess.version}",
            ],
            check=True,
        )
        if not (output_path / CONFIG_FILE).is_file():
            raise RuntimeError(
                f"Preprocess project did not write {CONFIG_FILE}: {source}"
            )
        if not (output_path / str(self.profile.preprocess.version)).is_dir():
            raise RuntimeError(
                "Preprocess project did not write version directory: "
                f"{output_path / str(self.profile.preprocess.version)}"
            )
        for required_path in contract.required_paths:
            path = self.artifact.path / required_path
            if not path.exists():
                raise RuntimeError(
                    f"Preprocess project did not write required path: {path}"
                )

        return TritonModelConfig(
            name=self.profile.preprocess.model_name,
            backend=contract.backend,
            default_model_filename="",
            version=self.profile.preprocess.version,
            max_batch_size=contract.max_batch_size,
            inputs=contract.inputs,
            outputs=contract.outputs,
        )

    def _ensemble_config(
        self,
        preprocess: TritonModelConfig,
        infer: TritonModelConfig,
    ) -> TritonModelConfig:
        preprocess_input_map = self._preprocess_input_map(preprocess)
        preprocess_output_map = self._preprocess_output_map(preprocess)
        infer_input_map = self._infer_input_map(infer)
        infer_output_map = self._infer_output_map(infer)
        _validate_stage_maps(
            preprocess=preprocess,
            infer=infer,
            preprocess_input_map=preprocess_input_map,
            preprocess_output_map=preprocess_output_map,
            infer_input_map=infer_input_map,
            infer_output_map=infer_output_map,
        )
        return TritonModelConfig(
            name=self.profile.model_name,
            platform="ensemble",
            default_model_filename="",
            version=self.profile.version,
            max_batch_size=self.profile.max_batch_size,
            inputs=_mapped_tensors(preprocess.inputs, preprocess_input_map),
            outputs=_mapped_tensors(infer.outputs, infer_output_map),
        )

    def _preprocess_input_map(self, config: TritonModelConfig) -> dict[str, str]:
        return _default_tensor_map(config.inputs) | self.profile.preprocess.input_map

    def _preprocess_output_map(self, config: TritonModelConfig) -> dict[str, str]:
        return _default_tensor_map(config.outputs) | self.profile.preprocess.output_map

    def _infer_input_map(self, config: TritonModelConfig) -> dict[str, str]:
        return _default_tensor_map(config.inputs) | self.profile.infer.input_map

    def _infer_output_map(self, config: TritonModelConfig) -> dict[str, str]:
        return _default_tensor_map(config.outputs) | self.profile.infer.output_map


def _read_stage_contract(path: Path) -> StageContractConfig:
    with path.open("rb") as f:
        return StageContractConfig.model_validate(tomli.load(f))


def _default_tensor_map(tensors: tuple[TritonTensorSpec, ...]) -> dict[str, str]:
    return {tensor.name: tensor.name for tensor in tensors}


def _mapped_tensors(
    tensors: tuple[TritonTensorSpec, ...],
    tensor_map: dict[str, str],
) -> tuple[TritonTensorSpec, ...]:
    return tuple(
        TritonTensorSpec(
            name=tensor_map[tensor.name],
            dtype=tensor.dtype,
            dims=tensor.dims,
        )
        for tensor in tensors
    )


def _validate_stage_maps(
    preprocess: TritonModelConfig,
    infer: TritonModelConfig,
    preprocess_input_map: dict[str, str],
    preprocess_output_map: dict[str, str],
    infer_input_map: dict[str, str],
    infer_output_map: dict[str, str],
) -> None:
    _validate_tensor_map(preprocess.inputs, preprocess_input_map, "preprocess input")
    _validate_tensor_map(preprocess.outputs, preprocess_output_map, "preprocess output")
    _validate_tensor_map(infer.inputs, infer_input_map, "inference input")
    _validate_tensor_map(infer.outputs, infer_output_map, "inference output")

    produced = {
        ensemble_name: tensor
        for tensor in preprocess.outputs
        for ensemble_name in (preprocess_output_map[tensor.name],)
    }
    consumed = {
        ensemble_name: tensor
        for tensor in infer.inputs
        for ensemble_name in (infer_input_map[tensor.name],)
    }
    for name, consumer_tensor in consumed.items():
        producer_tensor = produced.get(name)
        if producer_tensor is None:
            raise ValueError(f"No preprocess output mapped to inference input: {name}")
        if producer_tensor.dtype != consumer_tensor.dtype:
            raise ValueError(
                f"Stage dtype mismatch for {name}: "
                f"{producer_tensor.dtype} != {consumer_tensor.dtype}"
            )
        if producer_tensor.dims != consumer_tensor.dims:
            raise ValueError(
                f"Stage shape mismatch for {name}: "
                f"{producer_tensor.dims} != {consumer_tensor.dims}"
            )


def _validate_tensor_map(
    tensors: tuple[TritonTensorSpec, ...],
    tensor_map: dict[str, str],
    context: str,
) -> None:
    names = {tensor.name for tensor in tensors}
    for name in tensor_map:
        if name not in names:
            raise ValueError(f"Unknown {context} tensor in map: {name}")


def _write_ensemble_config(
    path: Path,
    config: TritonModelConfig,
    preprocess: TritonModelConfig,
    infer: TritonModelConfig,
    preprocess_input_map: dict[str, str],
    preprocess_output_map: dict[str, str],
    infer_input_map: dict[str, str],
    infer_output_map: dict[str, str],
) -> None:
    lines = [
        f'name: "{config.name}"',
        'platform: "ensemble"',
        f"max_batch_size: {config.max_batch_size}",
    ]
    for tensor in config.inputs:
        lines.extend(_tensor_block("input", tensor))
    for tensor in config.outputs:
        lines.extend(_tensor_block("output", tensor))

    lines.extend(
        [
            "ensemble_scheduling {",
            "  step [",
            "    {",
            f'      model_name: "{preprocess.name}"',
            f"      model_version: {preprocess.version}",
        ]
    )
    for tensor in preprocess.inputs:
        lines.extend(
            _map_block("input_map", tensor.name, preprocess_input_map[tensor.name])
        )
    for tensor in preprocess.outputs:
        lines.extend(
            _map_block("output_map", tensor.name, preprocess_output_map[tensor.name])
        )
    lines.extend(
        [
            "    },",
            "    {",
            f'      model_name: "{infer.name}"',
            f"      model_version: {infer.version}",
        ]
    )
    for tensor in infer.inputs:
        lines.extend(_map_block("input_map", tensor.name, infer_input_map[tensor.name]))
    for tensor in infer.outputs:
        lines.extend(
            _map_block("output_map", tensor.name, infer_output_map[tensor.name])
        )
    lines.extend(["    }", "  ]", "}"])

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _map_block(kind: str, key: str, value: str) -> list[str]:
    return [
        f"      {kind} {{",
        f'        key: "{key}"',
        f'        value: "{value}"',
        "      }",
    ]


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
