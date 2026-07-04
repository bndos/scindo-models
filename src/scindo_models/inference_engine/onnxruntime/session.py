from __future__ import annotations

from pathlib import Path
from typing import Any

from scindo_models.artifacts import (
    OnnxRuntimeBundleManifest,
    read_manifest_as,
)
from scindo_models.inference_engine.base import (
    InferSession,
    TensorMap,
    require_dense_outputs,
)


class OrtInferSession(InferSession):
    def __init__(self, artifact_path: Path):
        import onnxruntime as ort  # noqa: PLC0415

        self.artifact_path = artifact_path
        self.manifest = read_manifest_as(artifact_path, OnnxRuntimeBundleManifest)
        self.model_path = artifact_path / self.manifest.model_path

        available = set(ort.get_available_providers())
        missing = [
            provider
            for provider in self.manifest.providers
            if provider not in available
        ]
        if missing:
            raise RuntimeError(
                f"Missing ONNX Runtime providers for {artifact_path}: {missing}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path),
            providers=self._providers(artifact_path, self.manifest),
        )

    @staticmethod
    def _providers(
        artifact_path: Path,
        manifest: OnnxRuntimeBundleManifest,
    ) -> list[Any]:
        providers: list[Any] = []
        for provider in manifest.providers:
            options = dict(manifest.provider_options.get(provider, {}))
            if "trt_engine_cache_path" in options:
                options["trt_engine_cache_path"] = str(
                    artifact_path / str(options["trt_engine_cache_path"])
                )
            providers.append((provider, options) if options else provider)
        return providers

    @property
    def input_name(self) -> str:
        return self.session.get_inputs()[0].name

    def __call__(self, inputs: TensorMap) -> TensorMap:
        output_names = list(self.manifest.outputs or ())
        if not output_names:
            output_names = [output.name for output in self.session.get_outputs()]
        outputs = self.session.run(output_names, inputs)
        return require_dense_outputs(output_names, outputs)
