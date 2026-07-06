# scindo-models

Boilerplate for taking a model from a source export to a servable artifact: fetch, optimize, package for serving. The stages are model-agnostic. `pp_doclayout_v3` (PaddleOCR's document layout detector) is the first model wired through it.

## Pipeline

Each model goes through the same stages, though not every model needs every stage:

```
fetch (HuggingFace) -> onnx -> optimize (ONNXRuntime bundle: CUDA / TensorRT, FP16) -> package for serving
```

- **fetch**: pull a model's ONNX export from HuggingFace.
- **optimize**: build an ONNXRuntime bundle against a specific execution provider (CUDA, TensorRT) and precision, producing something runnable in-process via `InferSession`/`ScindoModel`.
- **package for serving**: wrap an optimized bundle into a Triton-servable artifact. Either a single Triton model, or a full ensemble (`triton-repo`) that adds a custom preprocessing stage in front of the model.

Each stage is declarative: a model's `.toml` file (`models/<model>.toml`) lists `artifacts` (named build outputs) and `build_profiles` (which builder produces each artifact, and with what options). `models/registry.toml` lists which model files exist.

## Models

### pp_doclayout_v3

- Build profiles only target NVIDIA GPU (CUDA and TensorRT execution providers); no CPU profile defined yet.
- Its Triton preprocessor features GPU image decoding via nvImageCodec. See `serving/triton/preprocessors/pp_doclayout_v3/src/preprocess.cc`.

## Usage

```bash
scindo-models inspect                      # list registered models, artifacts, build profiles
scindo-models optimize <model> <artifact>  # build one artifact (and its dependencies)
```

Example: `scindo-models optimize pp_doclayout_v3 triton_service_ort_trt` builds the full Triton ensemble (custom GPU preprocessing backend + TensorRT-optimized inference) into `artifacts/pp_doclayout_v3/triton_service_ort_trt/`.

## Layout

```
models/                       model definitions (registry.toml + one .toml per model)
src/scindo_models/
  builders/                   one builder per artifact kind (fetch-huggingface, onnxruntime-bundle, triton-onnx, triton-repo)
  artifacts/                  typed manifests for each artifact kind, read back by later stages
  inference_engine/           in-process ONNXRuntime session wrapper
  models/                     per-model pre/postprocessing (ScindoModel implementations)
serving/triton/preprocessors/ custom Triton backends (C++ and Python) used by triton-repo build profiles
docker/                       serving images (Dockerfile.triton, Dockerfile.onnxruntime-trt)
```

## Adding a model

Add `models/<name>.toml`, register it in `models/registry.toml`, and implement a `ScindoModel` under `src/scindo_models/models/<name>/` if it needs custom pre/postprocessing beyond what a generic ONNX graph provides. Triton-served models that need preprocessing heavier than what ONNX can express (e.g. image decode) get a custom backend under `serving/triton/preprocessors/<name>/`.
