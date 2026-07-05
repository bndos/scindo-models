from __future__ import annotations

import argparse
from pathlib import Path

from scindo_models.builders import ArtifactMaterializer
from scindo_models.model_spec import (
    FetchHuggingFaceBuildSpec,
    OnnxRuntimeBundleBuildSpec,
    TritonOnnxBuildSpec,
)
from scindo_models.registry import (
    DEFAULT_REGISTRY_PATH,
    ModelRegistry,
    load_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(prog="scindo-models")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_parser = subparsers.add_parser("inspect")
    inspect_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))

    optimize_parser = subparsers.add_parser("optimize")
    optimize_parser.add_argument("model")
    optimize_parser.add_argument("artifact")
    optimize_parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))

    args = parser.parse_args()
    registry = load_registry(Path(args.registry))
    if args.command == "inspect":
        _inspect(registry)
        return
    if args.command == "optimize":
        _optimize(
            registry=registry,
            model_name=args.model,
            artifact_name=args.artifact,
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


def _inspect(registry: ModelRegistry) -> None:
    for model in registry.models.values():
        print(model.name)
        for artifact in model.artifacts.values():
            print(
                f"  artifact {artifact.name}: {artifact.kind.value} -> {artifact.path}"
            )
        for profile in model.build_profiles.values():
            match profile:
                case FetchHuggingFaceBuildSpec():
                    print(
                        f"  build {profile.name}: {profile.builder.value}, "
                        f"repo={profile.repo_id}, output={profile.output}"
                    )
                case OnnxRuntimeBundleBuildSpec():
                    providers = ", ".join(profile.providers)
                    print(
                        f"  build {profile.name}: {profile.builder.value}, "
                        f"input={profile.input}, output={profile.output}, "
                        f"providers=[{providers}]"
                    )
                case TritonOnnxBuildSpec():
                    print(
                        f"  build {profile.name}: {profile.builder.value}, "
                        f"input={profile.input}, output={profile.output}, "
                        f"model={profile.model_name}"
                    )


def _optimize(
    registry: ModelRegistry,
    model_name: str,
    artifact_name: str,
) -> None:
    model = registry.model(model_name)
    artifact_path = ArtifactMaterializer(model).ensure(artifact_name).path
    print(f"{model_name}/{artifact_name}: {artifact_path}")


if __name__ == "__main__":
    main()
