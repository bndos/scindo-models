from __future__ import annotations

import argparse
from pathlib import Path

from scindo_models.optimize import optimize_artifact
from scindo_models.registry import (
    BuildType,
    DEFAULT_REGISTRY_PATH,
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
    if args.command == "inspect":
        _inspect(Path(args.registry))
        return
    if args.command == "optimize":
        _optimize(
            model_name=args.model,
            artifact_name=args.artifact,
            registry_path=Path(args.registry),
        )
        return

    raise ValueError(f"Unsupported command: {args.command}")


def _inspect(registry_path: Path) -> None:
    registry = load_registry(registry_path)
    for model in registry.models.values():
        print(model.name)
        for artifact in model.artifacts.values():
            print(
                f"  artifact {artifact.name}: {artifact.kind.value} -> {artifact.path}"
            )
        for profile in model.build_profiles.values():
            match profile.builder:
                case BuildType.FETCH:
                    print(
                        f"  build {profile.name}: {profile.builder.value}, "
                        f"source={profile.source.kind.value}, output={profile.output}"
                    )
                case BuildType.ONNXRUNTIME_BUNDLE:
                    providers = ", ".join(profile.providers)
                    print(
                        f"  build {profile.name}: {profile.builder.value}, "
                        f"input={profile.input}, output={profile.output}, "
                        f"providers=[{providers}]"
                    )


def _optimize(
    model_name: str,
    artifact_name: str,
    registry_path: Path,
) -> None:
    outputs = optimize_artifact(model_name, artifact_name, registry_path)
    for name, value in outputs.items():
        print(f"{name}: shape={value.shape}, dtype={value.dtype}")


if __name__ == "__main__":
    main()
