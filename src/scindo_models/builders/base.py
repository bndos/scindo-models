from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MaterializedArtifact:
    path: Path


class ArtifactBuilder(ABC):
    @property
    @abstractmethod
    def input_artifact(self) -> str | None:
        pass

    @abstractmethod
    def build(self) -> None:
        pass

    @abstractmethod
    def is_materialized(self) -> bool:
        pass
