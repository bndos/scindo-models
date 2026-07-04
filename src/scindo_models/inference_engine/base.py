from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


TensorMap = dict[str, np.ndarray]


class InferSession(ABC):
    @abstractmethod
    def __init__(self, artifact_path: Path):
        pass

    @property
    @abstractmethod
    def input_name(self) -> str:
        pass

    @abstractmethod
    def __call__(self, inputs: TensorMap) -> TensorMap:
        pass


def require_dense_outputs(
    output_names: list[str],
    outputs: Sequence[Any],
) -> TensorMap:
    dense_outputs: TensorMap = {}
    for name, output in zip(output_names, outputs):
        if not isinstance(output, np.ndarray):
            raise TypeError(f"Expected dense ndarray output for {name}")
        dense_outputs[name] = output
    return dense_outputs
