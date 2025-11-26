from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


class Model(Protocol):
    def predict(self, X) -> Any: ...


@dataclass
class ModelSpec:
    kind: str
    path: str
    metadata: dict
