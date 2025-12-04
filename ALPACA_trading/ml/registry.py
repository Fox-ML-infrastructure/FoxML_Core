"""
Copyright (c) 2025 Fox ML Infrastructure

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU Affero General Public License as published
by the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU Affero General Public License for more details.

You should have received a copy of the GNU Affero General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import hashlib
import pickle
from pathlib import Path

from .model_interface import Model, ModelSpec


def sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def load_model(spec: ModelSpec) -> tuple[Model, str]:
    p = Path(spec.path)
    if not p.exists():
        raise FileNotFoundError(p)
    if spec.kind == "pickle":
        with p.open("rb") as f:
            model = pickle.load(f)
    elif spec.kind == "torch":
        import torch  # lazy import

        model = torch.load(p, map_location="cpu")
        if hasattr(model, "eval"):
            model.eval()
    elif spec.kind == "onnx":
        import onnxruntime as ort  # lazy import

        model = ort.InferenceSession(str(p))
    else:
        raise ValueError(f"Unknown model kind: {spec.kind}")
    return model, sha256_path(p)
