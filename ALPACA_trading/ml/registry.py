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
