import hashlib
import json
from pathlib import Path


def _sha256_path(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def hash_configs(paths=("config/base.yaml",)):
    out = {}
    for p in paths:
        pp = Path(p)
        if pp.exists():
            out[p] = _sha256_path(pp)
    return out


def hash_cache(dir_="data/smoke_cache", exts=("*.parquet",)):
    d = Path(dir_)
    out = {}
    if d.exists():
        for pattern in exts:
            for fp in d.glob(pattern):
                out[str(fp)] = _sha256_path(fp)
    return out


def write_provenance(out_path="reports/smoke_provenance.json", configs=None):
    configs = configs or ["config/base.yaml"]
    prov = {"config_hashes": hash_configs(configs), "cache_hashes": hash_cache()}
    Path(out_path).write_text(json.dumps(prov, indent=2))
    return prov
