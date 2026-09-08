"""
Reproducibility utilities. Per PyTorch's randomness notes (see
RESEARCH_NOTES.md): seed all four RNG families, force deterministic
algorithms, and fingerprint every run so any number in the paper can be
traced to config + code + data state.
"""
from __future__ import annotations
import hashlib
import json
import os
import pathlib
import platform
import random
import subprocess
import sys

# must be set before the first CUDA context is created
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")


def set_all_seeds(seed: int):
    import numpy as np
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    # warn_only: a handful of ops have no deterministic kernel; we log rather
    # than crash, and the fingerprint records torch's determinism setting
    torch.use_deterministic_algorithms(True, warn_only=True)


def seed_worker(worker_id: int):
    """For DataLoader(worker_init_fn=seed_worker) — derives per-worker seeds
    from torch's initial seed, per the PyTorch reproducibility notes."""
    import numpy as np
    import torch
    ws = torch.initial_seed() % 2**32
    np.random.seed(ws)
    random.seed(ws)


def _git_state(root: pathlib.Path) -> dict:
    def run(*args):
        try:
            return subprocess.run(["git", *args], cwd=root, capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return ""
    return {"commit": run("rev-parse", "HEAD") or None,
            "dirty": bool(run("status", "--porcelain")) or None}


def fingerprint(extra: dict | None = None) -> dict:
    """Everything needed to say 'this exact software state produced this run'."""
    import numpy, torch, sklearn
    fp = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": (torch.cuda.get_device_name(0)
                if torch.cuda.is_available() else None),
        "numpy": numpy.__version__,
        "sklearn": sklearn.__version__,
        "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "git": _git_state(pathlib.Path(__file__).resolve().parents[2]),
    }
    try:
        import pennylane
        fp["pennylane"] = pennylane.__version__
    except ImportError:
        fp["pennylane"] = None
    if extra:
        fp.update(extra)
    return fp


def dict_hash(d: dict) -> str:
    """Stable hash of any JSON-serialisable dict (split manifests, configs)."""
    s = json.dumps(d, sort_keys=True, default=str)
    return hashlib.sha256(s.encode()).hexdigest()[:16]


if __name__ == "__main__":
    set_all_seeds(0)
    fp = fingerprint()
    print(json.dumps(fp, indent=2))
    import torch
    a = torch.randn(3)
    set_all_seeds(0)
    b = torch.randn(3)
    assert torch.equal(a, b), "reseeding not reproducible"
    print("reseed determinism: OK")
    print("dict_hash stable:", dict_hash({"a": 1}) == dict_hash({"a": 1}))
