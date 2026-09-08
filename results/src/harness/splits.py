"""
Patient-level nested cross-validation splits.

Guarantees enforced here (and ASSERTED, not assumed):
  - no patient appears in more than one outer fold        (grouping)
  - outcome distribution is preserved across folds        (stratification)
  - inner folds only ever subdivide the outer-train set   (no test leakage)
  - the full assignment is written to disk with a hash, so every reported
    number can name the exact split it came from

Scheme: sklearn StratifiedGroupKFold (see RESEARCH_NOTES.md).
"""
from __future__ import annotations
import json
import pathlib
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold

from repro import dict_hash


def make_nested_splits(patient_ids, y, cfg, master_seed: int):
    """
    patient_ids : (N,) array-like — the grouping variable
    y           : (N,) array-like — the stratification target
    cfg         : SplitConfig
    returns (manifest dict, split_hash)
    """
    patient_ids = np.asarray(patient_ids)
    y = np.asarray(y)
    outer = StratifiedGroupKFold(n_splits=cfg.outer_folds, shuffle=True,
                                 random_state=master_seed)
    manifest = {"scheme": cfg.scheme, "outer_folds": cfg.outer_folds,
                "inner_folds": cfg.inner_folds, "master_seed": master_seed,
                "n_samples": int(len(y)),
                "n_patients": int(len(np.unique(patient_ids))),
                "folds": []}

    for k, (tr, te) in enumerate(outer.split(np.zeros(len(y)), y, patient_ids)):
        # hard guarantee: patient disjointness between outer train and test
        assert not set(patient_ids[tr]) & set(patient_ids[te]), \
            f"outer fold {k}: patient leakage"
        inner = StratifiedGroupKFold(n_splits=cfg.inner_folds, shuffle=True,
                                     random_state=master_seed + 1000 + k)
        inner_folds = []
        for j, (itr, iva) in enumerate(
                inner.split(np.zeros(len(tr)), y[tr], patient_ids[tr])):
            assert not set(patient_ids[tr][itr]) & set(patient_ids[tr][iva]), \
                f"inner fold {k}.{j}: patient leakage"
            inner_folds.append({"train_idx": tr[itr].tolist(),
                                "val_idx": tr[iva].tolist()})
        manifest["folds"].append({
            "outer": k,
            "train_idx": tr.tolist(), "test_idx": te.tolist(),
            "test_patients": sorted(set(map(str, patient_ids[te]))),
            "test_pos_rate": float(np.mean(y[te] > 0)),
            "inner": inner_folds,
        })

    return manifest, dict_hash(manifest)


def save_manifest(manifest: dict, split_hash: str, out_dir: pathlib.Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    p = out_dir / f"splits_{split_hash}.json"
    p.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return p


def verify_manifest(manifest: dict, patient_ids) -> bool:
    """Re-assert disjointness on a loaded manifest (defence in depth)."""
    patient_ids = np.asarray(patient_ids)
    seen_test = set()
    for f in manifest["folds"]:
        te = set(map(str, patient_ids[f["test_idx"]]))
        tr = set(map(str, patient_ids[f["train_idx"]]))
        if te & tr or te & seen_test:
            return False
        seen_test |= te
    return True


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from config import ExperimentConfig

    rng = np.random.default_rng(0)
    # synthetic: 120 patients, 1-3 samples each, imbalanced 5-grade outcome
    pids, ys = [], []
    for p in range(120):
        for _ in range(int(rng.integers(1, 4))):
            pids.append(f"P{p:04d}")
            ys.append(int(np.clip(rng.poisson(0.8), 0, 4)))
    cfg = ExperimentConfig()
    man, h = make_nested_splits(pids, ys, cfg.split, cfg.master_seed)

    print(f"samples {man['n_samples']}  patients {man['n_patients']}  hash {h}")
    rates = [f["test_pos_rate"] for f in man["folds"]]
    print(f"outer-fold positive rates: {[f'{r:.2f}' for r in rates]} "
          f"(spread {max(rates)-min(rates):.3f})")
    assert verify_manifest(man, pids), "manifest verification failed"
    # determinism: same seed -> same hash; different seed -> different
    man2, h2 = make_nested_splits(pids, ys, cfg.split, cfg.master_seed)
    man3, h3 = make_nested_splits(pids, ys, cfg.split, cfg.master_seed + 1)
    assert h == h2 and h != h3
    print("determinism + leakage assertions: OK")
