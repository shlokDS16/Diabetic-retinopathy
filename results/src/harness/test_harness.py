"""
Harness integration test: run_arm drives the REAL full model (encoders ->
trunk -> concepts -> mixer -> CORN heads) on synthetic data, for the two
decisive arms, with a shrunk config. Verifies artefacts, split integrity,
and that arms share the identical split (same hash) so comparisons are paired.

Run:  python results/src/harness/test_harness.py
"""
import json
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent))

from config import ExperimentConfig, ABLATION_ARMS, apply_overrides
from trainer import run_arm
from concept_quantum import CornHead, N_CONCEPTS
from test_end_to_end import FullModel, synth


class SynthDataset(torch.utils.data.Dataset):
    """60 patients x 1 sample; a planted signal in hrv+static so learning is
    possible; grade in 0..4 correlated with the signal."""

    def __init__(self, n=60, seed=7):
        g = torch.Generator().manual_seed(seed)
        self.n = n
        self.batches = []
        z = torch.randn(n, generator=g)
        self.grade = torch.clamp((z * 1.2 + 1.6).round().long(), 0, 4)
        for i in range(n):
            b = {k: v[0] for k, v in synth(1).items()}
            b["hrv"] = b["hrv"] + z[i] * 0.8
            b["static"] = b["static"] + z[i] * 0.8
            self.batches.append(b)

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return self.batches[i], self.grade[i]


def dataset_factory(cfg):
    ds = SynthDataset()
    grades = np.array([int(g) for g in ds.grade])
    return ds, (grades > 0).astype(float), grades, \
        np.array([f"P{i:03d}" for i in range(len(ds))])


def model_factory(cfg):
    return FullModel(mixer=cfg.model.mixer)


def loss_factory(cfg):
    def loss_fn(out, y):
        return (CornHead.loss(out["dr_logits"], y)
                + cfg.train.w_align * out["loss_align"]
                + cfg.train.w_domain * out["loss_domain"])
    return loss_fn


if __name__ == "__main__":
    base = ExperimentConfig(name="smoke")
    # shrink for speed — via config, never by editing the trainer
    base = apply_overrides(base, {
        "split.outer_folds": 2, "split.inner_folds": 2,
        "train.seeds": [0, 1], "train.max_epochs": 4,
        "train.early_stop_patience": 2, "train.batch_size": 16,
        "eval.bootstrap_iters": 200, "eval.permutation_iters": 100,
    })

    results = {}
    for arm in ("fusion_q", "fusion_c80"):
        cfg = apply_overrides(base, ABLATION_ARMS[arm])
        print(f"--- arm {arm} (mixer={cfg.model.mixer})")
        results[arm] = run_arm(arm, cfg, model_factory, dataset_factory,
                               loss_factory)

    # --- artefact + pairing checks ------------------------------------------
    out = base.resolve_out()
    for arm in results:
        d = out / arm
        for f in ("config.json", "fingerprint.json", "metrics.json",
                  "predictions.npz"):
            assert (d / f).exists(), f"{arm}: missing {f}"
        assert list(d.glob("splits_*.json")), f"{arm}: missing split manifest"
    h = {arm: results[arm]["split_hash"] for arm in results}
    assert len(set(h.values())) == 1, f"arms not on the same split: {h}"

    print("\n=== SMOKE SUMMARY ===")
    for arm, r in results.items():
        print(f"  {arm:10s} AUC {r['auc']:.3f} [{r['auc_ci95'][0]:.3f},"
              f"{r['auc_ci95'][1]:.3f}]  seeds {r['seed_auc_mean']:.3f}"
              f"±{r['seed_auc_sd']:.3f}  qwk {r.get('qwk', float('nan')):.3f}"
              f"  {r['runtime_s']}s")
    print(f"  shared split hash: {list(h.values())[0]}")
    print("\nHARNESS: PASS")
