"""
Generic nested-CV trainer.

One function, `run_arm`, executes one ablation arm:
    for each outer fold:
        for each seed:
            train on inner-train, early-stop on inner-val AUC,
            predict the untouched outer-test once
    aggregate patient-level metrics; write run_dir artefacts

Everything variable comes from ExperimentConfig. The model is supplied by a
FACTORY the caller passes in, so this file knows nothing about architectures
— it cannot hardcode model details even by accident.

Artefacts per arm (out_dir/<arm>/):
    config.json         the exact resolved config
    fingerprint.json    software + git + split hash
    predictions.npz     out-of-fold predictions per seed
    metrics.json        the standard metric block + per-seed spread
"""
from __future__ import annotations
import json
import pathlib
import sys
import time

import numpy as np
import torch

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from config import ExperimentConfig, save_config          # noqa: E402
from repro import set_all_seeds, fingerprint, dict_hash   # noqa: E402
from splits import make_nested_splits, save_manifest, verify_manifest  # noqa: E402
import metrics as M                                        # noqa: E402


def _device(cfg):
    if cfg.train.device != "auto":
        return torch.device(cfg.train.device)
    # Quantum arms resolve to CPU: PennyLane default.qubit simulates on CPU
    # (mixing devices crashes), and the measured benchmark shows CPU beats GPU
    # below ~20 qubits anyway (memory: quantum-sim-backend-choice).
    if cfg.model.mixer == "quantum":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _epoch(model, loader, loss_fn, opt=None, device="cpu"):
    training = opt is not None
    model.train(training)
    tot, n = 0.0, 0
    ys, ps = [], []
    with torch.set_grad_enabled(training):
        for batch, y in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            y = y.to(device)
            out = model(batch)
            loss = loss_fn(out, y)
            if training:
                opt.zero_grad()
                loss.backward()
                opt.step()
            tot += float(loss) * len(y)
            n += len(y)
            ys.append(y.detach().cpu())
            ps.append(out["risk"].detach().cpu())
    return tot / max(n, 1), torch.cat(ys).numpy(), torch.cat(ps).numpy()


def run_arm(arm_name: str, cfg: ExperimentConfig, model_factory,
            dataset_factory, loss_factory, quiet=False):
    """
    model_factory(cfg)                    -> nn.Module
    dataset_factory(cfg)                  -> (data, y_binary, y_grade, patient_ids)
                                             data indexable -> (batch_dict, y)
    loss_factory(cfg)                     -> callable(out, y) -> scalar loss
    """
    t0 = time.time()
    out_dir = cfg.resolve_out() / arm_name
    out_dir.mkdir(parents=True, exist_ok=True)
    dev = _device(cfg)

    data, y_bin, y_grade, pids = dataset_factory(cfg)
    manifest, split_hash = make_nested_splits(pids, y_grade, cfg.split,
                                              cfg.master_seed)
    assert verify_manifest(manifest, pids)
    save_manifest(manifest, split_hash, out_dir)
    save_config(cfg, out_dir / "config.json")
    (out_dir / "fingerprint.json").write_text(json.dumps(
        fingerprint({"split_hash": split_hash, "arm": arm_name}), indent=2),
        encoding="utf-8")

    def subset_loader(idx, seed, shuffle):
        g = torch.Generator().manual_seed(seed)
        sub = torch.utils.data.Subset(data, list(idx))
        return torch.utils.data.DataLoader(
            sub, batch_size=cfg.train.batch_size, shuffle=shuffle,
            generator=g if shuffle else None)

    per_seed_oof = {}
    for seed in cfg.train.seeds:
        oof = np.full(len(y_bin), np.nan)
        for fold in manifest["folds"]:
            set_all_seeds(seed * 10_000 + fold["outer"])
            model = model_factory(cfg).to(dev)
            loss_fn = loss_factory(cfg)
            opt = torch.optim.Adam(model.parameters(), lr=cfg.train.lr,
                                   weight_decay=cfg.train.weight_decay)
            # first inner split provides the early-stop validation set
            inner = fold["inner"][0]
            tr_loader = subset_loader(inner["train_idx"], seed, shuffle=True)
            va_loader = subset_loader(inner["val_idx"], seed, shuffle=False)

            best, best_state, patience = -np.inf, None, 0
            for epoch in range(cfg.train.max_epochs):
                _epoch(model, tr_loader, loss_fn, opt, dev)
                _, yv, pv = _epoch(model, va_loader, loss_fn, None, dev)
                try:
                    from sklearn.metrics import roc_auc_score
                    score = roc_auc_score((yv > 0).astype(int), pv)
                except ValueError:
                    score = -np.inf
                if score > best:
                    best, patience = score, 0
                    best_state = {k: v.detach().clone()
                                  for k, v in model.state_dict().items()}
                else:
                    patience += 1
                    if patience >= cfg.train.early_stop_patience:
                        break
            if best_state is not None:
                model.load_state_dict(best_state)

            te_loader = subset_loader(fold["test_idx"], seed, shuffle=False)
            _, _, pt = _epoch(model, te_loader, loss_fn, None, dev)
            oof[np.asarray(fold["test_idx"])] = pt
        per_seed_oof[seed] = oof
        if not quiet:
            from sklearn.metrics import roc_auc_score
            print(f"  [{arm_name}] seed {seed}: "
                  f"AUC {roc_auc_score(y_bin, oof):.3f}")

    # aggregate: mean prediction across seeds -> the arm's headline number;
    # per-seed AUCs give the spread
    P = np.stack([per_seed_oof[s] for s in cfg.train.seeds])
    p_mean = P.mean(0)
    from sklearn.metrics import roc_auc_score
    seed_aucs = [roc_auc_score(y_bin, P[i]) for i in range(len(P))]
    # risk IS the expected ordinal grade (CORN head), range 0..K-1.
    # For binary metrics it is normalised to [0,1]; calibration numbers on
    # this normalised risk are score-calibration, stated as such in the paper.
    grade_pred = p_mean
    block = M.summarize(y_bin, np.clip(p_mean / (cfg.model.dr_grades - 1), 0, 1),
                        np.asarray(pids), cfg.eval,
                        y_grade=y_grade, pred_grade=grade_pred)
    block.update({"arm": arm_name, "split_hash": split_hash,
                  "seed_aucs": [float(a) for a in seed_aucs],
                  "seed_auc_mean": float(np.mean(seed_aucs)),
                  "seed_auc_sd": float(np.std(seed_aucs)),
                  "runtime_s": round(time.time() - t0, 1)})
    np.savez_compressed(out_dir / "predictions.npz",
                        y=y_bin, y_grade=y_grade, p_mean=p_mean, P=P,
                        pids=np.asarray(pids).astype(str))
    (out_dir / "metrics.json").write_text(json.dumps(block, indent=2),
                                          encoding="utf-8")
    return block
