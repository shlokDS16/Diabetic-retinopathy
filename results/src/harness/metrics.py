"""
Central metrics for every arm. One implementation, used everywhere, so no two
scripts can disagree about how a number was computed.

All uncertainty is bootstrap at the PATIENT level (EvalConfig.bootstrap_level)
— resampling rows when patients contribute multiple samples understates
variance.
"""
from __future__ import annotations
import numpy as np
from sklearn.metrics import roc_auc_score, brier_score_loss, cohen_kappa_score


def _patient_resample(rng, patient_ids):
    """Sample patients with replacement; return row indices."""
    uniq = np.unique(patient_ids)
    chosen = rng.choice(uniq, size=len(uniq), replace=True)
    rows = []
    by_pid = {}
    for i, p in enumerate(patient_ids):
        by_pid.setdefault(p, []).append(i)
    for p in chosen:
        rows.extend(by_pid[p])
    return np.asarray(rows)


def bootstrap_metric(fn, y, p, patient_ids, iters, ci, seed):
    rng = np.random.default_rng(seed)
    point = fn(y, p)
    vals = []
    for _ in range(iters):
        r = _patient_resample(rng, patient_ids)
        try:
            vals.append(fn(y[r], p[r]))
        except ValueError:            # a resample lost one class
            continue
    a = (1 - ci) / 2 * 100
    lo, hi = np.percentile(vals, [a, 100 - a])
    return float(point), float(lo), float(hi)


def auc_ci(y, p, patient_ids, iters=2000, ci=0.95, seed=20260828):
    return bootstrap_metric(roc_auc_score, y, p, patient_ids, iters, ci, seed)


def qwk_ci(y_grade, pred_grade, patient_ids, iters=2000, ci=0.95, seed=20260828):
    fn = lambda a, b: cohen_kappa_score(a, np.round(b).astype(int),
                                        weights="quadratic")
    return bootstrap_metric(fn, y_grade, pred_grade, patient_ids, iters, ci, seed)


def calibration_intercept_slope(y, p, eps=1e-6):
    """Logistic recalibration: intercept from an offset model
    (calibration-in-the-large, target 0) and slope (target 1).
    Both must be reported together — see RESEARCH_NOTES.md."""
    import statsmodels.api as sm
    lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
    fi = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(),
                offset=lp).fit()
    fs = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
    return float(fi.params[0]), float(fs.params[1])


def paired_bootstrap_delta_auc(y, p_a, p_b, patient_ids,
                               iters=2000, seed=20260828):
    """AUC(a) - AUC(b) with a patient-level paired bootstrap CI and a
    two-sided p-value against 0. Complements DeLong (anti-conservative at
    small n with correlated models)."""
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(iters):
        r = _patient_resample(rng, patient_ids)
        try:
            deltas.append(roc_auc_score(y[r], p_a[r])
                          - roc_auc_score(y[r], p_b[r]))
        except ValueError:
            continue
    deltas = np.asarray(deltas)
    point = roc_auc_score(y, p_a) - roc_auc_score(y, p_b)
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    p_two = 2 * min((deltas <= 0).mean(), (deltas >= 0).mean())
    return {"delta": float(point), "ci95": [float(lo), float(hi)],
            "p_boot": float(min(1.0, p_two))}


def permutation_pvalue(y, p, patient_ids, iters=1000, seed=20260828):
    """Null: predictions carry no information. Permute labels at the PATIENT
    level (all rows of a patient flip together)."""
    rng = np.random.default_rng(seed)
    obs = roc_auc_score(y, p)
    uniq = np.unique(patient_ids)
    pid_label = {pid: y[patient_ids == pid].max() for pid in uniq}
    labels = np.array([pid_label[pid] for pid in uniq])
    count = 0
    for _ in range(iters):
        perm = rng.permutation(labels)
        y_perm = np.array([perm[np.searchsorted(uniq, pid)]
                           for pid in patient_ids])
        if len(np.unique(y_perm)) < 2:
            continue
        if roc_auc_score(y_perm, p) >= obs:
            count += 1
    return float((count + 1) / (iters + 1))


def summarize(y, p, patient_ids, eval_cfg, y_grade=None, pred_grade=None):
    """The standard block every arm reports."""
    auc, lo, hi = auc_ci(y, p, patient_ids, eval_cfg.bootstrap_iters,
                         eval_cfg.ci, seed=20260828)
    ci_int, ci_slope = calibration_intercept_slope(y, p)
    out = {"auc": auc, "auc_ci95": [lo, hi],
           "brier": float(brier_score_loss(y, p)),
           "cal_intercept": ci_int, "cal_slope": ci_slope,
           "n": int(len(y)), "n_pos": int(y.sum()),
           "n_patients": int(len(np.unique(patient_ids)))}
    if y_grade is not None and pred_grade is not None:
        k, klo, khi = qwk_ci(y_grade, pred_grade, patient_ids,
                             eval_cfg.bootstrap_iters, eval_cfg.ci)
        out.update({"qwk": k, "qwk_ci95": [klo, khi]})
    return out


if __name__ == "__main__":
    import sys, pathlib
    sys.path.insert(0, str(pathlib.Path(__file__).parent))
    from config import EvalConfig
    rng = np.random.default_rng(0)
    n_pat = 80
    pids = np.repeat([f"P{i}" for i in range(n_pat)], 2)
    y = np.repeat(rng.integers(0, 2, n_pat), 2)
    p_good = np.clip(y + rng.normal(0, 0.6, len(y)), 0, 1)
    p_rand = rng.random(len(y))

    cfg = EvalConfig(bootstrap_iters=500, permutation_iters=300)
    s = summarize(y.astype(float), p_good, pids, cfg)
    print(f"good model : AUC {s['auc']:.3f} [{s['auc_ci95'][0]:.3f},"
          f"{s['auc_ci95'][1]:.3f}]  slope {s['cal_slope']:.2f}")
    d = paired_bootstrap_delta_auc(y.astype(float), p_good, p_rand, pids, 500)
    print(f"good vs random: dAUC {d['delta']:+.3f} {d['ci95']}  p {d['p_boot']:.4f}")
    pv_good = permutation_pvalue(y.astype(float), p_good, pids, 300)
    pv_rand = permutation_pvalue(y.astype(float), p_rand, pids, 300)
    print(f"permutation p: good {pv_good:.4f} (expect small)  "
          f"random {pv_rand:.3f} (expect large)")
    assert pv_good < 0.05 < pv_rand, "permutation test sanity failed"
    print("metrics sanity: OK")
