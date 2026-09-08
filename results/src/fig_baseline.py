"""
Figure R1 — clinical-risk-factor baseline: discrimination, calibration, utility.

A  ROC with subject-level bootstrap 95% pointwise band (not simultaneous)
B  LOESS-smoothed calibration curve + 95% CI, intercept AND slope, and the
   predicted-probability histogram
C  Decision curve analysis with treat-all / treat-none references

Spec and sources: results/RESEARCH_NOTES.md. Later models add curves via the
figstyle.MODEL_COLORS registry — this script is the template for every
subsequent comparison figure.

Run:  python results/src/fig_baseline.py
"""
import pathlib, sys, json
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as fs                       # sets rcParams — import before pyplot
import matplotlib.pyplot as plt

import statsmodels.api as sm
from sklearn.metrics import roc_auc_score, roc_curve

OUT = HERE.parent / "out"
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MODEL = "clinical_lr"
COL = fs.MODEL_COLORS[MODEL]
LABEL = fs.MODEL_LABELS[MODEL]

d = np.load(OUT / "baseline_roc.npz")
y, p = d["y"].astype(int), d["p"]
meta = json.loads((OUT / "baseline_clinical.json").read_text(encoding="utf-8"))
rng = np.random.default_rng(fs.SEED)

# ---------------------------------------------------------------- panel A ---
# bootstrap pointwise ROC band on a common FPR grid
grid = np.linspace(0, 1, 200)
boot_tpr, boot_auc = [], []
idx = np.arange(len(y))
for _ in range(2000):
    b = rng.choice(idx, len(idx), replace=True)
    if len(np.unique(y[b])) < 2:
        continue
    fpr_b, tpr_b, _ = roc_curve(y[b], p[b])
    boot_tpr.append(np.interp(grid, fpr_b, tpr_b))
    boot_auc.append(roc_auc_score(y[b], p[b]))
boot_tpr = np.array(boot_tpr)
tpr_lo, tpr_hi = np.percentile(boot_tpr, [2.5, 97.5], axis=0)
auc = roc_auc_score(y, p)
auc_lo, auc_hi = np.percentile(boot_auc, [2.5, 97.5])

# ---------------------------------------------------------------- panel B ---
# LOESS calibration with bootstrap CI.
# Display only where predictions actually exist (2nd-98th percentile of p) --
# a LOESS drawn beyond the data is extrapolation dressed as calibration.
p_lo, p_hi = np.percentile(p, [2, 98])
pgrid = np.linspace(p_lo, p_hi, 120)
lo_main = sm.nonparametric.lowess(y, p, frac=0.6, it=0, xvals=pgrid)
boot_cal = []
for _ in range(500):
    b = rng.choice(idx, len(idx), replace=True)
    boot_cal.append(sm.nonparametric.lowess(y[b], p[b], frac=0.6, it=0, xvals=pgrid))
cal_lo, cal_hi = np.percentile(np.array(boot_cal), [2.5, 97.5], axis=0)

# intercept (calibration-in-the-large): logit(y) ~ 1 + offset(logit(p))
# slope: logit(y) ~ logit(p)
eps = 1e-6
lp = np.log(np.clip(p, eps, 1 - eps) / np.clip(1 - p, eps, 1 - eps))
fit_int = sm.GLM(y, np.ones_like(lp), family=sm.families.Binomial(),
                 offset=lp).fit()
fit_slope = sm.GLM(y, sm.add_constant(lp), family=sm.families.Binomial()).fit()
cal_intercept = float(fit_int.params[0])
cal_slope = float(fit_slope.params[1])
ci_int = fit_int.conf_int()[0]
ci_slope = fit_slope.conf_int()[1]

# ---------------------------------------------------------------- panel C ---
def net_benefit(y, p, thresholds):
    n = len(y)
    nb = []
    for t in thresholds:
        pred = p >= t
        tp = np.sum(pred & (y == 1))
        fp = np.sum(pred & (y == 0))
        nb.append(tp / n - fp / n * t / (1 - t))
    return np.array(nb)

thr = np.linspace(0.05, 0.80, 150)          # clinically relevant screening range
nb_model = net_benefit(y, p, thr)
prev = y.mean()
nb_all = prev - (1 - prev) * thr / (1 - thr)
# bootstrap CI on model net benefit
boot_nb = []
for _ in range(500):
    b = rng.choice(idx, len(idx), replace=True)
    boot_nb.append(net_benefit(y[b], p[b], thr))
nb_lo, nb_hi = np.percentile(np.array(boot_nb), [2.5, 97.5], axis=0)

# ================================================================== figure ==
fig = plt.figure(figsize=(7.16, 2.65))
gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.05, 1.0], wspace=0.5)

# A — ROC
ax = fig.add_subplot(gs[0, 0])
ax.plot([0, 1], [0, 1], color=fs.RULE, lw=0.8, ls="--", zorder=1)
ax.fill_between(grid, tpr_lo, tpr_hi, color=COL, alpha=0.18, lw=0, zorder=2)
fpr, tpr, _ = roc_curve(y, p)
ax.plot(fpr, tpr, color=COL, lw=1.6, zorder=3,
        label=f"{LABEL}\nAUC {auc:.3f} [{auc_lo:.3f}, {auc_hi:.3f}]")
ax.set_xlabel("false positive rate")
ax.set_ylabel("true positive rate")
ax.set_xlim(-0.02, 1.0); ax.set_ylim(0, 1.02)
ax.legend(fontsize=6.3, loc="lower right", handlelength=1.4)
fs.panel_title(ax, "A", "Discrimination")
ax.text(0.03, 0.965, "band: bootstrap 95% pointwise",
        transform=ax.transAxes, fontsize=5.6, color=fs.MUTED, va="top")

# B — calibration
ax = fig.add_subplot(gs[0, 1])
ax.plot([0, 1], [0, 1], color=fs.RULE, lw=0.8, ls="--")
ax.fill_between(pgrid, np.clip(cal_lo, 0, 1), np.clip(cal_hi, 0, 1),
                color=COL, alpha=0.18, lw=0)
ax.plot(pgrid, np.clip(lo_main, 0, 1), color=COL, lw=1.6)
ax.text(0.03, 0.97,
        f"intercept {cal_intercept:+.3f} [{ci_int[0]:+.3f}, {ci_int[1]:+.3f}]\n"
        f"slope      {cal_slope:.3f} [{ci_slope[0]:.3f}, {ci_slope[1]:.3f}]\n"
        f"Brier      {meta['brier']:.3f}",
        transform=ax.transAxes, fontsize=6.3, va="top", family="monospace")
# histogram of predictions along the floor
hist, edges = np.histogram(p, bins=28, range=(0, 1))
ax.bar((edges[:-1] + edges[1:]) / 2, hist / hist.max() * 0.10,
       width=edges[1] - edges[0], bottom=-0.005, color=COL, alpha=0.45, lw=0)
ax.set_xlabel("predicted probability")
ax.set_ylabel("observed proportion (LOESS)")
ax.set_xlim(0, 1); ax.set_ylim(-0.01, 1.02)
fs.panel_title(ax, "B", "Calibration")

# C — decision curve
ax = fig.add_subplot(gs[0, 2])
ax.fill_between(thr, nb_lo, nb_hi, color=COL, alpha=0.18, lw=0)
ax.plot(thr, nb_model, color=COL, lw=1.6, label=LABEL.split(" (")[0])
ax.plot(thr, nb_all, color=fs.MUTED, lw=1.0, ls="-.", label="treat all")
ax.axhline(0, color=fs.RULE, lw=0.9, label="treat none")
ax.set_xlabel("threshold probability")
ax.set_ylabel("net benefit")
ax.set_xlim(0.05, 0.80)
# clip the floor: strongly negative net benefit is uninformative and letting
# treat-all plunge off-scale squashes the region where decisions are made
ax.set_ylim(-0.05, max(nb_model.max(), prev) * 1.18)
ax.legend(fontsize=6.3, loc="upper right", handlelength=1.4)
fs.panel_title(ax, "C", "Clinical utility")

fs.save(fig, str(FIG / "figR1_baseline"))
plt.close(fig)

# machine-readable results, appended to the metrics record
metrics = {
    "model": MODEL,
    "auc": float(auc), "auc_ci95": [float(auc_lo), float(auc_hi)],
    "calibration_intercept": cal_intercept,
    "calibration_intercept_ci95": [float(ci_int[0]), float(ci_int[1])],
    "calibration_slope": cal_slope,
    "calibration_slope_ci95": [float(ci_slope[0]), float(ci_slope[1])],
    "brier": meta["brier"],
    "dca_thresholds": [0.05, 0.80],
    "bootstrap": {"roc_iters": 2000, "cal_iters": 500, "nb_iters": 500,
                  "resampling": "subject-level", "seed": fs.SEED},
}
(OUT / "figR1_metrics.json").write_text(json.dumps(metrics, indent=2),
                                        encoding="utf-8")

print(f"AUC                 {auc:.4f} [{auc_lo:.4f}, {auc_hi:.4f}]")
print(f"calibration int     {cal_intercept:+.4f} [{ci_int[0]:+.4f}, {ci_int[1]:+.4f}]  (target 0)")
print(f"calibration slope   {cal_slope:.4f} [{ci_slope[0]:.4f}, {ci_slope[1]:.4f}]  (target 1)")
print(f"Brier               {meta['brier']:.4f}")
rng_thr = thr[(nb_model > np.maximum(nb_all, 0))]
if len(rng_thr):
    print(f"model beats treat-all AND treat-none for thresholds "
          f"{rng_thr.min():.2f}-{rng_thr.max():.2f}")
print(f"\nwrote {FIG/'figR1_baseline'}.png/.pdf and {OUT/'figR1_metrics.json'}")
