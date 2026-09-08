"""
F13 - Physiological signals on open cohorts: heart-rate variability (ECG) and fingertip PPG.
  a  HRV -> diabetes vs control, leave-one-out logistic ROC with bootstrap band (two PhysioNet cohorts pooled).
  b  HRV -> retinopathy within diabetics: Cohen d per feature with BH-corrected p (the honest null).
  c  HRV features vs HbA1c: Spearman rho.
  d  PPG-BP -> diabetes: AUC by feature route vs the demographics-only baseline; hypertension as positive control.
  e  PaPaGei foundation-model embedding: age regression parity (positive control on 2.1-s tiled segments).
Data: results/out/hrv_open.json, hrv_pred.npz, ppg_open.json, ppg_pred.npz.
Run: python results/src/figF13_physio.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parents[1] / "out"
H = json.loads((OUT / "hrv_open.json").read_text(encoding="utf-8")); HP = np.load(OUT / "hrv_pred.npz")
P = json.loads((OUT / "ppg_open.json").read_text(encoding="utf-8")); PP = np.load(OUT / "ppg_pred.npz")
FEAT = {"hr_mean": "mean HR", "sdnn": "SDNN", "rmssd": "RMSSD", "pnn50": "pNN50", "cvnn": "CVNN", "lf_hf": "LF/HF"}

ps.apply()
fig = plt.figure(figsize=(ps.DOUBLE_MM * ps.MM, 100 * ps.MM), layout="constrained")
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1])
axa, axb, axc = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[0, 2])
axd, axe = fig.add_subplot(gs[1, :2]), fig.add_subplot(gs[1, 2])

# a: HRV diabetes ROC
ax = axa
y, p = HP["dm_vs_control_y"], HP["dm_vs_control_p"]
ps.plot_roc(ax, y, p, ps.C["hrv"], "LOO logistic")
ps.refline(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity")
ax.set_title(f"HRV → diabetes vs control\nn = {len(y)} ({int(y.sum())} diabetic), two PhysioNet cohorts", fontsize=6, loc="right")
ps.finish_legend(ax, loc="lower right", fontsize=5.5)
ps.label(ax, "a")

# b: Cohen d within diabetics
ax = axb
eff = H["dr_within_dm"]["effects"]; names = list(eff); d = np.array([eff[k]["d"] for k in names]); pbh = np.array([eff[k].get("p_bh", 1.0) for k in names])
yy = np.arange(len(names))[::-1]
ax.axvline(0, color=ps.RULE, lw=0.6)
for yi, di, pi in zip(yy, d, pbh):
    ax.plot([0, di], [yi, yi], color=ps.C["hrv"], lw=0.8)
    ax.plot(di, yi, "o", color=ps.C["warn"] if pi < 0.05 else ps.C["hrv"], ms=3.5)
    ax.text(di + (0.12 if di >= 0 else -0.12), yi, f"$p_{{BH}}$ = {pi:.2f}", fontsize=5, va="center", ha="left" if di >= 0 else "right", color=ps.MUTED)
ax.set_yticks(yy); ax.set_yticklabels([FEAT[k] for k in names], fontsize=5.5)
lim = max(1.0, np.abs(d).max() * 2.2); ax.set_xlim(-lim, lim)
ax.set_xlabel("Cohen d, DR+ vs DR− (diabetics)")
a = H["dr_within_dm"]["auc"]
ax.set_title(f"HRV → DR within diabetics\nn = {H['dr_within_dm']['n']} ({H['dr_within_dm']['n_dr']} DR+), LOO AUC {a['auc']:.2f} [{a['auc_ci'][0]:.2f}, {a['auc_ci'][1]:.2f}]", fontsize=6, loc="right")
ps.label(ax, "b", dx_mm=-9)

# c: Spearman with HbA1c
ax = axc
sp = H["hba1c_spearman"]; names = list(sp); rho = np.array([sp[k]["rho"] for k in names]); pv = np.array([sp[k]["p"] for k in names])
yy = np.arange(len(names))[::-1]
ax.barh(yy, rho, color=[ps.C["warn"] if q < 0.05 else ps.C["hrv"] for q in pv], height=0.6)
for yi, r, q in zip(yy, rho, pv):
    ax.text(r + (0.03 if r >= 0 else -0.03), yi, f"p = {q:.2f}" if q >= 0.01 else f"p = {q:.3f}", fontsize=5, va="center", ha="left" if r >= 0 else "right", color=ps.MUTED)
ax.axvline(0, color=ps.RULE, lw=0.6)
ax.set_yticks(yy); ax.set_yticklabels([FEAT[k] for k in names], fontsize=5.5)
ax.set_xlim(-0.8, 0.8); ax.set_xlabel("Spearman ρ with HbA1c")
ax.set_title(f"HRV vs glycaemia\nn = {H['ge75_n']} diabetics with HbA1c", fontsize=6, loc="right")
ps.label(ax, "c", dx_mm=-9)

# d: PPG AUC by route
ax = axd
keys = ["demographics_only", "morphology", "morphology_plus_demo", "papagei_pca16", "papagei_plus_demo"]
labels = ["demographics only\n(age, sex, BMI)", "pulse morphology\n(11 features)", "morphology\n+ demographics", "PaPaGei-S embedding\n(PCA-16)", "PaPaGei\n+ demographics"]
v = np.array([P["diabetes"][k]["auc"] for k in keys]); lo = v - np.array([P["diabetes"][k]["auc_ci"][0] for k in keys]); hi = np.array([P["diabetes"][k]["auc_ci"][1] for k in keys]) - v
x = np.arange(len(keys))
ax.bar(x, v, yerr=[lo, hi], color=ps.C["ppg"], width=0.6, error_kw={"lw": 0.6})
for xi, vi, h_ in zip(x, v, hi):
    ax.text(xi, vi + h_ + 0.015, f"{vi:.2f}", ha="center", va="bottom", fontsize=5.5, color=ps.INK)
hv = P["hypertension"]["morphology"]
ax.bar([len(keys)], [hv["auc"]], yerr=[[hv["auc"] - hv["auc_ci"][0]], [hv["auc_ci"][1] - hv["auc"]]], color=ps.C["good"], width=0.6, error_kw={"lw": 0.6})
ax.text(len(keys), hv["auc_ci"][1] + 0.015, f"{hv['auc']:.2f}", ha="center", va="bottom", fontsize=5.5, color=ps.INK)
ax.axhline(0.5, color=ps.RULE, lw=0.6, ls=(0, (3, 2)))
ax.set_xticks(list(x) + [len(keys)]); ax.set_xticklabels(labels + ["hypertension\n(positive control)"], fontsize=5.5)
ax.set_ylim(0.2, 0.95); ax.set_ylabel("AUC (5-fold CV, subject level)")
ax.set_title(f"PPG-BP → diabetes, n = {P['n_subjects']} subjects ({P['n_diabetic']} type-2 diabetic), 3 × 2.1-s fingertip PPG each", fontsize=6, loc="right")
ps.label(ax, "d")

# e: PaPaGei age parity
ax = axe
age, ap = PP["age"], PP["age_pred"]
ax.scatter(age, ap, s=6, color=ps.C["ppg"], alpha=0.6, lw=0)
lo_, hi_ = min(age.min(), ap.min()) - 3, max(age.max(), ap.max()) + 3
ax.plot([lo_, hi_], [lo_, hi_], color=ps.RULE, lw=0.6, ls=(0, (3, 2)))
ax.set_xlim(lo_, hi_); ax.set_ylim(lo_, hi_)
ax.set_xlabel("age (years)"); ax.set_ylabel("PaPaGei-predicted age (years)")
ag = P["age_from_papagei"]
ax.set_title(f"embedding positive control\nage r = {ag['r']:.2f}, MAE {ag['mae_years']:.1f} y (tiled 2.1-s segments)", fontsize=6, loc="right")
ps.label(ax, "e")

ps.save(fig, "figF13_physio")
