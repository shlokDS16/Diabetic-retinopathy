"""
F12 - Systemic (EHR) models of diabetic retinopathy and diabetic neuropathy, n = 77,724 (Istanbul export).
  a  ROC, retinopathy: logistic vs HistGradientBoosting (5-fold OOF, bootstrap bands).
  b  ROC, diabetic neuropathy.
  c  Calibration (quantile bins) of the boosted models, with slope / intercept.
  d  Logistic odds ratios per SD, top features for each outcome (forest).
  e  Permutation importance of the boosted model (held-out 20 %).
  f  Joint outcome: one-vs-rest AUC of the 4-class model and the DR-DN 2x2 with the Fisher odds ratio.
Data: results/out/tabular_ehr.json, results/out/ehr_oof.npz.
Run: python results/src/figF12_ehr.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve

OUT = pathlib.Path(__file__).resolve().parents[1] / "out"
E = json.loads((OUT / "tabular_ehr.json").read_text(encoding="utf-8"))
Z = np.load(OUT / "ehr_oof.npz")
FN = {"HbA1c": "HbA1c", "Hba1c_change": "ΔHbA1c over 1 y", "hypertension_i10": "hypertension (I10)", "lipoprotein_met_dis": "lipoprotein disorder",
      "kidney_failure": "kidney failure", "nephropaties": "nephropathy", "ceserian_multiple": "caesarean / multiple birth", "vitamin_deficiency": "vitamin deficiency",
      "insulin_aspart": "insulin aspart", "insulin_glarjin": "insulin glargine", "insulin_lispro": "insulin lispro", "insulin_glusilin": "insulin glulisine",
      "insulin_detemir": "insulin detemir", "insulin_nph": "insulin NPH", "insulin_reguler": "regular insulin", "pioglitazon_hcl": "pioglitazone",
      "metformin_hcl": "metformin", "gliklazid": "gliclazide", "glimepirid": "glimepiride", "pshycoanaleptics": "psychoanaleptics",
      "other_digestive": "other alimentary (A16)", "antiepileptics": "antiepileptics (N03)", "Creatinine": "creatinine", "Triglyceride": "triglycerides",
      "Cholesterol": "total cholesterol", "ischemic_heart_dis": "ischaemic heart disease", "cerebrovascular": "cerebrovascular disease"}
pretty = lambda f: FN.get(f, f.replace("_", " "))

ps.apply()
fig, axs = ps.figure(cols=3, rows=2, width="double", height_mm=112)
axs = axs.ravel()

# a, b: ROC
for ax, tgt, lab, letter in ((axs[0], "retinopathy", "retinopathy", "a"), (axs[1], "diabetic_neuropathy", "diabetic neuropathy", "b")):
    y = Z["y_dr"] if tgt == "retinopathy" else Z["y_dn"]
    r = E["targets"][tgt]
    from sklearn.metrics import roc_curve
    for mdl, col, name in (("logistic", ps.C["ref"], "logistic (L2)"), ("hgb", ps.C["ehr"], "gradient boosting")):
        f_, t_, _ = roc_curve(y, Z[f"{tgt}_{mdl}"]); ci = r[mdl]["auc_ci"]
        ax.plot(f_, t_, color=col, label=f"{name.split(' ')[0]}: {r[mdl]['auc']:.3f} [{ci[0]:.3f}–{ci[1]:.3f}]")
    ps.refline(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity")
    ax.set_title(f"{lab}\n{r['prevalence']*100:.1f} % prevalence, n = {r['n']:,}", fontsize=6, loc="right")
    ps.finish_legend(ax, loc="lower right", fontsize=5, handlelength=1.0, title="AUC, 5-fold OOF", title_fontsize=5)
    ps.label(ax, letter)

# c: calibration
ax = axs[2]
for tgt, col, lab in (("retinopathy", ps.C["external"], "retinopathy"), ("diabetic_neuropathy", ps.C["ehr"], "diabetic neuropathy")):
    y = Z["y_dr"] if tgt == "retinopathy" else Z["y_dn"]; p = Z[f"{tgt}_hgb"]
    pt, pp = calibration_curve(y, p, n_bins=10, strategy="quantile")
    c = E["targets"][tgt]["hgb"]["calibration"]
    ax.plot(pp, pt, "o-", color=col, ms=2.5, label=f"{lab}: slope {c['slope']:.2f}, intercept {c['intercept']:+.2f}, Brier {c['brier']:.3f}")
ps.refline(ax); ax.set_xlim(0, 0.8); ax.set_ylim(0, 0.8)
ax.set_xlabel("predicted probability (boosted, OOF)"); ax.set_ylabel("observed fraction")
ps.finish_legend(ax, loc="upper left", fontsize=5)
ps.label(ax, "c")

# d: odds ratios per SD (forest), union of top features
ax = axs[3]
ors = {t: E["targets"][t]["logistic_odds_ratio_per_sd"] for t in ("retinopathy", "diabetic_neuropathy")}
feats = []
for t in ors:
    for f in list(ors[t])[:8]:
        if f not in feats:
            feats.append(f)
feats = sorted(feats, key=lambda f: -max(abs(np.log(ors[t].get(f, 1))) for t in ors))[:12]
y = np.arange(len(feats))[::-1]
for k, (t, col, mk, dy) in enumerate((("retinopathy", ps.C["external"], "o", 0.18), ("diabetic_neuropathy", ps.C["ehr"], "s", -0.18))):
    vals = [ors[t].get(f, np.nan) for f in feats]
    ax.plot(vals, y + dy, mk, color=col, ms=3, label="retinopathy" if t == "retinopathy" else "diabetic neuropathy")
ax.axvline(1, color=ps.RULE, lw=0.6); ax.set_xscale("log")
from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator
tk = [0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6]
ax.xaxis.set_major_locator(FixedLocator(tk)); ax.xaxis.set_major_formatter(FixedFormatter([f"{v:g}" for v in tk])); ax.xaxis.set_minor_locator(NullLocator())
ax.set_yticks(y); ax.set_yticklabels([pretty(f) for f in feats], fontsize=5.5)
ax.set_xlabel("odds ratio per SD (logistic, all data)")
ps.finish_legend(ax, loc="upper left", fontsize=5, handletextpad=0.2)
ps.label(ax, "d", dx_mm=-14)

# e: permutation importance
ax = axs[4]
pi = {t: E["targets"][t]["hgb_permutation_importance"] for t in ("retinopathy", "diabetic_neuropathy")}
feats = []
for t in pi:
    for f in list(pi[t])[:7]:
        if f not in feats:
            feats.append(f)
feats = sorted(feats, key=lambda f: -max(pi[t].get(f, 0) for t in pi))[:10]
y = np.arange(len(feats))[::-1]
ax.barh(y + 0.18, [pi["retinopathy"].get(f, 0) for f in feats], height=0.34, color=ps.C["external"], label="retinopathy")
ax.barh(y - 0.18, [pi["diabetic_neuropathy"].get(f, 0) for f in feats], height=0.34, color=ps.C["ehr"], label="diabetic neuropathy")
ax.set_yticks(y); ax.set_yticklabels([pretty(f) for f in feats], fontsize=5.5)
ax.set_xlabel("permutation importance (ΔAUC, held-out 20 %)")
ps.finish_legend(ax, loc="center right", fontsize=5)
ps.label(ax, "e", dx_mm=-14)

# f: joint outcome
ax = axs[5]
jo = E["joint_ovr_auc"]; names = ["none", "DR only", "DN only", "both"]
cnt = E["joint_counts"]; n_by = [cnt.get(str(k), 0) for k in range(4)]
x = np.arange(4)
bars = ax.bar(x, [jo[n] for n in names], color=[ps.C["ref"], ps.C["external"], ps.C["ehr"], ps.C["good"]], width=0.62)
for xi, n in zip(x, names):
    ax.text(xi, jo[n] + 0.01, f"{jo[n]:.2f}", ha="center", va="bottom", fontsize=5.5, color=ps.INK)
ax.set_xticks(x); ax.set_xticklabels([f"{n}\nn = {c:,}" for n, c in zip(names, n_by)], fontsize=5)
ax.set_ylim(0.5, 1.0); ax.set_ylabel("one-vs-rest AUC (4-class boosted)")
a = E["dr_dn_association"]
ax.text(0.03, 0.97, f"DR–DN co-occurrence:\nOR {a['odds_ratio']:.2f}, Fisher p = {a['p']:.0e}", transform=ax.transAxes, fontsize=5.5, va="top", color=ps.INK)
ps.label(ax, "f")

ps.save(fig, "figF12_ehr")
