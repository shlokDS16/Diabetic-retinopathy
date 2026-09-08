"""
F9 - Fundus DR grading: patient-level cross-validation on DeepDRiD and external tests (IDRiD; APTOS 2019 when present).
  a  Out-of-fold confusion matrix (5-fold, patient-grouped) with per-grade recall.
  b  Referable-DR (grade >= 2) ROC with bootstrap bands: DeepDRiD OOF (patient-level), IDRiD, APTOS.
  c  Distribution of the ordinal score per true grade (DeepDRiD OOF) with the fitted thresholds.
  d  Validation QWK per epoch for each fold.
Data: results/out/fundus_deepdrid.json, fundus_oof.npz, fundus_idrid_pred.npz, fundus_aptos.json + fundus_aptos_pred.npz (optional).
Run: python results/src/figF09_fundus_dr.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parents[1] / "out"
D = json.loads((OUT / "fundus_deepdrid.json").read_text(encoding="utf-8"))
# allow_pickle: our own npz files (object arrays of image paths written by train_fundus.py / eval_aptos.py), not external input
Z = np.load(OUT / "fundus_oof.npz", allow_pickle=True); I = np.load(OUT / "fundus_idrid_pred.npz", allow_pickle=True)
A = np.load(OUT / "fundus_aptos_pred.npz", allow_pickle=True) if (OUT / "fundus_aptos_pred.npz").exists() else None
AJ = json.loads((OUT / "fundus_aptos.json").read_text(encoding="utf-8")) if (OUT / "fundus_aptos.json").exists() else None
th = np.array(D["thresholds"]); m = ~np.isnan(Z["oof"])
GR = ["0 none", "1 mild", "2 moderate", "3 severe", "4 PDR"]

ps.apply()
fig, axs = ps.figure(cols=2, rows=2, width="double", height_mm=112)
axs = axs.ravel()

# a: OOF confusion
ax = axs[0]
cm = np.array(D["cv"]["confusion"])
ps.confusion(ax, cm, ["0", "1", "2", "3", "4"])
rec = np.diag(cm) / np.maximum(cm.sum(1), 1)
for i, r in enumerate(rec):
    ax.text(4.62, i, f"{r*100:.0f} %", va="center", ha="left", fontsize=5, color=ps.MUTED)
ax.text(4.62, -0.75, "recall", ha="left", fontsize=5, color=ps.MUTED)
ax.set_xlabel("predicted ICDR grade"); ax.set_ylabel("true ICDR grade")
c = D["cv"]
ax.set_title(f"DeepDRiD OOF, {c['n']:,} images / {c['n_patients']} patients\nQWK {c['qwk']:.3f} [{c['qwk_ci'][0]:.3f}, {c['qwk_ci'][1]:.3f}] · accuracy {c['accuracy']*100:.1f} %", fontsize=6, loc="right")
ps.label(ax, "a", dx_mm=-9)

# b: referable ROC
ax = axs[1]
ps.plot_roc(ax, Z["grade"][m] >= 2, Z["oof"][m], ps.C["fundus"], f"DeepDRiD OOF, n = {int(m.sum())}", groups=Z["pid"][m])
ps.plot_roc(ax, I["grade"] >= 2, I["pred"], ps.C["external"], f"IDRiD external, n = {len(I['grade'])}")
if A is not None:
    ps.plot_roc(ax, A["grade"] >= 2, A["pred"], ps.C["external2"], f"APTOS 2019 external, n = {len(A['grade'])}", n_boot=500)
ps.refline(ax); ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity")
ax.set_title("referable DR (grade ≥ 2), fold-ensemble score", fontsize=6, loc="right")
ps.finish_legend(ax, loc="lower right", fontsize=5)
ps.label(ax, "b")

# c: score distribution per true grade
ax = axs[2]
data = [Z["oof"][m][Z["grade"][m] == g] for g in range(5)]
vp = ax.violinplot(data, positions=range(5), showextrema=False, widths=0.85)
for body in vp["bodies"]:
    body.set_facecolor(ps.C["fundus"]); body.set_alpha(0.45); body.set_edgecolor("none")
for g, d in enumerate(data):
    q = np.percentile(d, [25, 50, 75]); ax.plot([g, g], [q[0], q[2]], color=ps.INK, lw=0.9); ax.plot(g, q[1], "o", color="white", mec=ps.INK, ms=2.8)
    ax.text(g, -0.85, f"n = {len(d)}", ha="center", fontsize=5, color=ps.MUTED)
for t in th:
    ax.axhline(t, color=ps.RULE, lw=0.6, ls=(0, (3, 2)))
ax.set_xticks(range(5)); ax.set_xticklabels(GR, fontsize=5.5); ax.set_ylabel("ordinal score (OOF)"); ax.set_ylim(-1.1, 5)
ax.text(4.45, th[-1] + 0.08, "fitted thresholds", fontsize=5, color=ps.MUTED, ha="right", va="bottom")
ax.set_title("score distribution by true grade (median, IQR)", fontsize=6, loc="right")
ps.label(ax, "c")

# d: training curves
ax = axs[3]
for k, f in enumerate(D["folds"]):
    h = f["history"]; ax.plot([e["epoch"] + 1 for e in h], [e["val_qwk"] for e in h], "o-", ms=2, color=ps.OKABE_ITO[k % len(ps.OKABE_ITO)], label=f"fold {k} (n = {f['n_val']})", lw=0.8)
ax.set_xlabel("epoch"); ax.set_ylabel("validation QWK (fixed thresholds)")
ax.set_title(f"{D['model']}, {D['img']} px, {D['epochs']} epochs, one-cycle AdamW, flip TTA", fontsize=6, loc="right")
ps.finish_legend(ax, loc="lower right", fontsize=5, ncol=2)
ps.label(ax, "d")

ps.save(fig, "figF09_fundus_dr")
