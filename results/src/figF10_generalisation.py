"""
F10 - Cross-population generalisation of the fundus DR grader.
  a  QWK, referable AUC and accuracy per dataset with bootstrap CIs (DeepDRiD OOF, IDRiD, APTOS 2019).
  b  Sensitivity / specificity for referable DR at the DeepDRiD-fitted operating point, per dataset.
  c  Thresholds: DeepDRiD-fitted (used unchanged) vs re-fitted on each external set (reported, not used).
  d  Grade prevalence per dataset (label shift).
Data: results/out/fundus_deepdrid.json, fundus_oof.npz, fundus_idrid_pred.npz, fundus_aptos.json / fundus_aptos_pred.npz (optional).
Run: python results/src/figF10_generalisation.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import train_fundus as TF
import matplotlib.pyplot as plt

OUT = pathlib.Path(__file__).resolve().parents[1] / "out"
D = json.loads((OUT / "fundus_deepdrid.json").read_text(encoding="utf-8"))
# allow_pickle: our own npz files (object arrays of image paths), not external input
Z = np.load(OUT / "fundus_oof.npz", allow_pickle=True); I = np.load(OUT / "fundus_idrid_pred.npz", allow_pickle=True)
A = np.load(OUT / "fundus_aptos_pred.npz", allow_pickle=True) if (OUT / "fundus_aptos_pred.npz").exists() else None
AJ = json.loads((OUT / "fundus_aptos.json").read_text(encoding="utf-8")) if (OUT / "fundus_aptos.json").exists() else None
th = np.array(D["thresholds"]); m = ~np.isnan(Z["oof"])

sets = [("DeepDRiD\nOOF, China", Z["grade"][m], Z["oof"][m], Z["pid"][m], D["cv"], ps.C["fundus"]),
        ("IDRiD\nexternal, India", I["grade"], I["pred"], np.arange(len(I["grade"])), D["external_idrid"], ps.C["external"])]
if A is not None and AJ is not None:
    sets.append(("APTOS 2019\nexternal, India", A["grade"], A["pred"], np.arange(len(A["grade"])), AJ["external_aptos"], ps.C["external2"]))


def sens_spec(y, s, groups, n_boot=1000):
    yh = TF.apply_thresholds(s, th) >= 2; yr = y >= 2
    f = lambda a, b: (b[a].mean(), (~b[~a]).mean())
    se, sp = f(yr, yh)
    rng = np.random.default_rng(ps.SEED); g = np.asarray(groups); ug = np.unique(g); idx_by = {k: np.where(g == k)[0] for k in ug}
    bs = []
    for _ in range(n_boot):
        idx = np.concatenate([idx_by[k] for k in rng.choice(ug, len(ug), replace=True)]); bs.append(f(yr[idx], yh[idx]))
    bs = np.array(bs)
    return se, sp, np.percentile(bs[:, 0], [2.5, 97.5]), np.percentile(bs[:, 1], [2.5, 97.5])


ps.apply()
fig, axs = ps.figure(cols=4, width="double", height_mm=58, gridspec_kw={"width_ratios": [1.2, 1, 1, 1]})

# a: metrics forest
ax = axs[0]
metrics = [("qwk", "QWK (5-class)"), ("auc_referable", "AUC referable"), ("accuracy", "accuracy")]
for j, (name, y, s, g, R, col) in enumerate(sets):
    for i, (k, lab) in enumerate(metrics):
        yy = i * (len(sets) + 1) + j
        lo, hi = R[f"{k}_ci"]
        ax.plot([lo, hi], [yy, yy], color=col, lw=1.0); ax.plot(R[k], yy, "o", color=col, ms=3.2, mec="white", mew=0.4)
        ax.text(hi + 0.012, yy, f"{R[k]:.2f}", va="center", fontsize=5, color=col)
yt = [i * (len(sets) + 1) + (len(sets) - 1) / 2 for i in range(len(metrics))]
ax.set_yticks(yt); ax.set_yticklabels([lab for _, lab in metrics]); ax.invert_yaxis()
ax.set_xlim(0.35, 1.05); ax.set_xlabel("metric (95 % bootstrap CI)")
from matplotlib.lines import Line2D
ax.legend(handles=[Line2D([], [], color=col, marker="o", ms=3, lw=1, label=name.replace("\n", ", ")) for name, *_, col in sets], loc="center left", fontsize=5, handlelength=1.2)
ax.set_title("DeepDRiD thresholds applied unchanged", fontsize=6, loc="right")
ps.label(ax, "a", dx_mm=-11)

# b: sens/spec at the operating point
ax = axs[1]
x = np.arange(len(sets)); w = 0.36
for j, (name, y, s, g, R, col) in enumerate(sets):
    se, sp, sec, spc = sens_spec(y, s, g)
    ax.bar(j - w / 2, se, w, color=col, yerr=[[se - sec[0]], [sec[1] - se]], error_kw={"lw": 0.6}, label="sensitivity" if j == 0 else None)
    ax.bar(j + w / 2, sp, w, color=col, alpha=0.45, yerr=[[sp - spc[0]], [spc[1] - sp]], error_kw={"lw": 0.6}, hatch="////", label="specificity" if j == 0 else None)
    ax.text(j - w / 2, 0.03, f"{se*100:.0f}", ha="center", fontsize=5, color="white"); ax.text(j + w / 2, 0.03, f"{sp*100:.0f}", ha="center", fontsize=5, color=ps.INK)
ax.set_xticks(x); ax.set_xticklabels([n.split("\n")[0] for n, *_ in sets], fontsize=5.5); ax.set_ylim(0, 1.05); ax.set_ylabel("referable DR, at the OOF operating point")
ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, fontsize=5, handles=[plt.Rectangle((0, 0), 1, 1, color=ps.C["ref"]), plt.Rectangle((0, 0), 1, 1, color=ps.C["ref"], alpha=0.45, hatch="////")], labels=["sensitivity", "specificity"])
ps.label(ax, "b")

# c: thresholds
ax = axs[2]
ax.plot(range(4), th, "o-", color=ps.C["fundus"], label="fitted on DeepDRiD OOF (used)")
th_i = TF.fit_thresholds(I["pred"], I["grade"]); ax.plot(range(4), th_i, "s--", color=ps.C["external"], label="re-fitted on IDRiD (not used)")
if A is not None and AJ is not None:
    ax.plot(range(4), AJ["external_aptos_refit_thresholds"]["thresholds"], "^:", color=ps.C["external2"], label="re-fitted on APTOS (not used)")
ax.set_xticks(range(4)); ax.set_xticklabels(["0|1", "1|2", "2|3", "3|4"]); ax.set_xlabel("grade boundary"); ax.set_ylabel("threshold on the ordinal score")
ps.finish_legend(ax, loc="upper left", fontsize=5)
ps.label(ax, "c")

# d: prevalence
ax = axs[3]
bottom = np.zeros(len(sets)); shades = ["#d9d9d9", "#a6cee3", "#1f78b4", "#08519c", "#08306b"]
for gr in range(5):
    vals = [(y == gr).mean() * 100 for _, y, *_ in sets]
    ax.bar(x, vals, 0.6, bottom=bottom, color=shades[gr], label=f"grade {gr}", edgecolor="white", linewidth=0.3)
    for j, v in enumerate(vals):
        if v > 6:
            ax.text(j, bottom[j] + v / 2, f"{v:.0f}", ha="center", va="center", fontsize=5, color="white" if gr >= 2 else ps.INK)
    bottom += vals
ax.set_xticks(x); ax.set_xticklabels([n.split("\n")[0] for n, *_ in sets], fontsize=5.5); ax.set_ylabel("share of images (%)"); ax.set_ylim(0, 100)
ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=5, handlelength=1.0, borderaxespad=0)
ps.label(ax, "d")

ps.save(fig, "figF10_generalisation")
