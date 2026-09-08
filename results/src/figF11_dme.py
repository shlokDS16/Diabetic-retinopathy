"""
F11 - Macular-oedema (DME) head on IDRiD and DR-DME co-occurrence.
  a  DME confusion matrix on the fixed IDRiD test split (103 images).
  b  Any-DME ROC: 5-fold OOF on the 413 training images and the fixed test split, bootstrap bands.
  c  DR grade x DME grade co-occurrence on all 516 IDRiD images (row-normalised).
  d  DME score vs DR score on the test split (two independent heads, same images).
Data: results/out/fundus_dme_idrid.json, dme_idrid_pred.npz, fundus_idrid_pred.npz.
Run: python results/src/figF11_dme.py
"""
import json, pathlib, sys
import numpy as np, pandas as pd
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

OUT = pathlib.Path(__file__).resolve().parents[1] / "out"
D = json.loads((OUT / "fundus_dme_idrid.json").read_text(encoding="utf-8"))
Z = np.load(OUT / "dme_idrid_pred.npz")
F = np.load(OUT / "fundus_idrid_pred.npz", allow_pickle=True)          # our own file (object array of paths)
DME_LAB = ["0 none", "1 outside 1 DD", "2 within 1 DD"]

ps.apply()
fig, axs = ps.figure(cols=4, width="double", height_mm=52, gridspec_kw={"width_ratios": [1, 1.1, 1.05, 1]})

# a: test confusion
ax = axs[0]
t = D["test_fixed_split"]
ps.confusion(ax, t["confusion"], ["0", "1", "2"])
ax.set_xlabel("predicted DME grade"); ax.set_ylabel("true DME grade")
ax.set_title(f"test n = {t['n']}: QWK {t['qwk']:.2f} [{t['qwk_ci'][0]:.2f}, {t['qwk_ci'][1]:.2f}]", fontsize=6)
ps.label(ax, "a")

# b: any-DME ROC
ax = axs[1]
ps.plot_roc(ax, Z["y_train"] >= 1, Z["oof"], ps.C["dme"], f"5-fold OOF, n = {len(Z['oof'])}")
ps.plot_roc(ax, Z["y_test"] >= 1, Z["test_pred"], ps.C["external"], f"fixed test, n = {len(Z['test_pred'])}")
ps.refline(ax); ax.set_xlabel("1 − specificity"); ax.set_ylabel("sensitivity"); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.set_title("any DME (grade ≥ 1)", fontsize=6)
ps.finish_legend(ax, loc="lower right", fontsize=5)
ps.label(ax, "b")

# c: DR x DME co-occurrence on all IDRiD images
ax = axs[2]
ct = pd.crosstab(F["grade"], F["dme"]).reindex(index=range(5), columns=range(3), fill_value=0)
cmn = ct.values / np.maximum(ct.values.sum(1, keepdims=True), 1)
im = ax.imshow(cmn, cmap="Purples", vmin=0, vmax=1, aspect="auto")
for i in range(5):
    for j in range(3):
        ax.text(j, i, int(ct.values[i, j]), ha="center", va="center", fontsize=5.5, color="white" if cmn[i, j] > 0.55 else ps.INK)
ax.set_xticks(range(3)); ax.set_xticklabels(["0", "1", "2"]); ax.set_yticks(range(5)); ax.set_yticklabels([str(i) for i in range(5)])
ax.set_xlabel("DME grade"); ax.set_ylabel("DR grade (ICDR)"); ax.tick_params(length=0); ax.spines[["top", "right"]].set_visible(True)
any_dme = (F["dme"] >= 1)
share_ref = any_dme[F["grade"] >= 2].mean() * 100; share_nonref = any_dme[F["grade"] < 2].mean() * 100
ax.set_title(f"all IDRiD, n = {len(F['grade'])}\nany DME: {share_ref:.0f} % of referable, {share_nonref:.0f} % of non-referable", fontsize=5.2, loc="right")
ps.label(ax, "c", dx_mm=-9)

# d: DME score vs DR score on the test split (same image order: both loaders filter the same IDRiD table)
ax = axs[3]
paths = np.array([str(p) for p in F["path"]]); test = np.array(["Testing" in p for p in paths])
assert test.sum() == len(Z["y_test"]) and np.array_equal(F["dme"][test], Z["y_test"]), "test-split alignment failed"
dr_s, dme_s, y_dme = F["pred"][test], Z["test_pred"], Z["y_test"]
cols = [ps.C["ref"], ps.C["optical"], ps.C["dme"]]
for g in range(3):
    m = y_dme == g
    ax.scatter(dr_s[m], dme_s[m], s=7, color=cols[g], alpha=0.8, lw=0, label=DME_LAB[g])
for thv in D["thresholds"]:
    ax.axhline(thv, color=ps.RULE, lw=0.5, ls=(0, (3, 2)))
r = np.corrcoef(dr_s, dme_s)[0, 1]
ax.set_title(f"test split, n = {len(dr_s)}, Pearson r = {r:.2f}", fontsize=6)
ax.set_xlabel("DR head score (ordinal)"); ax.set_ylabel("DME head score (ordinal)")
ax.legend(loc="upper left", fontsize=5, title="true DME", title_fontsize=5, markerscale=1.2)
ps.label(ax, "d")

ps.save(fig, "figF11_dme")
