"""
F5 - Uncertainty quantification of the thermal readouts (Saltelli / Sobol, 8 parameters, 640 solves).
  a  Total-order Sobol indices S_T (cell text: first-order S_1) for the four readouts.
  b  Distribution of each readout over the 640 samples (violin, 5-95 % band).
  c  Ambient temperature vs corneal-apex temperature, coloured by choroidal perfusion fraction:
     ambient dominates, choroid modulates within a band.
Data: sim/out/surrogate_thermal.json, sim/out/uq_thermal_samples.npz.
Run: python results/src/figF05_uq.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt
from cmcrameri import cm as ccm

SIM = pathlib.Path(__file__).resolve().parents[2] / "sim" / "out"
S = json.loads((SIM / "surrogate_thermal.json").read_text(encoding="utf-8"))
Z = np.load(SIM / "uq_thermal_samples.npz"); X, Y = Z["X"], Z["Y"]

PN = {"choroid_frac": "choroidal perfusion", "skin_perf_frac": "skin perfusion", "eyelid_perf_frac": "eyelid perfusion",
      "orbit_perf_frac": "orbital perfusion", "T_amb": "ambient temperature", "h_skin": "skin convection $h$",
      "E_tear": "tear evaporation", "k_sclera_scale": "scleral conductivity"}
ON = {"apex_C": "corneal apex", "canthus_C": "canthus skin", "dci_limbus_C": "thermopile → limbus", "dci_canthus_C": "thermopile → canthus"}
params = [p[0] for p in S["params"]]; outs = S["outputs"]
ST = np.clip(np.array(S["ST"]), 0, None); S1 = np.clip(np.array(S["S1"]), 0, None)

ps.apply()
fig, axs = ps.figure(cols=3, width="double", height_mm=62, gridspec_kw={"width_ratios": [1.25, 1, 1]})

# a: Sobol heat-grid
ax = axs[0]
im = ax.imshow(ST, cmap=ccm.lajolla_r if hasattr(ccm, "lajolla_r") else "YlOrRd", vmin=0, vmax=1, aspect="auto")
for i in range(ST.shape[0]):
    for j in range(ST.shape[1]):
        v = ST[i, j]
        ax.text(j, i, f"{v:.2f}\n({S1[i, j]:.2f})" if v >= 0.01 else "<0.01", ha="center", va="center", fontsize=5,
                color="white" if v > 0.55 else ps.INK)
ax.set_xticks(range(len(outs))); ax.set_xticklabels([ON[o] for o in outs], rotation=25, ha="right")
ax.set_yticks(range(len(params))); ax.set_yticklabels([f"{PN[p]}\n[{lo:g}–{hi:g}]" for p, lo, hi in S["params"]], fontsize=5.5)
ax.tick_params(length=0); ax.spines[["top", "right"]].set_visible(True)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("total-order Sobol $S_T$ (first-order $S_1$ in brackets)", fontsize=5.5); cb.ax.tick_params(labelsize=5)
ps.label(ax, "a", dx_mm=-14)

# b: output distributions
ax = axs[1]
cols = [ps.C["thermal"], ps.C["optical"], ps.C["electrical"], ps.C["pupil"]]
vp = ax.violinplot([Y[:, j] for j in range(Y.shape[1])], positions=range(len(outs)), showextrema=False, widths=0.8)
for body, c in zip(vp["bodies"], cols):
    body.set_facecolor(c); body.set_alpha(0.55); body.set_edgecolor("none")
for j, o in enumerate(outs):
    st = S["output_stats"][o]
    ax.plot([j, j], [st["p05"], st["p95"]], color=ps.INK, lw=0.8)
    ax.plot(j, st["mean"], "o", color="white", mec=ps.INK, ms=3)
SHORT = {"apex_C": "corneal\napex", "canthus_C": "canthus\nskin", "dci_limbus_C": "thermopile\n→ limbus", "dci_canthus_C": "thermopile\n→ canthus"}
ax.set_xticks(range(len(outs))); ax.set_xticklabels([f"{SHORT[o]}\n{S['output_stats'][o]['mean']:.1f} ± {S['output_stats'][o]['sd']:.1f} °C" for o in outs], fontsize=5.2)
ax.set_ylabel("temperature (°C)"); ax.set_xlim(-0.6, len(outs) - 0.2)
ax.text(0.02, 0.97, f"{S['n_solves']} Saltelli solves\nmean ± SD, bar = 5–95 %", transform=ax.transAxes, fontsize=5.5, va="top", color=ps.MUTED)
ps.label(ax, "b")

# c: ambient vs apex coloured by choroid fraction
ax = axs[2]
i_amb, i_ch, j_apex = params.index("T_amb"), params.index("choroid_frac"), outs.index("apex_C")
# X is stored in the unit hypercube [0, 1]; convert to physical units with the parameter ranges (fixed 2026-09-08)
lo_a, hi_a = S["params"][i_amb][1], S["params"][i_amb][2]; lo_c, hi_c = S["params"][i_ch][1], S["params"][i_ch][2]
x_amb = lo_a + X[:, i_amb] * (hi_a - lo_a); x_ch = lo_c + X[:, i_ch] * (hi_c - lo_c)
sc = ax.scatter(x_amb, Y[:, j_apex], c=x_ch, cmap=ccm.batlow, s=5, lw=0, alpha=0.85)
a, b = np.polyfit(x_amb, Y[:, j_apex], 1)
xx = np.linspace(x_amb.min(), x_amb.max(), 2)
ax.plot(xx, a * xx + b, color=ps.INK, lw=0.8, ls=(0, (3, 2)))
r2 = np.corrcoef(x_amb, Y[:, j_apex])[0, 1] ** 2
ax.text(0.03, 0.97, f"slope {a:.2f} °C per °C ambient\n$R^2$ = {r2:.2f} (ambient alone)", transform=ax.transAxes, fontsize=5.5, va="top", color=ps.INK)
ax.set_xlabel("ambient temperature (°C)"); ax.set_ylabel("corneal-apex temperature (°C)")
cb = fig.colorbar(sc, ax=ax, fraction=0.05, pad=0.02); cb.set_label("choroidal perfusion fraction", fontsize=5.5); cb.ax.tick_params(labelsize=5)
ps.label(ax, "c")

ps.save(fig, "figF05_uq")
