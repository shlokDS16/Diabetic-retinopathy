"""
F7 - Periorbital bioimpedance: what the local tetrapolar set actually measures.
  a  Tissue contribution to the tetrapolar transfer impedance at 50 kHz (Geselowitz decomposition).
  b  |Z| vs frequency, tetrapolar transfer impedance and bipolar tissue impedance (contact impedance excluded).
  c  Phase vs frequency.
  d  Sensitivity to orbital / eyelid edema: relative |Z| change vs added water fraction, tetrapolar vs bipolar.
Data: sim/out/bioimpedance3d_coarse.json.
Run: python results/src/figF07_bioimpedance.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt

SIM = pathlib.Path(__file__).resolve().parents[2] / "sim" / "out"
B = json.loads((SIM / "bioimpedance3d_coarse.json").read_text(encoding="utf-8"))
TN = {"artery_angular": "angular artery", "artery_temporal": "temporal artery", "cornea": "cornea", "sclera": "sclera", "lens": "lens",
      "aqueous": "aqueous", "vitreous": "vitreous", "eyelid": "eyelid", "orbital_fat": "orbital fat", "skin": "skin",
      "fat_subcutaneous": "subcutaneous fat", "muscle": "muscle", "bone_cortical": "bone", "brain_grey": "brain"}
ORBIT = {"artery_angular", "cornea", "sclera", "lens", "aqueous", "vitreous", "eyelid", "orbital_fat"}

ps.apply()
fig, axs = ps.figure(cols=4, width="double", height_mm=55, gridspec_kw={"width_ratios": [1.3, 1, 1, 1]})

# a: tissue share at 50 kHz
ax = axs[0]
share = B["tissue_share_50kHz"]; keys = sorted(share, key=lambda k: share[k])
vals = np.array([share[k] for k in keys]) * 100
y = np.arange(len(keys))
ax.barh(y, vals, color=[ps.C["electrical"] if k in ORBIT else ps.C["ref"] for k in keys], height=0.7)
for yi, v in zip(y, vals):
    ax.text(max(v, 0.03) * 1.12, yi, f"{v:.2f} %" if v >= 0.01 else "<0.01 %", va="center", fontsize=5, color=ps.INK)
ax.set_xscale("log"); ax.set_xlim(0.01, 300)
ax.set_yticks(y); ax.set_yticklabels([TN[k] for k in keys], fontsize=5.5)
ax.set_xlabel("share of tetrapolar |Z| at 50 kHz (%)")
ax.text(0.98, 0.04, f"orbit + eye + lid: {B['orbit_share_50kHz']*100:.1f} %", transform=ax.transAxes, fontsize=5.5, ha="right", color=ps.C["electrical"])
ps.label(ax, "a", dx_mm=-15)

# b, c: spectra
f = np.array(sorted(float(k) for k in B["sweep"]))
tet = [B["sweep"][f"{int(v)}"]["tetrapolar"] for v in f]
bip = [B["sweep"][f"{int(v)}"]["bipolar"]["Z_tissue"] for v in f]
ax = axs[1]
ax.semilogx(f, [t["abs"] for t in tet], "o-", color=ps.C["electrical"], label="tetrapolar transfer |Z|")
ax.set_ylabel("|Z| (Ω)", color=ps.C["electrical"]); ax.tick_params(axis="y", colors=ps.C["electrical"])
ax2 = ax.twinx(); ax2.semilogx(f, [b["abs"] for b in bip], "s--", color=ps.C["ref"], label="bipolar tissue |Z|")
ax2.set_ylabel("bipolar tissue |Z| (Ω)", color=ps.C["ref"]); ax2.tick_params(axis="y", colors=ps.C["ref"]); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(ps.C["ref"])
ax.set_xlabel("frequency (Hz)")
h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
ax.legend(h1 + h2, l1 + l2, loc="lower left", fontsize=5)
rr = max(B["sweep"][k]["reciprocity_residual"] for k in B["sweep"]); gr = max(B["sweep"][k]["geselowitz_residual"] for k in B["sweep"])
ps.label(ax, "b")

ax = axs[2]
ax.semilogx(f, [t["phase_deg"] for t in tet], "o-", color=ps.C["electrical"], label="tetrapolar")
ax.semilogx(f, [b["phase_deg"] for b in bip], "s--", color=ps.C["ref"], label="bipolar tissue")
ax.set_xlabel("frequency (Hz)"); ax.set_ylabel("phase (°)")
ax.text(0.97, 0.97, f"reciprocity residual < {rr:.0e}\nGeselowitz residual < {gr:.0e}", transform=ax.transAxes, fontsize=5, va="top", ha="right", color=ps.MUTED)
ps.finish_legend(ax, loc="lower left", fontsize=5)
ps.label(ax, "c")

# d: edema sensitivity
ax = axs[3]
ed = B["edema_50kHz"]; fr = np.array(sorted(float(k) for k in ed))
ks = [k for k in ed]; ks = sorted(ks, key=float)
series = (("dZ_abs_percent", "o-", ps.C["electrical"], "tetrapolar"), ("bipolar_tissue_dZ_abs_percent", "s--", ps.C["ref"], "bipolar, tissue"),
          ("bipolar_total_dZ_abs_percent", "^:", ps.C["warn"], "bipolar, with contact Z"))
for key, fmt, col, lab in series:
    yy = [0, *[ed[k][key] for k in ks]]
    ax.plot([0, *fr * 100], yy, fmt, color=col)
    ax.text(fr.max() * 100 + 1.5, yy[-1], lab, fontsize=5, va="center", ha="left", color=col)
ax.set_xlim(-1, fr.max() * 100 + 26)
ax.axhline(0, color=ps.RULE, lw=0.5)
ax.set_xlabel("added orbital water (%)"); ax.set_ylabel("Δ|Z| at 50 kHz (%)")
tc = B["thermal_coupling_50kHz"]["dZ_abs_percent"]
ax.text(0.47, 0.62, f"σ(T) coupling: +{tc:.1f} % |Z|\nover the thermal field", transform=ax.transAxes, fontsize=5, va="center", ha="left", color=ps.MUTED)
ps.label(ax, "d")

ps.save(fig, "figF07_bioimpedance")
