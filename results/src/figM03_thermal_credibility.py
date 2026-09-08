"""
Main Fig. 3 (npj manuscript): the thermal forward model, its verification and its uncertainty, in one figure.
Composite of panels from figF03 (a, b, c), figF04 (d, e) and figF05 (f). Same data files, same pubstyle.
  a  Sagittal temperature slice through the left globe with tissue boundaries.
  b  Change in each readout as choroidal perfusion falls (thermopile 90 deg vs 5 deg vs true apex).
  c  Simulated apex, canthus and temple temperatures against published ranges.
  d  Method of manufactured solutions: error vs element size, observed order.
  e  Grid convergence index per quantity of interest.
  f  Total-order Sobol indices, eight parameters x four readouts.
Data: results/out/thermal_slices.npz, sim/out/thermal3d_full.json, verification.json, surrogate_thermal.json.
Run: python results/src/figM03_thermal_credibility.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.collections import LineCollection
from cmcrameri import cm as ccm

ROOT = pathlib.Path(__file__).resolve().parents[1]; SIM = ROOT.parent / "sim" / "out"
S = np.load(ROOT / "out" / "thermal_slices.npz")
T3 = json.loads((SIM / "thermal3d_full.json").read_text(encoding="utf-8"))
ver = json.loads((SIM / "verification.json").read_text(encoding="utf-8"))
UQ = json.loads((SIM / "surrogate_thermal.json").read_text(encoding="utf-8"))
TMIN, TMAX = 31.0, 37.4


def region_edges(pts, tris, region):
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]); e.sort(axis=1)
    reg = np.tile(region, 3)
    order = np.lexsort((e[:, 1], e[:, 0])); e, reg = e[order], reg[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    segs, i = [], 0
    while i < len(e):
        if i + 1 < len(e) and same[i]:
            if reg[i] != reg[i + 1]:
                segs.append(pts[e[i]])
            i += 2
        else:
            segs.append(pts[e[i]]); i += 1
    return np.array(segs)


ps.apply()
fig = plt.figure(figsize=(ps.DOUBLE_MM * ps.MM, 120 * ps.MM), layout="constrained")
gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 1.15], height_ratios=[1.0, 1.0])
axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

# a: sagittal slice
ax = axs[0]
pts, tris, T, reg = S["sagittal_eye_pts"], S["sagittal_eye_tris"], S["sagittal_eye_T"], S["sagittal_eye_region"]
tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
im = ax.tripcolor(tri, T, shading="gouraud", cmap=ccm.lipari, vmin=TMIN, vmax=TMAX)
ax.add_collection(LineCollection(region_edges(pts, tris, reg), colors="white", linewidths=0.25, alpha=0.7))
ax.set_xlim(-42, 16); ax.set_ylim(-30, 30); ax.set_aspect("equal", adjustable="box")
ax.set_xlabel("anterior (mm)"); ax.set_ylabel("up (mm)"); ax.spines[["top", "right"]].set_visible(True)
ax.set_title("sagittal plane, left globe", fontsize=6, loc="right")
cb = fig.colorbar(im, ax=ax, fraction=0.06, pad=0.03, location="bottom", shrink=0.9); cb.set_label("tissue temperature (°C), ambient 25 °C", fontsize=5.5); cb.ax.tick_params(labelsize=5)
ps.label(ax, "a")

# b: readouts vs choroidal perfusion
ax = axs[1]
sw = sorted(T3["choroid_sweep"], key=lambda r: r["choroid_frac"]); cf = np.array([r["choroid_frac"] for r in sw]) * 100
base = {k: [r[k] for r in sw if r["choroid_frac"] == 1.0][0] for k in ("apex_C", "dci_C", "baa_C", "canthus_C")}
for k, lab, col, mk in (("apex_C", "corneal apex (true)", ps.C["thermal"], "o"), ("dci_C", f"thermopile 5° FOV ({T3['thermopile_DCI_5deg']['spot_mm']:.1f} mm spot)", ps.C["electrical"], "s"),
                        ("baa_C", f"thermopile 90° FOV ({T3['thermopile_BAA_90deg']['spot_mm']:.0f} mm spot)", ps.C["optical"], "^"), ("canthus_C", "canthus skin", ps.C["pupil"], "D")):
    ax.plot(cf, [r[k] - base[k] for r in sw], marker=mk, color=col, label=lab, ms=2.8)
ax.axhline(0, color=ps.RULE, lw=0.5); ax.invert_xaxis()
ax.set_xlabel("choroidal perfusion (% of nominal)"); ax.set_ylabel("change in reading (°C)")
ps.finish_legend(ax, loc="lower left", fontsize=5, handlelength=1.2)
ps.label(ax, "b")

# c: simulated vs published
ax = axs[2]
v = T3["validation"]
items = [("corneal apex", T3["corneal_apex_C"], v["corneal_apex_published_C"], ps.C["thermal"]),
         ("inner canthus skin", T3["medial_canthus_skin_C"], v["inner_canthus_published_C"], ps.C["pupil"]),
         ("temple skin", T3["temple_skin_C"], None, ps.C["ref"])]
y = np.arange(len(items))[::-1]
for yi, (lab, sim, pub, col) in zip(y, items):
    if pub:
        ax.plot(pub, [yi, yi], color=col, lw=4, alpha=0.3, solid_capstyle="butt")
        ax.text(pub[1] + 0.15, yi, "published\n" + f"{pub[0]:.1f} to {pub[1]:.1f}", fontsize=5, va="center", color=ps.MUTED)
    ax.plot(sim, yi, "o", color=col, ms=4, mec="white", mew=0.4)
    ax.text(sim, yi + 0.28, f"{sim:.2f}", fontsize=5, ha="center", color=col)
sp = sorted(T3["skin_perfusion_sweep"], key=lambda r: r["skin_perf_frac"])
for r in sp:
    if r["skin_perf_frac"] < 1.0:
        ax.plot(r["canthus_C"], y[1], "|", color=ps.C["pupil"], ms=5, mew=0.8); ax.plot(r["temple_C"], y[2], "|", color=ps.C["ref"], ms=5, mew=0.8)
ax.text(sp[0]["canthus_C"] - 0.1, y[1] - 0.32, f"ticks: skin perfusion {int(sp[0]['skin_perf_frac']*100)} % and {int(sp[1]['skin_perf_frac']*100)} %", fontsize=5, ha="left", va="top", color=ps.MUTED)
ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=5.5)
ax.set_xlabel("temperature (°C), ambient 25 °C"); ax.set_xlim(32.0, 38.6); ax.set_ylim(-0.8, len(items) - 0.3)
ps.label(ax, "c", dx_mm=-12)

# d: MMS
ax = axs[3]
h = np.array([ver["mms_coarse"]["h_mm"], ver["mms_full"]["h_mm"]])
l2 = np.array([ver["mms_coarse"]["L2_rel"], ver["mms_full"]["L2_rel"]]); li = np.array([ver["mms_coarse"]["Linf_K"], ver["mms_full"]["Linf_K"]])
p = ver["observed_order"]
ax.loglog(h, l2, "o-", color=ps.C["thermal"], label="relative $L_2$ error"); ax.loglog(h, li, "s-", color=ps.C["electrical"], label="$L_\\infty$ error (K)")
hh = np.array([h.min() * 0.85, h.max() * 1.15])
for q, ls, lab in ((1, (0, (1, 1.5)), "slope 1"), (2, (0, (3, 2)), "slope 2")):
    ax.loglog(hh, l2[1] * (hh / h[1]) ** q, color=ps.RULE, ls=ls, lw=0.7, label=lab)
ax.set_xlabel("mean element size $h$ (mm)"); ax.set_ylabel("error"); ax.set_xticks([2, 3, 4, 5]); ax.set_xticklabels(["2", "3", "4", "5"]); ax.set_ylim(1.0e-3, 0.7)
ax.text(0.04, 0.05, f"observed order $p$ = {p:.2f}\nP1 tetrahedra, {ver['mms_coarse']['nodes']:,} / {ver['mms_full']['nodes']:,} nodes", transform=ax.transAxes, fontsize=6, va="bottom", color=ps.INK)
ps.finish_legend(ax, loc="upper left", fontsize=5.5)
ps.label(ax, "d")

# e: GCI
ax = axs[4]
names = {"corneal_apex_C": "corneal apex T", "canthus_skin_C": "canthus skin T", "thermopile_DCI_C": "thermopile 5° reading", "deficit_50pct_choroid_C": "ΔT at 50 % choroidal perfusion"}
keys = list(names); g = [ver["gci"][k]["GCI_fine_percent"] for k in keys]; y = np.arange(len(keys))[::-1]
ax.barh(y, g, color=[ps.C["thermal"] if q < 1 else ps.C["warn"] for q in g], height=0.6)
for yi, q, k in zip(y, g, keys):
    ax.text(q * 1.15, yi, f"{q:.2f} %", va="center", fontsize=5.5, color=ps.INK)
ax.set_xscale("log"); ax.set_xlim(3e-3, 60); ax.set_yticks(y); ax.set_yticklabels([names[k] for k in keys], fontsize=5.5)
ax.set_xlabel("grid convergence index, fine mesh (%)")
ax.axvline(5, color=ps.RULE, lw=0.6, ls=(0, (3, 2))); ax.text(5, len(keys) - 0.45, "5 %", fontsize=5, color=ps.MUTED, ha="center", va="bottom")
ps.label(ax, "e", dx_mm=-14)

# f: Sobol total-order grid
ax = axs[5]
PN = {"choroid_frac": "choroidal perfusion", "skin_perf_frac": "skin perfusion", "eyelid_perf_frac": "eyelid perfusion", "orbit_perf_frac": "orbital perfusion",
      "T_amb": "ambient temperature", "h_skin": "skin convection", "E_tear": "tear evaporation", "k_sclera_scale": "scleral conductivity"}
ON = {"apex_C": "corneal apex", "canthus_C": "canthus skin", "dci_limbus_C": "thermopile at limbus", "dci_canthus_C": "thermopile at canthus"}
params = [q[0] for q in UQ["params"]]; outs = UQ["outputs"]
ST = np.clip(np.array(UQ["ST"]), 0, None); S1 = np.clip(np.array(UQ["S1"]), 0, None)
im = ax.imshow(ST, cmap=ccm.lajolla_r if hasattr(ccm, "lajolla_r") else "YlOrRd", vmin=0, vmax=1, aspect="auto")
for i in range(ST.shape[0]):
    for j in range(ST.shape[1]):
        vv = ST[i, j]
        ax.text(j, i, f"{vv:.2f}" if vv >= 0.01 else "<0.01", ha="center", va="center", fontsize=5, color="white" if vv > 0.55 else ps.INK)
NL = chr(10)
ON2 = {"apex_C": "corneal" + NL + "apex", "canthus_C": "canthus" + NL + "skin", "dci_limbus_C": "thermopile" + NL + "at limbus", "dci_canthus_C": "thermopile" + NL + "at canthus"}
ax.set_xticks(range(len(outs))); ax.set_xticklabels([ON2[o] for o in outs], fontsize=5.2)
ax.set_yticks(range(len(params))); ax.set_yticklabels([PN[q] for q, lo, hi in UQ["params"]], fontsize=5.5)
ax.tick_params(length=0); ax.spines[["top", "right"]].set_visible(True)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label(f"total-order Sobol index ({UQ['n_solves']} solves)", fontsize=5.5); cb.ax.tick_params(labelsize=5)
ps.label(ax, "f", dx_mm=-20)

ps.save(fig, "figM03_thermal_credibility")
