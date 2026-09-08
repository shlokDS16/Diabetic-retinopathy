"""
F3 - Bioheat field and what the thermopile actually reads.
  a  Sagittal temperature slice through the left globe (3-D Pennes, 122k-node mesh) with tissue boundaries.
  b  Axial slice at pupil height.
  c  Readouts vs choroidal perfusion: corneal apex, thermopile in the two fields of view (DCI 5 deg, BAA 90 deg), canthus.
  d  Validation against published ranges (corneal apex, inner canthus) and the skin-perfusion sweep.
Data: results/out/thermal_slices.npz (from prep_thermal_slices.py), sim/out/thermal3d_full.json.
Run: python results/src/figF03_thermal.py
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
TMIN, TMAX = 31.0, 37.4


def region_edges(pts, tris, region):
    """Segments between adjacent triangles of different region (tissue boundaries), plus the outer boundary."""
    e = np.concatenate([tris[:, [0, 1]], tris[:, [1, 2]], tris[:, [2, 0]]]); e.sort(axis=1)
    reg = np.tile(region, 3)
    order = np.lexsort((e[:, 1], e[:, 0])); e, reg = e[order], reg[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    segs = []
    i = 0
    while i < len(e):
        if i + 1 < len(e) and same[i]:
            if reg[i] != reg[i + 1]:
                segs.append(pts[e[i]])
            i += 2
        else:
            segs.append(pts[e[i]]); i += 1
    return np.array(segs)


def slice_panel(ax, name, xlab, ylab, title):
    pts, tris, T, reg = S[f"{name}_pts"], S[f"{name}_tris"], S[f"{name}_T"], S[f"{name}_region"]
    tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
    im = ax.tripcolor(tri, T, shading="gouraud", cmap=ccm.lipari, vmin=TMIN, vmax=TMAX)
    segs = region_edges(pts, tris, reg)
    ax.add_collection(LineCollection(segs, colors="white", linewidths=0.25, alpha=0.7))
    ax.set_aspect("equal"); ax.set_xlabel(xlab); ax.set_ylabel(ylab); ax.set_title(title, fontsize=6, loc="right")
    ax.spines[["top", "right"]].set_visible(True)
    return im


ps.apply()
fig = plt.figure(figsize=(ps.DOUBLE_MM * ps.MM, 64 * ps.MM), layout="constrained")
gs = fig.add_gridspec(1, 4, width_ratios=[1.15, 1.15, 1, 1])
axs = [fig.add_subplot(gs[0, i]) for i in range(4)]

# a: sagittal through the globe (y anterior, z up)
ax = axs[0]
im = slice_panel(ax, "sagittal_eye", "anterior (mm)", "up (mm)", "sagittal plane through the left globe")
ax.set_xlim(-45, 20); ax.set_ylim(-32, 32)
ps.label(ax, "a")

# b: axial at pupil height (x lateral, y anterior)
ax = axs[1]
slice_panel(ax, "axial_pupil", "lateral (mm)", "anterior (mm)", "axial plane at pupil height")
ax.set_xlim(0, 72); ax.set_ylim(-50, 20)
cb = fig.colorbar(im, ax=axs[:2], fraction=0.035, pad=0.01, location="bottom", shrink=0.8); cb.set_label("tissue temperature (°C)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
ps.label(ax, "b")

# c: readouts vs choroidal perfusion
ax = axs[2]
sw = sorted(T3["choroid_sweep"], key=lambda r: r["choroid_frac"]); cf = np.array([r["choroid_frac"] for r in sw]) * 100
base = {k: [r[k] for r in sw if r["choroid_frac"] == 1.0][0] for k in ("apex_C", "dci_C", "baa_C", "canthus_C")}
for k, lab, col, mk in (("apex_C", "corneal apex (true)", ps.C["thermal"], "o"), ("dci_C", f"thermopile 5° ({T3['thermopile_DCI_5deg']['spot_mm']:.1f} mm spot)", ps.C["electrical"], "s"),
                        ("baa_C", f"thermopile 90° ({T3['thermopile_BAA_90deg']['spot_mm']:.0f} mm spot)", ps.C["optical"], "^"), ("canthus_C", "canthus skin", ps.C["pupil"], "D")):
    ax.plot(cf, [r[k] - base[k] for r in sw], marker=mk, color=col, label=lab, ms=2.8)
ax.axhline(0, color=ps.RULE, lw=0.5)
ax.set_xlabel("choroidal perfusion (% of normal)"); ax.set_ylabel("change in reading (°C)")
ax.invert_xaxis()
ax.set_title("reading vs choroidal\nperfusion deficit", fontsize=6, loc="right")
ps.finish_legend(ax, loc="lower left", fontsize=5, handlelength=1.2)
ps.label(ax, "c")

# d: validation + skin perfusion sweep
ax = axs[3]
v = T3["validation"]
items = [("corneal apex", T3["corneal_apex_C"], v["corneal_apex_published_C"], ps.C["thermal"]),
         ("inner canthus skin", T3["medial_canthus_skin_C"], v["inner_canthus_published_C"], ps.C["pupil"]),
         ("temple skin", T3["temple_skin_C"], None, ps.C["ref"])]
y = np.arange(len(items))[::-1]
for yi, (lab, sim, pub, col) in zip(y, items):
    if pub:
        ax.plot(pub, [yi, yi], color=col, lw=4, alpha=0.3, solid_capstyle="butt")
        ax.text(pub[1] + 0.15, yi, f"published {pub[0]:.1f}–{pub[1]:.1f}", fontsize=5, va="center", color=ps.MUTED)
    ax.plot(sim, yi, "o", color=col, ms=4, mec="white", mew=0.4)
    ax.text(sim, yi + 0.28, f"{sim:.2f}", fontsize=5, ha="center", color=col)
sp = sorted(T3["skin_perfusion_sweep"], key=lambda r: r["skin_perf_frac"])
for r in sp:
    if r["skin_perf_frac"] < 1.0:
        ax.plot(r["canthus_C"], y[1], "|", color=ps.C["pupil"], ms=5, mew=0.8)
        ax.plot(r["temple_C"], y[2], "|", color=ps.C["ref"], ms=5, mew=0.8)
ax.text(sp[0]["canthus_C"] - 0.1, y[1] - 0.32, f"skin perfusion {int(sp[0]['skin_perf_frac']*100)} % → {int(sp[1]['skin_perf_frac']*100)} %", fontsize=5, ha="left", va="top", color=ps.MUTED)
ax.set_yticks(y); ax.set_yticklabels([i[0] for i in items], fontsize=5.5)
ax.set_xlabel("temperature (°C), ambient 25 °C"); ax.set_xlim(32.0, 37.2); ax.set_ylim(-0.8, len(items) - 0.3)
ax.set_title("simulated vs published (dot vs band)", fontsize=6, loc="right")
ps.label(ax, "d", dx_mm=-12)

ps.save(fig, "figF03_thermal")
