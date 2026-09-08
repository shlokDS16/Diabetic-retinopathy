"""
F8 - Pupillography model (Longtin-Milton delay-differential equation with an autonomic gain g).
  a  Pupil-diameter traces, healthy (g = 1) vs reduced autonomic gain (g = 0.92).
  b  Reflex metrics vs autonomic gain, normalised to healthy.
  c  Constriction amplitude vs stimulus illuminance: a 940-nm illuminator delivers ~0 photopic lux.
  d  Camera geometry: off-axis vergence foreshortens the pupil; naive minor-axis reading under-reads.
Data: sim/out/pupil_model.json.
Run: python results/src/figF08_pupil.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Circle

SIM = pathlib.Path(__file__).resolve().parents[2] / "sim" / "out"
P = json.loads((SIM / "pupil_model.json").read_text(encoding="utf-8"))

ps.apply()
fig, axs = ps.figure(cols=4, width="double", height_mm=52)

# a: traces
ax = axs[0]
t = np.array(P["traces"]["t"])
ax.plot(t, P["traces"]["D_healthy"], color=ps.C["pupil"], label="healthy, $g$ = 1")
ax.plot(t, P["traces"]["D_gain_0.92"], color=ps.C["warn"], label="autonomic gain $g$ = 0.92")
h, g = P["healthy"], P["gain_0.92"]
ax.set_xlabel("time (s)"); ax.set_ylabel("pupil diameter (mm)")
ax.text(0.98, 0.04, f"amplitude {h['amplitude_mm']:.2f} → {g['amplitude_mm']:.2f} mm\nlatency {h['latency_s']*1e3:.0f} → {g['latency_s']*1e3:.0f} ms\nMCV {h['max_constriction_velocity_mm_s']:.1f} → {g['max_constriction_velocity_mm_s']:.1f} mm s$^{{-1}}$",
        transform=ax.transAxes, fontsize=5, ha="right", va="bottom", color=ps.INK)
ps.finish_legend(ax, loc="upper right", fontsize=5.5)
ps.label(ax, "a")

# b: metrics vs gain (normalised)
ax = axs[1]
sw = sorted(P["gain_sweep"], key=lambda r: r["gain"]); gg = np.array([r["gain"] for r in sw])
ref = {k: v for k, v in P["healthy"].items()}
for key, lab, col in (("amplitude_mm", "constriction amplitude", ps.C["pupil"]), ("max_constriction_velocity_mm_s", "max constriction velocity", ps.C["electrical"]),
                      ("avg_dilation_velocity_mm_s", "mean dilation velocity", ps.C["optical"]), ("latency_s", "latency", ps.C["warn"])):
    ax.plot(gg, [r[key] / ref[key] * 100 for r in sw], "o-", color=col, label=lab, ms=2.5)
ax.axhline(100, color=ps.RULE, lw=0.5); ax.axvline(0.92, color=ps.RULE, lw=0.5, ls=(0, (3, 2)))
ax.set_xlabel("autonomic gain $g$"); ax.set_ylabel("% of healthy value")
ax.set_xlim(0.28, 1.02); ax.set_ylim(15, 200); ax.set_yticks(range(20, 141, 20))
ps.finish_legend(ax, loc="upper center", fontsize=5, ncol=2, bbox_to_anchor=(0.5, 1.02))
ps.label(ax, "b")

# c: stimulus sweep
ax = axs[2]
ss = sorted(P["stimulus_sweep"], key=lambda r: r["stimulus_lux"])
lux = np.array([r["stimulus_lux"] for r in ss]); amp = np.array([r["amplitude_mm"] for r in ss])
m = lux > 0
ax.semilogx(lux[m], amp[m], "o-", color=ps.C["pupil"])
ax.axhline(amp[~m][0] if (~m).any() else 0, color=ps.C["warn"], lw=0.8, ls=(0, (3, 2)))
ax.text(lux[m].max() * 0.9, (amp[~m][0] if (~m).any() else 0) + 0.08, "0 lx: 940-nm illuminator,\nphotopic V(λ) ≈ 0, no reflex", fontsize=5, color=ps.C["warn"], va="bottom", ha="right")
ax.set_xlabel("stimulus illuminance (lx)"); ax.set_ylabel("constriction amplitude (mm)")
ax.set_ylim(0, amp.max() * 1.25)
ps.label(ax, "c")

# d: camera foreshortening
ax = axs[3]
cam = P["camera"]; r = cam["minor_over_major"]
ax.add_patch(Circle((0, 0), 1, fill=False, ec=ps.C["ref"], lw=0.8, ls=(0, (3, 2))))
ax.add_patch(Ellipse((0, 0), 2 * r, 2, fill=True, fc=ps.C["pupil"], alpha=0.25, ec=ps.C["pupil"], lw=0.9))
ax.annotate("", xy=(r, 0), xytext=(-r, 0), arrowprops=dict(arrowstyle="<->", lw=0.6, color=ps.INK))
ax.text(0, -0.12, f"minor = {r:.2f} D", ha="center", va="top", fontsize=5.5, color=ps.INK)
ax.annotate("", xy=(0, 1), xytext=(0, -1), arrowprops=dict(arrowstyle="<->", lw=0.6, color=ps.C["ref"]))
ax.text(0.06, 1.04, "major = D", fontsize=5.5, color=ps.C["ref"], va="bottom")
ax.set_xlim(-1.6, 1.6); ax.set_ylim(-2.0, 1.55); ax.set_aspect("equal"); ax.axis("off")
ax.text(0, 1.55, f"camera {cam['working_distance_mm']:.0f} mm off-axis, vergence {cam['vergence_deg']:.0f}°", ha="center", va="top", fontsize=5.5, color=ps.INK)
ax.text(0, -1.15, f"naive minor-axis read: −{cam['naive_underread_percent']:.0f} %\ncorrected: major axis (or minor / cos θ)\npixel pitch at pupil {cam['pixel_pitch_at_pupil_mm']*1e3:.0f} µm",
        ha="center", va="top", fontsize=5, color=ps.INK)
ps.label(ax, "d")

ps.save(fig, "figF08_pupil")
