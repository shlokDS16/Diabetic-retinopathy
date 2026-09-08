"""
F4 - Verification of the thermal solver and the opto-thermal load.
  a  Method of manufactured solutions: relative L2 and Linf error vs mesh size h, observed order.
  b  Grid-convergence index (Roache) for the four quantities of interest.
  c  Corneal / lid / canthus heating from the 940-nm illuminator vs LED optical power, IEC 62471 limit.
Data: sim/out/verification.json, sim/out/opto_thermal.json, sim/out/optical_results.json.
Run: python results/src/figF04_verification.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt

SIM = pathlib.Path(__file__).resolve().parents[2] / "sim" / "out"
ver = json.loads((SIM / "verification.json").read_text(encoding="utf-8"))
opt = json.loads((SIM / "opto_thermal.json").read_text(encoding="utf-8"))
led = json.loads((SIM / "optical_results.json").read_text(encoding="utf-8"))["led"]

ps.apply()
fig, axs = ps.figure(cols=3, width="double", height_mm=58)

# a: MMS convergence
ax = axs[0]
h = np.array([ver["mms_coarse"]["h_mm"], ver["mms_full"]["h_mm"]])
l2 = np.array([ver["mms_coarse"]["L2_rel"], ver["mms_full"]["L2_rel"]])
li = np.array([ver["mms_coarse"]["Linf_K"], ver["mms_full"]["Linf_K"]])
p = ver["observed_order"]
ax.loglog(h, l2, "o-", color=ps.C["thermal"], label="relative $L_2$ error")
ax.loglog(h, li, "s-", color=ps.C["electrical"], label="$L_\\infty$ error (K)")
hh = np.array([h.min() * 0.85, h.max() * 1.15])
for q, ls, lab in ((1, (0, (1, 1.5)), "slope 1"), (2, (0, (3, 2)), "slope 2")):
    ax.loglog(hh, l2[1] * (hh / h[1]) ** q, color=ps.RULE, ls=ls, lw=0.7, label=lab)
ax.set_xlabel("mean element size $h$ (mm)"); ax.set_ylabel("error")
ax.set_xticks([2, 3, 4, 5]); ax.set_xticklabels(["2", "3", "4", "5"]); ax.set_ylim(1.0e-3, 0.7)
ax.text(0.04, 0.05, f"observed order $p$ = {p:.2f}\nP1 tetrahedra, {ver['mms_coarse']['nodes']:,} / {ver['mms_full']['nodes']:,} nodes",
        transform=ax.transAxes, fontsize=6, va="bottom", color=ps.INK)
ps.finish_legend(ax, loc="upper left", fontsize=5.5)
ps.label(ax, "a")

# b: GCI
ax = axs[1]
names = {"corneal_apex_C": "corneal apex T", "canthus_skin_C": "canthus skin T", "thermopile_DCI_C": "thermopile (DCI) reading",
         "deficit_50pct_choroid_C": "ΔT at 50 % choroidal flow"}
keys = list(names); g = [ver["gci"][k]["GCI_fine_percent"] for k in keys]
y = np.arange(len(keys))[::-1]
ax.barh(y, g, color=[ps.C["thermal"] if v < 1 else ps.C["warn"] for v in g], height=0.6)
for yi, v, k in zip(y, g, keys):
    fine = ver["gci"][k]["fine"]; rx = ver["gci"][k]["richardson_extrapolated"]
    ax.text(v * 1.15, yi, f"{v:.2f} %  (fine {fine:.2f}, extrap. {rx:.2f} °C)", va="center", fontsize=5.5, color=ps.INK)
ax.set_xscale("log"); ax.set_xlim(3e-3, 3e3)
ax.set_yticks(y); ax.set_yticklabels([names[k] for k in keys])
ax.set_xlabel("GCI on the fine mesh (%)")
ax.axvline(5, color=ps.RULE, lw=0.6, ls=(0, (3, 2))); ax.text(5, len(keys) - 0.45, "5 % (ASME V&V 20 typical)", fontsize=5, color=ps.MUTED, ha="center", va="bottom")
ps.label(ax, "b")

# c: opto-thermal heating vs LED power
ax = axs[2]
P = np.array(sorted(float(k) for k in opt["led_power_mW"]))
for site, col, lab in (("apex", ps.C["thermal"], "corneal apex"), ("lid", ps.C["optical"], "eyelid"), ("canthus", ps.C["pupil"], "medial canthus")):
    dT = [opt["led_power_mW"][f"{v:.1f}"][site] for v in P]
    ax.plot(P, dT, "o-", color=col, label=lab)
ax.set_xlabel("940-nm LED optical power (mW)"); ax.set_ylabel("steady-state heating ΔT (°C)")
ax.set_xlim(0, 46); ax.set_ylim(0, 0.29)
lim = led["max_led_optical_power_mW"]
ax.axvline(lim, color=ps.RULE, lw=0.6, ls=(0, (3, 2)))
ax.text(lim - 1, 0.285, f"IEC 62471 corneal limit\n{led['iec62471_cornea_limit_W_m2']:.0f} W m$^{{-2}}$ → {lim:.1f} mW", fontsize=5.5, ha="right", va="top", color=ps.MUTED)
ps.finish_legend(ax, loc="center left", fontsize=5.5)
ps.label(ax, "c")

ps.save(fig, "figF04_verification")
