"""
Main Fig. 4 (npj manuscript): optical, electrical and pupil forward models and the design verdicts they return.
Composite of panels from figF06 (a, b), figF04 (c), figF07 (d, e) and figF08 (f). Same data files, same pubstyle.
  a  Photon measurement density of the 3-mm temple PPG at 880 nm with tissue boundaries.
  b  Arterial share of the pulsatile signal and detected photon fraction vs source-detector spacing.
  c  Steady heating of cornea, lid and canthus vs 940-nm LED power up to the IEC 62471 limit.
  d  Tissue shares of the tetrapolar transfer impedance at 50 kHz.
  e  Sensitivity of |Z| to added orbital water: tetrapolar vs bipolar with contact impedance.
  f  Pupil constriction amplitude vs stimulus illuminance (a 940-nm illuminator evokes no measurable reflex).
Run: python results/src/figM04_optical_electrical_pupil.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmcrameri import cm as ccm

ROOT = pathlib.Path(__file__).resolve().parents[1]; SIM = ROOT.parent / "sim" / "out"
O = json.loads((SIM / "optical_results.json").read_text(encoding="utf-8"))
M = json.loads((SIM / "voxel_meta.json").read_text(encoding="utf-8"))["rois"]
opt = json.loads((SIM / "opto_thermal.json").read_text(encoding="utf-8"))
B = json.loads((SIM / "bioimpedance3d_coarse.json").read_text(encoding="utf-8"))
P = json.loads((SIM / "pupil_model.json").read_text(encoding="utf-8"))
TN = {"artery_angular": "angular artery", "artery_temporal": "temporal artery", "cornea": "cornea", "sclera": "sclera", "lens": "lens", "aqueous": "aqueous",
      "vitreous": "vitreous", "eyelid": "eyelid", "orbital_fat": "orbital fat", "skin": "skin", "fat_subcutaneous": "subcutaneous fat", "muscle": "muscle",
      "bone_cortical": "bone", "brain_grey": "brain"}
ORBIT = {"artery_angular", "cornea", "sclera", "lens", "aqueous", "vitreous", "eyelid", "orbital_fat"}

ps.apply()
fig = plt.figure(figsize=(ps.DOUBLE_MM * ps.MM, 112 * ps.MM), layout="constrained")
gs = fig.add_gridspec(2, 3, width_ratios=[1.15, 1, 1])
axs = [fig.add_subplot(gs[i, j]) for i in range(2) for j in range(3)]

# a: 880-nm banana
ax = axs[0]
src, det = np.array(O["ppg"]["src_grid"], int), np.array(O["ppg"]["det_grid"], int)
res = M["temple"]["res_mm"]; lab = np.load(SIM / "labels_temple.npy", mmap_mode="r"); zi = src[2]
labsl = np.asarray(lab[:, :, zi]).T
ban = np.asarray(np.load(SIM / "ppg_banana_880.npy", mmap_mode="r")[:, :, zi]).T
ban = ban / np.percentile(ban[ban > 0], 99.9); ban = np.where(np.isfinite(ban) & (ban > 0), ban, 1e-7)
x0 = max(0, min(src[0], det[0]) - int(3.5 / res)); x1 = min(ban.shape[1], max(src[0], det[0]) + int(0.6 / res))
y0 = max(0, min(src[1], det[1]) - int(4 / res)); y1 = min(ban.shape[0], max(src[1], det[1]) + int(4 / res))
depth = (src[0] - np.arange(x0, x1)) * res; along = (np.arange(y0, y1) - y0) * res
img = ban[y0:y1, x0:x1].T[::-1]; ext = [0, along[-1], depth.max(), depth.min()]
im = ax.imshow(img, cmap=ccm.lipari, norm=LogNorm(1e-5, 1), extent=ext, aspect="equal", interpolation="bilinear")
ax.contour(along, depth, labsl[y0:y1, x0:x1].T, levels=[0.5, 2.5, 10.5, 11.5, 12.5, 13.5], colors="white", linewidths=0.4, alpha=0.9)
ax.plot([(src[1] - y0) * res, (det[1] - y0) * res], [0, 0], "v", color="white", ms=3, mec=ps.INK, mew=0.4)
ax.set_ylim(depth.max(), depth.min()); ax.set_ylabel("depth (mm)"); ax.set_xlabel("along the skin (mm)")
w = O["ppg"]["wavelengths"]["880"]
ax.set_title(f"880 nm, {O['ppg']['pd_spacing_mm']:.0f} mm spacing" + "\n" + f"sampled depth {w['banana_weighted_depth_mm']:.1f} mm, arterial share {w['arterial_attenuation_share']*100:.1f} %", fontsize=5.5, loc="right")
ax.spines[["top", "right"]].set_visible(True)
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("photon measurement density (normalized)", fontsize=5.5); cb.ax.tick_params(labelsize=5)
ps.label(ax, "a")

# b: spacing sweep
ax = axs[1]
sw = O["ppg"]["spacing_sweep_880"]; sp = [r["spacing_mm"] for r in sw]
ax.plot(sp, [r["artery_share_of_pulse"] * 100 for r in sw], "o-", color=ps.C["optical"], label="arterial share of pulsatile signal")
ax.set_xlabel("source to detector spacing (mm)"); ax.set_ylabel("arterial share of AC signal (%)", color=ps.C["optical"]); ax.tick_params(axis="y", colors=ps.C["optical"])
ax2 = ax.twinx(); ax2.semilogy(sp, [r["detected_fraction"] for r in sw], "s--", color=ps.C["ref"], label="detected photon fraction")
ax2.set_ylabel("detected photon fraction", color=ps.C["ref"]); ax2.tick_params(axis="y", colors=ps.C["ref"]); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(ps.C["ref"])
ax.axvline(O["ppg"]["pd_spacing_mm"], color=ps.RULE, lw=0.6, ls=(0, (3, 2)))
ax.text(O["ppg"]["pd_spacing_mm"] + 0.3, ax.get_ylim()[0] + 0.03 * np.ptp(ax.get_ylim()), "current\ndesign", fontsize=5, color=ps.MUTED, va="bottom")
ax.set_title("880 nm, temple", fontsize=6, loc="right")
ps.label(ax, "b")

# c: LED heating
ax = axs[2]
Pw = np.array(sorted(float(k) for k in opt["led_power_mW"]))
for site, col, lab in (("apex", ps.C["thermal"], "corneal apex"), ("lid", ps.C["optical"], "eyelid"), ("canthus", ps.C["pupil"], "medial canthus")):
    ax.plot(Pw, [opt["led_power_mW"][f"{v:.1f}"][site] for v in Pw], "o-", color=col, label=lab)
ax.set_xlabel("940-nm LED optical power (mW)"); ax.set_ylabel("steady heating (°C)"); ax.set_xlim(0, 46); ax.set_ylim(0, 0.29)
lim = O["led"]["max_led_optical_power_mW"]
ax.axvline(lim, color=ps.RULE, lw=0.6, ls=(0, (3, 2)))
ax.text(lim - 1, 0.285, f"IEC 62471 corneal limit\n{O['led']['iec62471_cornea_limit_W_m2']:.0f} W m$^{{-2}}$ = {lim:.1f} mW", fontsize=5.5, ha="right", va="top", color=ps.MUTED)
ps.finish_legend(ax, loc="center left", fontsize=5.5)
ps.label(ax, "c")

# d: tissue share at 50 kHz
ax = axs[3]
share = B["tissue_share_50kHz"]; keys = sorted(share, key=lambda k: share[k]); vals = np.array([share[k] for k in keys]) * 100; y = np.arange(len(keys))
ax.barh(y, vals, color=[ps.C["electrical"] if k in ORBIT else ps.C["ref"] for k in keys], height=0.7)
for yi, vv in zip(y, vals):
    ax.text(max(vv, 0.03) * 1.12, yi, f"{vv:.2f} %" if vv >= 0.01 else "<0.01 %", va="center", fontsize=5, color=ps.INK)
ax.set_xscale("log"); ax.set_xlim(0.01, 300); ax.set_yticks(y); ax.set_yticklabels([TN[k] for k in keys], fontsize=5.5)
ax.set_xlabel("share of tetrapolar |Z| at 50 kHz (%)")
ax.text(0.98, 0.04, f"orbit, eye and lid: {B['orbit_share_50kHz']*100:.1f} %", transform=ax.transAxes, fontsize=5.5, ha="right", color=ps.C["electrical"])
ps.label(ax, "d", dx_mm=-15)

# e: edema sensitivity
ax = axs[4]
ed = B["edema_50kHz"]; ks = sorted(ed, key=float); fr = np.array([float(k) for k in ks])
for key, fmt, col, lab in (("dZ_abs_percent", "o-", ps.C["electrical"], "tetrapolar"), ("bipolar_tissue_dZ_abs_percent", "s--", ps.C["ref"], "bipolar, tissue only"),
                           ("bipolar_total_dZ_abs_percent", "^:", ps.C["warn"], "bipolar, with contact Z")):
    yy = [0, *[ed[k][key] for k in ks]]
    ax.plot([0, *fr * 100], yy, fmt, color=col); ax.text(fr.max() * 100 + 1.5, yy[-1], lab, fontsize=5, va="center", ha="left", color=col)
ax.set_xlim(-1, fr.max() * 100 + 30); ax.axhline(0, color=ps.RULE, lw=0.5)
ax.set_xlabel("added orbital water (%)"); ax.set_ylabel("change in |Z| at 50 kHz (%)")
tc = B["thermal_coupling_50kHz"]["dZ_abs_percent"]
ax.text(0.45, 0.62, f"thermal coupling: +{tc:.1f} % |Z|\nover the simulated temperature field", transform=ax.transAxes, fontsize=5, va="center", ha="left", color=ps.MUTED)
ps.label(ax, "e")

# f: pupil stimulus sweep
ax = axs[5]
ss = sorted(P["stimulus_sweep"], key=lambda r: r["stimulus_lux"]); lux = np.array([r["stimulus_lux"] for r in ss]); amp = np.array([r["amplitude_mm"] for r in ss]); m = lux > 0
ax.semilogx(lux[m], amp[m], "o-", color=ps.C["pupil"])
a0 = amp[~m][0]
ax.axhline(a0, color=ps.C["warn"], lw=0.8, ls=(0, (3, 2)))
ax.text(lux[m].max() * 0.9, a0 + 0.10, f"0 lx, 940 nm only:\n{a0:.2f} mm, no measurable reflex", fontsize=5, color=ps.C["warn"], va="bottom", ha="right")
h, g = P["healthy"], P["gain_0.92"]
ax.text(0.03, 0.97, f"autonomic gain 1.0 to 0.92 at 300 lx:\namplitude {h['amplitude_mm']:.2f} to {g['amplitude_mm']:.2f} mm\nlatency {h['latency_s']*1e3:.0f} to {g['latency_s']*1e3:.0f} ms", transform=ax.transAxes, fontsize=5, va="top", color=ps.INK)
ax.set_xlabel("stimulus illuminance (lx)"); ax.set_ylabel("constriction amplitude (mm)"); ax.set_ylim(0, amp.max() * 1.3)
ps.label(ax, "f")

ps.save(fig, "figM04_optical_electrical_pupil")
