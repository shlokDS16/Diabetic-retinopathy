"""
F6 - Monte Carlo optics (MCX): what the temple PPG samples, and where the 940-nm illuminator's light goes.
  a  Photon-measurement-density ("banana") at 660 and 880 nm through the source-detector plane of the temple
     PPG at 3 mm spacing, with tissue boundaries; the sampled depth is dermal, the temporal artery is barely touched.
  b  Source-detector spacing sweep at 880 nm: arterial share of the pulsatile signal vs detected photon fraction.
  c  Fraction of 940-nm LED power absorbed per tissue (IEC 62471 corneal limit sets the power).
  d  Absorbed 940-nm power density through the mid-sagittal plane of the eye ROI.
Data: sim/out/optical_results.json, ppg_banana_{660,880}.npy, labels_temple.npy, led_absorbed_940.npy, labels_eye.npy, voxel_meta.json.
Run: python results/src/figF06_optical.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from cmcrameri import cm as ccm

SIM = pathlib.Path(__file__).resolve().parents[2] / "sim" / "out"
O = json.loads((SIM / "optical_results.json").read_text(encoding="utf-8"))
M = json.loads((SIM / "voxel_meta.json").read_text(encoding="utf-8"))["rois"]
TN = {"artery_angular": "angular artery", "artery_temporal": "temporal artery", "cornea": "cornea", "sclera": "sclera", "lens": "lens",
      "aqueous": "aqueous", "vitreous": "vitreous", "eyelid": "eyelid", "orbital_fat": "orbital fat", "skin": "skin",
      "fat_subcutaneous": "subcutaneous fat", "muscle": "muscle", "bone_cortical": "bone", "brain_grey": "brain"}

ps.apply()
fig = plt.figure(figsize=(ps.DOUBLE_MM * ps.MM, 62 * ps.MM), layout="constrained")
gs = fig.add_gridspec(1, 4, width_ratios=[1.35, 0.95, 1.0, 1.0])
axs = [fig.add_subplot(gs[0, i]) for i in range(4)]

# a: bananas (two wavelengths stacked in one panel via inset grid)
ax = axs[0]; ax.axis("off")
sub = ax.inset_axes([0, 0.52, 1, 0.48]), ax.inset_axes([0, 0.0, 1, 0.48])
src, det = np.array(O["ppg"]["src_grid"], int), np.array(O["ppg"]["det_grid"], int)
res = M["temple"]["res_mm"]; lab = np.load(SIM / "labels_temple.npy", mmap_mode="r")
zi = src[2]
labsl = np.asarray(lab[:, :, zi]).T                                   # rows = y (along skin), cols = x (depth)
for a, wl in zip(sub, ("660", "880")):
    ban = np.asarray(np.load(SIM / f"ppg_banana_{wl}.npy", mmap_mode="r")[:, :, zi]).T
    ban = ban / np.percentile(ban[ban > 0], 99.9); ban = np.where(np.isfinite(ban) & (ban > 0), ban, 1e-7)
    # temple ROI: x increases laterally, so tissue depth = (x_src - x) * res; skin at the source row
    x0 = max(0, min(src[0], det[0]) - int(3.5 / res)); x1 = min(ban.shape[1], max(src[0], det[0]) + int(0.6 / res))
    y0 = max(0, min(src[1], det[1]) - int(4 / res)); y1 = min(ban.shape[0], max(src[1], det[1]) + int(4 / res))
    depth = (src[0] - np.arange(x0, x1)) * res                          # per column, increasing toward x0
    along = (np.arange(y0, y1) - y0) * res
    img = ban[y0:y1, x0:x1].T[::-1]                                       # rows: shallow (air) -> deep
    ext = [0, along[-1], depth.max(), depth.min()]
    a.imshow(img, cmap=ccm.lipari, norm=LogNorm(1e-5, 1), extent=ext, aspect="equal", interpolation="bilinear")
    a.contour(along, depth, labsl[y0:y1, x0:x1].T, levels=[0.5, 2.5, 10.5, 11.5, 12.5, 13.5], colors="white", linewidths=0.4, alpha=0.9)
    a.plot([(src[1] - y0) * res, (det[1] - y0) * res], [0, 0], "v", color="white", ms=3, mec=ps.INK, mew=0.4)
    a.set_ylim(depth.max(), depth.min())
    a.set_ylabel("depth (mm)", fontsize=5.5); a.tick_params(labelsize=5)
    if wl == "880":
        a.set_xlabel("along skin (mm)", fontsize=5.5)
    else:
        a.tick_params(labelbottom=False)
    w = O["ppg"]["wavelengths"][wl]
    a.set_title(f"{wl} nm · sampled depth {w['banana_weighted_depth_mm']:.1f} mm · arterial {w['arterial_attenuation_share']*100:.1f} %", fontsize=5.5, loc="right")
    a.spines[["top", "right"]].set_visible(True)
ps.label(sub[0], "a")

# b: spacing sweep
ax = axs[1]
sw = O["ppg"]["spacing_sweep_880"]; sp = [r["spacing_mm"] for r in sw]
ax.plot(sp, [r["artery_share_of_pulse"] * 100 for r in sw], "o-", color=ps.C["optical"], label="arterial share of pulsatile signal")
ax.set_xlabel("source–detector spacing (mm)"); ax.set_ylabel("arterial share of AC signal (%)", color=ps.C["optical"]); ax.tick_params(axis="y", colors=ps.C["optical"])
ax2 = ax.twinx(); ax2.semilogy(sp, [r["detected_fraction"] for r in sw], "s--", color=ps.C["ref"], label="detected photon fraction")
ax2.set_ylabel("detected photon fraction", color=ps.C["ref"]); ax2.tick_params(axis="y", colors=ps.C["ref"]); ax2.spines["right"].set_visible(True); ax2.spines["right"].set_color(ps.C["ref"])
ax.axvline(O["ppg"]["pd_spacing_mm"], color=ps.RULE, lw=0.6, ls=(0, (3, 2)))
ax.text(O["ppg"]["pd_spacing_mm"] + 0.3, ax.get_ylim()[0] + 0.03 * np.ptp(ax.get_ylim()), "current\ndesign", fontsize=5, color=ps.MUTED, va="bottom")
ax.set_title("880 nm, temple", fontsize=6)
ps.label(ax, "b")

# c: absorbed fraction per tissue (LED)
ax = axs[2]
af = O["led"]["absorbed_fraction_per_tissue"]; keys = sorted(af, key=lambda k: af[k])
vals = np.array([af[k] for k in keys]) * 100; y = np.arange(len(keys))
ocular = {"cornea", "sclera", "lens", "aqueous", "vitreous"}
ax.barh(y, vals, color=[ps.C["optical"] if k in ocular else (ps.C["external"] if k == "eyelid" else ps.C["ref"]) for k in keys], height=0.7)
for yi, v in zip(y, vals):
    ax.text(v + 0.3, yi, f"{v:.1f}", fontsize=5, va="center", color=ps.INK)
ax.set_yticks(y); ax.set_yticklabels([TN[k] for k in keys], fontsize=5.5); ax.set_xlim(0, vals.max() * 1.25)
ax.set_xlabel("absorbed fraction of LED power (%)")
L = O["led"]
ax.set_title(f"940 nm: {L['absorbed_total_fraction']*100:.0f} % of LED power absorbed\nIEC 62471 corneal limit → {L['max_led_optical_power_mW']:.1f} mW", fontsize=6, loc="right")
ps.label(ax, "c", dx_mm=-14)

# d: absorbed field slice (eye ROI)
ax = axs[3]
E = M["eye"]; resE = E["res_mm"]; labE = np.load(SIM / "labels_eye.npy", mmap_mode="r"); A = np.load(SIM / "led_absorbed_940.npy", mmap_mode="r")
xi = int(np.argmax([(np.asarray(labE[i]) == 7).sum() for i in range(0, labE.shape[0], 2)]) * 2)      # plane through the vitreous centre
sl = np.asarray(A[xi]).T; ls = np.asarray(labE[xi]).T                                                  # rows = z (up), cols = y (anterior)
sl = np.where(sl > 0, sl, np.nan) / np.nanmax(sl)
ext = [E["origin_mm"][1], E["origin_mm"][1] + sl.shape[1] * resE, E["origin_mm"][2], E["origin_mm"][2] + sl.shape[0] * resE]
im = ax.imshow(sl, origin="lower", cmap=ccm.lipari, norm=LogNorm(1e-4, 1), extent=ext, aspect="equal", interpolation="bilinear")
ax.contour(np.linspace(ext[0], ext[1], sl.shape[1]), np.linspace(ext[2], ext[3], sl.shape[0]), ls, levels=[0.5, 2.5, 4.5, 7.5, 8.5, 9.5], colors="white", linewidths=0.35, alpha=0.8)
ax.set_xlabel("anterior (mm)"); ax.set_ylabel("up (mm)")
cb = fig.colorbar(im, ax=ax, fraction=0.05, pad=0.02); cb.set_label("absorbed power density (normalised)", fontsize=5.5); cb.ax.tick_params(labelsize=5)
ax.set_title("absorbed 940-nm field\nmid-globe sagittal plane", fontsize=6, loc="right")
ax.spines[["top", "right"]].set_visible(True)
ps.label(ax, "d")

ps.save(fig, "figF06_optical")
