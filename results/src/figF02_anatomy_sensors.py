"""
F2 - The NRDI frame, its sensors, and the anatomical model they are simulated against.
  a  Product render of the v3 frame (Cycles path tracing of the build123d CAD).
  b  Exploded assembly with part callouts.
  c  Sagittal section of the 14-tissue periorbital finite-element model (122k nodes, 705k tetrahedra).
  d  Frame on the head with the simulated skin-temperature field (thermography render).
Data: sim/render/out/hero_product.png, explode_cycles_f090.png, physics_thermography.png; results/out/thermal_slices.npz;
      sim/out/periorbital_summary.json, sensor_layout.json.
Run: python results/src/figF02_anatomy_sensors.py
"""
import json, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.patches import Patch
from PIL import Image

ROOT = pathlib.Path(__file__).resolve().parents[1]; SIM = ROOT.parent / "sim"
S = np.load(ROOT / "out" / "thermal_slices.npz")
summ = json.loads((SIM / "out" / "periorbital_summary.json").read_text(encoding="utf-8"))
lay = json.loads((SIM / "out" / "sensor_layout.json").read_text(encoding="utf-8"))
REG = {1: "angular artery", 2: "temporal artery", 3: "cornea", 4: "sclera", 5: "lens", 6: "aqueous", 7: "vitreous", 8: "eyelid",
       9: "orbital fat", 10: "skin", 11: "subcutaneous fat", 12: "muscle", 13: "bone", 14: "brain"}
COL = {1: "#b2182b", 2: "#d6604d", 3: "#92c5de", 4: "#f7f7f7", 5: "#fddbc7", 6: "#d1e5f0", 7: "#4393c3", 8: "#f4a582",
       9: "#fee08b", 10: "#e0a080", 11: "#fff1c9", 12: "#c48a8a", 13: "#e6e6e6", 14: "#c7b7d6"}


def crop(path, box):
    im = Image.open(path).convert("RGB"); w, h = im.size
    l, t, r, b = box
    return im.crop((int(l * w), int(t * h), int(r * w), int(b * h)))


ps.apply()
fig = plt.figure(figsize=(ps.DOUBLE_MM * ps.MM, 118 * ps.MM), layout="constrained")
gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1, 1])
axa, axb, axc, axd = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

# a: hero
ax = axa
ax.imshow(crop(SIM / "render" / "out" / "hero_product.png", (0.0, 0.02, 1.0, 0.98))); ax.axis("off")
fr = lay["frame"]; an = lay["anthropometry"]
ax.set_title(f"NRDI frame v3 (path-traced CAD)\nlens {fr['lens_w_mm']:.0f} × {fr['lens_h_mm']:.0f} mm · bridge {fr['bridge_w_mm']:.0f} mm · IPD {an['ipd_mm']:.0f} mm · vertex {an['vertex_distance_mm']:.0f} mm",
             fontsize=6, loc="left")
ps.label(ax, "a", dx_mm=-2)

# b: explode
ax = axb
ax.imshow(crop(SIM / "render" / "out" / "explode_cycles_f090.png", (0.02, 0.03, 0.98, 0.97))); ax.axis("off")
n_sens = len(lay["sensors"]); kinds = sorted({s["kind"] for s in lay["sensors"]})
ax.set_title(f"exploded assembly with part callouts\n{n_sens} sensing components: thermopile, camera + 940-nm LED, Ag/AgCl electrodes, PPG", fontsize=6, loc="left")
ps.label(ax, "b", dx_mm=-2)

# c: tissue section
ax = axc
pts, tris, reg = S["sagittal_eye_pts"], S["sagittal_eye_tris"], S["sagittal_eye_region"]
tri = mtri.Triangulation(pts[:, 0], pts[:, 1], tris)
face = np.array([COL.get(int(r), "#ffffff") for r in reg])
ax.tripcolor(tri, facecolors=np.arange(len(reg)), cmap=plt.matplotlib.colors.ListedColormap(face), edgecolors="none", vmin=0, vmax=len(reg) - 1)
ax.set_aspect("equal"); ax.set_xlim(-48, 20); ax.set_ylim(-34, 34)
ax.set_xlabel("anterior (mm)"); ax.set_ylabel("up (mm)"); ax.spines[["top", "right"]].set_visible(True)
present = sorted({int(r) for r in reg})
handles = [Patch(facecolor=COL[k], edgecolor=ps.RULE, linewidth=0.3, label=REG[k]) for k in present]
ax.legend(handles=handles, loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=5, frameon=False, handlelength=1.0, handleheight=0.8, borderaxespad=0)
ax.set_title(f"finite-element anatomy, sagittal section through the left globe\n{summ['nodes']:,} nodes · {summ['tets']:,} tetrahedra · {len(summ['regions'])} tissues", fontsize=6, loc="left")
ps.label(ax, "c")

# d: thermography
ax = axd
ax.imshow(crop(SIM / "render" / "out" / "physics_thermography.png", (0.0, 0.0, 0.78, 1.0))); ax.axis("off")
ax.set_title("frame on the head with the simulated skin-temperature field\n3-D Pennes bioheat solution mapped onto the face", fontsize=6, loc="left")
ps.label(ax, "d", dx_mm=-2)

ps.save(fig, "figF02_anatomy_sensors")
