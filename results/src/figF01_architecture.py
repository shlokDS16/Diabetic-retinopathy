"""
F1 - System architecture: periorbital wearable -> physics digital twin (design loop) -> multimodal AI -> joint DR/DN outputs.
Pure matplotlib (vector PDF, editable text). No data.
Run: python results/src/figF01_architecture.py
"""
import pathlib, sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import pubstyle as ps
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ps.apply()
fig = plt.figure(figsize=(ps.DOUBLE_MM * ps.MM, 92 * ps.MM), layout="none")
ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 180); ax.set_ylim(0, 92); ax.axis("off")
INK = ps.INK


def box(x, y, w, h, title, lines=(), fc="#ffffff", ec=INK, lw=0.6, tfs=6.2, lfs=5.2, tcol=INK, lcol=INK, dashed=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=1.6", fc=fc, ec=ec, lw=lw, ls=(0, (3, 2)) if dashed else "-"))
    ty = y + h - 1.6
    ax.text(x + w / 2, ty, title, ha="center", va="top", fontsize=tfs, fontweight="bold", color=tcol)
    for i, ln in enumerate(lines):
        ax.text(x + w / 2, ty - 3.1 - 2.6 * i, ln, ha="center", va="top", fontsize=lfs, color=lcol)
    return (x, y, w, h)


def arrow(p0, p1, color=INK, lw=0.7, style="-|>", rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(p0, p1, arrowstyle=style, mutation_scale=6, color=color, lw=lw, connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0, ls=ls))


def line(pts, color=INK, lw=0.7):
    ax.plot([p[0] for p in pts], [p[1] for p in pts], color=color, lw=lw, solid_capstyle="round", solid_joinstyle="round", zorder=1)


def group(x, y, w, h, label, color):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=2.5", fc=color, ec="none", alpha=0.10, zorder=0))
    ax.text(x + 2, y + h - 1.5, label, ha="left", va="top", fontsize=6.5, fontweight="bold", color=color)


# ---------------- groups
group(2, 30, 46, 60, "periorbital wearable (patent pending)", ps.C["electrical"])
group(54, 52, 58, 38, "physics digital twin (design loop)", ps.C["thermal"])
group(54, 30, 58, 20, "signal features", ps.C["ref"])
group(116, 30, 62, 60, "multimodal AI", ps.C["ehr"])
group(2, 2, 176, 25, "evidence at submission (open data; credentialed mBRSET / BRSET pending)", ps.C["good"])

# ---------------- wearable sensors + bus to features
sx, sw_, sh = 5, 38, 10
s1 = box(sx, 72, sw_, sh, "thermopile (MLX90614)", ("ocular-surface temperature, 2 fields of view",), fc="#eef4fa")
s2 = box(sx, 60, sw_, sh, "camera + 940-nm LED", ("pupillography; IEC 62471-limited irradiance",), fc="#eef4fa")
s3 = box(sx, 48, sw_, sh, "4 Ag/AgCl dry electrodes per eye", ("local tetrapolar periorbital bioimpedance",), fc="#eef4fa")
s4 = box(sx, 36, sw_, sh, "temple PPG (MAX30102)", ("660 / 880 nm, 3 mm spacing",), fc="#eef4fa")
ax.text(sx + sw_ / 2, 33.0, "ESP32-S3 · I²C multiplexer · 250 mAh LiPo · BLE", ha="center", va="center", fontsize=5, color=ps.MUTED)
bus_x = 46.5
for s in (s1, s2, s3, s4):
    line([(s[0] + s[2], s[1] + s[3] / 2), (bus_x, s[1] + s[3] / 2)], color=ps.C["ref"], lw=0.6)
line([(bus_x, s1[1] + s1[3] / 2), (bus_x, 40)], color=ps.C["ref"], lw=0.6)
arrow((bus_x, 40), (57, 40), color=ps.C["ref"], lw=0.6)

# ---------------- digital twin
tx, tw, th = 57, 25, 8.5
t1 = box(tx, 76, tw, th, "3-D Pennes bioheat", ("122k-node FE, 14 tissues",), fc="#fdf1ea", lfs=5)
t2 = box(tx, 66, tw, th, "Monte Carlo optics", ("PPG banana, LED absorption",), fc="#fdf1ea", lfs=5)
t3 = box(tx + 27, 76, tw, th, "complex-conductivity FE", ("lead fields, Geselowitz",), fc="#fdf1ea", lfs=5)
t4 = box(tx + 27, 66, tw, th, "pupil model (DDE)", ("autonomic gain, camera",), fc="#fdf1ea", lfs=5)
t5 = box(tx, 54, 52, 10.5, "verification & uncertainty", ("MMS order 1.8 · GCI < 2 % · Sobol, 8 parameters, 640 solves", "ASME V&V 40 credibility framing"), fc="#ffffff", lfs=5)
# design loop across the top: CAD/poses in, verdicts out
arrow((47, 84.5), (55, 84.5), color=ps.C["electrical"], lw=0.7, ls=(0, (2, 1.5)))
ax.text(51, 85.6, "CAD, poses", ha="center", va="bottom", fontsize=4.4, color=ps.MUTED)
arrow((55, 81.5), (47, 81.5), color=ps.C["thermal"], lw=0.9)
ax.text(51, 80.4, "8 verdicts", ha="center", va="top", fontsize=4.6, color=ps.C["thermal"], fontweight="bold")

# ---------------- signal features
f1 = box(57, 33, 16, 12, "thermal", ("ambient-\ncompensated ΔT",), fc="#f7f7f7", tfs=5.6, lfs=4.8)
f2 = box(75, 33, 16, 12, "pupil", ("amplitude, latency,\nvelocities",), fc="#f7f7f7", tfs=5.6, lfs=4.8)
f3 = box(93, 33, 17, 12, "|Z|, phase; PPG", ("edema index,\nHRV, morphology",), fc="#f7f7f7", tfs=5.6, lfs=4.8)
arrow((110, 39), (120, 39), color=ps.C["ref"], lw=0.6)

# ---------------- AI
ax_, aw, ah = 119, 26, 9
a1 = box(ax_, 76, aw, ah, "fundus DR grader", ("ConvNeXt-T, ordinal 0–4",), fc="#eeecf6", lfs=5)
a2 = box(ax_ + 30, 76, aw, ah, "macular-oedema head", ("ordinal 0–2",), fc="#eeecf6", lfs=5)
a3 = box(ax_, 65, aw, ah, "systemic EHR model", ("boosted trees, n = 77,724",), fc="#eeecf6", lfs=5)
a4 = box(ax_ + 30, 65, aw, ah, "physiological branch", ("HRV, PPG, wearable features",), fc="#eeecf6", lfs=5)
fu = box(ax_ + 8, 50, 40, 9.5, "ordinal late fusion", ("calibrated, threshold-optimised (QWK)",), fc=ps.C["ehr"], ec=ps.C["ehr"], tcol="white", lcol="white", lfs=5)
by = 62.5; c = ps.C["ehr"]
line([(a1[0] - 1.5, a1[1] + a1[3] / 2), (a1[0] - 1.5, by)], color=c, lw=0.6)          # left branch down the outside
line([(a1[0], a1[1] + a1[3] / 2), (a1[0] - 1.5, a1[1] + a1[3] / 2)], color=c, lw=0.6)
line([(a2[0] + a2[2] + 1.5, a2[1] + a2[3] / 2), (a2[0] + a2[2] + 1.5, by)], color=c, lw=0.6)
line([(a2[0] + a2[2], a2[1] + a2[3] / 2), (a2[0] + a2[2] + 1.5, a2[1] + a2[3] / 2)], color=c, lw=0.6)
line([(a1[0] - 1.5, by), (a2[0] + a2[2] + 1.5, by)], color=c, lw=0.6)                      # bus
for a in (a3, a4):
    line([(a[0] + a[2] / 2, a[1]), (a[0] + a[2] / 2, by)], color=c, lw=0.6)
arrow((fu[0] + fu[2] / 2, by), (fu[0] + fu[2] / 2, fu[1] + fu[3]), color=c, lw=0.8)
o = box(ax_ + 3, 34, 50, 11, "joint outputs", ("DR grade 0–4 · referable DR · DME · DN risk",), fc="#ffffff", ec=ps.C["good"], lw=0.9, lfs=5.2)
arrow((fu[0] + fu[2] / 2, fu[1]), (o[0] + o[2] / 2, o[1] + o[3]), color=ps.C["good"], lw=0.9)

# ---------------- evidence ribbon
items = [("DeepDRiD", "1,589 images, 399 patients\npatient-level 5-fold", ps.C["fundus"]),
         ("IDRiD", "516 images, external test\n+ DME grades", ps.C["external"]),
         ("APTOS 2019", "3,662 images, India\nsecond external test", ps.C["external2"]),
         ("Istanbul EHR", "77,724 diabetic patients\nDR + DN outcomes", ps.C["ehr"]),
         ("PhysioNet ECG", "94 subjects, HRV\nDM vs control; DR null", ps.C["hrv"]),
         ("PPG-BP", "219 subjects\nnull + HTN control", ps.C["ppg"]),
         ("mBRSET / BRSET", "pending credentialing\n(smartphone fundus + EHR)", ps.C["ref"])]
n = len(items); w = 176 / n - 1.6
for i, (name, desc, col) in enumerate(items):
    x = 3 + i * (w + 1.6)
    box(x, 4, w, 17, name, desc.split("\n"), fc="#ffffff", ec=col, lw=0.8, tfs=5.6, lfs=4.8, tcol=col, dashed=name.startswith("mBRSET"))

ps.save(fig, "figF01_architecture")
