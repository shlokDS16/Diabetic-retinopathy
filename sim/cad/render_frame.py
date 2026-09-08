"""Offscreen multi-view render of the frame, for inspection and for figures."""
import pathlib, sys, json
import pyvista as pv

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "out"
FIG = HERE.parent / "figures"
FIG.mkdir(parents=True, exist_ok=True)

pv.OFF_SCREEN = True

frame = pv.read(OUT / "frame.stl")
fov = pv.read(OUT / "fov.stl") if (OUT / "fov.stl").exists() else None
layout = json.loads((OUT / "sensor_layout.json").read_text(encoding="utf-8"))

BODY = "#c9cdd4"
ACCENT = "#b04a20"

views = [
    ("iso",   dict(position=(220, 200, 160), viewup=(0, 0, 1)), "isometric"),
    ("front", dict(position=(0, 340, 0),     viewup=(0, 0, 1)), "front (wearer's view out)"),
    ("top",   dict(position=(0, 0, 340),     viewup=(0, 1, 0)), "top"),
    ("side",  dict(position=(340, 0, 0),     viewup=(0, 0, 1)), "left side"),
]

pl = pv.Plotter(shape=(2, 2), window_size=(1700, 1300), border=False)
for i, (key, cam, title) in enumerate(views):
    pl.subplot(i // 2, i % 2)
    pl.add_mesh(frame, color=BODY, smooth_shading=True,
                specular=0.35, specular_power=18, ambient=0.22)
    if fov is not None:
        pl.add_mesh(fov, color=ACCENT, opacity=0.20, smooth_shading=True)
    pl.add_text(title, font_size=11, color="#333333")
    pl.camera.position = cam["position"]
    pl.camera.focal_point = (0, -55, 0)
    pl.camera.up = cam["viewup"]
    pl.camera.zoom(1.25)

pl.set_background("white")
pl.screenshot(str(FIG / "frame_views.png"))
pl.close()
print("wrote", FIG / "frame_views.png")

# --- close-up on the periorbital sensor cluster -----------------------------
pl = pv.Plotter(window_size=(1500, 1050), border=False)
pl.add_mesh(frame, color=BODY, smooth_shading=True, specular=0.35, ambient=0.25)
if fov is not None:
    pl.add_mesh(fov, color=ACCENT, opacity=0.26, smooth_shading=True)

# mark the pupil positions and the medial canthus targets
ipd = layout["anthropometry"]["ipd_mm"]
mc = layout["anthropometry"]["medial_canthus_offset_mm"]
for sgn in (+1, -1):
    pl.add_mesh(pv.Sphere(radius=2.2, center=(sgn * ipd / 2, 0, 0)),
                color="#2f6b4f")
    pl.add_mesh(pv.Sphere(radius=1.8, center=(sgn * (ipd / 2 - mc), 2, -2)),
                color="#a02d20")

pl.camera.position = (95, 165, 70)
pl.camera.focal_point = (0, 5, 0)
pl.camera.up = (0, 0, 1)
pl.camera.zoom(2.4)
pl.set_background("white")
pl.screenshot(str(FIG / "frame_closeup.png"))
pl.close()
print("wrote", FIG / "frame_closeup.png")
print("  green spheres = pupil centres, red = medial canthus targets")
