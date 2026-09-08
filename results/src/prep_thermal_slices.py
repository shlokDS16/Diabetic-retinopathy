"""
Extract planar slices of the 3-D Pennes solution and a clipped tissue-region render for figures F2/F3.
Runs in the physics environment (pyvista):  C:/Users/Shlok/nrdi-env/Scripts/python.exe results/src/prep_thermal_slices.py
Outputs: results/out/thermal_slices.npz  (per slice: points (n,2), tris (m,3), T (n,), region (m,))
         results/figures/parts/mesh_clip.png  (region-coloured clip of periorbital.msh through the left globe)
"""
import pathlib, json, numpy as np, pyvista as pv
ROOT = pathlib.Path(__file__).resolve().parents[2]
m = pv.read(ROOT / "sim/out/thermal3d_full.vtu")
lay = json.loads((ROOT / "sim/out/sensor_layout.json").read_text(encoding="utf-8"))
ipd = lay["anthropometry"]["ipd_mm"]
out = {}
for name, normal, origin in (("sagittal_eye", "x", (ipd / 2, 0, 0)), ("axial_pupil", "z", (0, 0, 0)), ("coronal_lid", "y", (0, 4.0, 0))):
    s = m.slice(normal=normal, origin=origin).triangulate()
    faces = s.faces.reshape(-1, 4)[:, 1:]
    pts = s.points
    keep = [i for i in range(3) if "xyz"[i] != normal]
    out[f"{name}_pts"] = pts[:, keep].astype(np.float32); out[f"{name}_tris"] = faces.astype(np.int32)
    out[f"{name}_T"] = np.asarray(s.point_data["T"], np.float32); out[f"{name}_region"] = np.asarray(s.cell_data["region"], np.int16)
    print(name, s.n_points, s.n_cells, "T", float(out[f"{name}_T"].min()), float(out[f"{name}_T"].max()))
np.savez_compressed(ROOT / "results/out/thermal_slices.npz", **out)
# region-coloured clip render for F2c
pv.OFF_SCREEN = True
try:
    clip = m.clip(normal="x", origin=(ipd / 2, 0, 0), invert=False)
    pl = pv.Plotter(off_screen=True, window_size=(1600, 1400))
    pl.set_background("white")
    pl.add_mesh(clip, scalars="region", cmap="tab20", clim=(1, 14), show_scalar_bar=False, smooth_shading=False)
    pl.camera_position = "yz"; pl.camera.azimuth = 35; pl.camera.elevation = 18; pl.camera.zoom(1.9)
    pl.camera.focal_point = (ipd / 2, -5, 0)
    (ROOT / "results/figures/parts").mkdir(exist_ok=True)
    pl.screenshot(str(ROOT / "results/figures/parts/mesh_clip.png"))
    print("wrote mesh_clip.png")
except Exception as e:
    print("clip render failed:", e)
