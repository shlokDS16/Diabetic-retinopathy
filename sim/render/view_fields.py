"""
Interactive scientific viewer for the physics results (PyVista, VTK) — the
"this is a real finite-element model" demo, independent of Blender or the web.

Shows, in one window with linked cameras (press keys to toggle):
  - the 3D periorbital mesh coloured by tissue region (wireframe edges on request)
  - the steady temperature field (inferno) with a live clip plane through the globe
  - the glasses assembly (STL parts) in the same coordinate frame
  - sensor poses (spheres) and the electrode contacts (discs) from sensor_layout.json

Keys inside the window:  q quit · w wireframe · s surface · r reset camera
Widgets: drag the clip-plane arrow to cut through the head; the field updates live.

Run (nrdi-env):  python sim/render/view_fields.py [--coarse] [--screenshot out.png]
"""
import sys, json, pathlib, argparse
import numpy as np
import pyvista as pv

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "sim" / "out"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--coarse", action="store_true")
    ap.add_argument("--screenshot", default=None, help="render off-screen to this PNG instead of opening a window")
    a = ap.parse_args()
    tag = "coarse" if a.coarse else "full"
    vtu = OUT / f"thermal3d_{tag}.vtu"
    if not vtu.exists():
        vtu = OUT / "thermal3d_coarse.vtu"
    grid = pv.read(str(vtu))                      # points in mm, point_data T, cell_data region
    layout = json.loads((OUT / "sensor_layout.json").read_text(encoding="utf-8"))
    parts = json.loads((OUT / "parts.json").read_text(encoding="utf-8"))["parts"]

    pv.set_plot_theme("dark")
    off = a.screenshot is not None
    pl = pv.Plotter(shape=(1, 2), window_size=(1800, 900), off_screen=off, title="NRDI periorbital model — FEM results")

    # --- left: anatomy by region -------------------------------------------
    pl.subplot(0, 0)
    pl.add_text("Anatomy: 14 tissue regions, %s tets" % f"{grid.n_cells:,}", font_size=11)
    regions = grid.cell_data["region"] if "region" in grid.cell_data else None
    surf = grid.extract_surface()
    pl.add_mesh(surf, scalars="region" if regions is not None else None, cmap="tab20", show_scalar_bar=False,
                opacity=0.35, smooth_shading=True)
    # clip half the head to reveal the orbit
    clipped = grid.clip(normal=(1, 0, 0), origin=(31.5, 0, 0), invert=True)
    pl.add_mesh(clipped, scalars="region", cmap="tab20", show_edges=False, show_scalar_bar=False)
    _add_glasses(pl, parts, opacity=0.9)
    _add_sensors(pl, layout)
    pl.add_axes(xlabel="x (lateral)", ylabel="y (anterior)", zlabel="z (up)")

    # --- right: temperature with live clip plane -----------------------------
    pl.subplot(0, 1)
    pl.add_text("Steady temperature, Pennes bioheat (inferno, 31–38 °C)", font_size=11)
    _add_glasses(pl, parts, opacity=0.25)
    pl.add_mesh(surf, scalars="T", cmap="inferno", clim=(31, 38), opacity=0.25, show_scalar_bar=False, smooth_shading=True)
    pl.add_mesh_clip_plane(grid, normal=(1, 0, 0), origin=(31.5, 0, 0), invert=True, scalars="T", cmap="inferno",
                           clim=(31, 38), scalar_bar_args={"title": "T (°C)", "fmt": "%.1f"}, assign_to_axis=None)
    _add_sensors(pl, layout)
    pl.add_axes()
    pl.link_views()
    pl.camera_position = [(240, 190, 120), (30, -20, 0), (0, 0, 1)]

    if off:
        pl.screenshot(a.screenshot)
        print("wrote", a.screenshot)
    else:
        pl.show()


def _add_glasses(pl, parts, opacity=1.0):
    for p in parts:
        stl = OUT / "parts_stl_web" / f"{p['label']}.stl"
        if not stl.exists():
            continue
        col = p["color"]
        pl.add_mesh(pv.read(str(stl)), color=tuple(col), opacity=opacity if p["category"] != "lens" else 0.25 * opacity,
                    smooth_shading=True, specular=0.4)


def _add_sensors(pl, layout):
    for s in layout["sensors"]:
        pos = np.array(s["pos"], float)
        if s["kind"] == "electrode":
            disc = pv.Disc(center=pos, inner=0, outer=3.0, normal=s["axis"], c_res=32)
            pl.add_mesh(disc, color="#3fb8ae", opacity=0.9)
        else:
            pl.add_mesh(pv.Sphere(radius=1.6, center=pos), color={"thermopile": "#f0a35a", "camera": "#8b98f0",
                        "ir_led": "#e0679c", "ppg": "#e0679c"}.get(s["kind"], "#cccccc"))
        if s.get("fov_deg", 0) > 0 and s["kind"] == "thermopile":
            ax = np.array(s["axis"], float); ax /= np.linalg.norm(ax)
            L = 14.0
            cone = pv.Cone(center=pos + ax * L / 2, direction=-ax, height=L, radius=L * np.tan(np.radians(2.5)), resolution=24)
            pl.add_mesh(cone, color="#f0a35a", opacity=0.5)


if __name__ == "__main__":
    main()
