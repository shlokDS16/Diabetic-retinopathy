"""
VTK-free viewer for the finite-element results: the same content as view_fields.py, rendered in the browser with Plotly.
Written because Windows Smart App Control blocks the unsigned VTK binaries that PyVista needs (Code Integrity event
3077, 2026-09-09); this script needs only meshio, numpy and plotly, all pure Python or signed.

Shows: the periorbital mesh surface coloured by steady temperature (inferno), sagittal cut planes through the globe
that expose the interior field, the same surface coloured by tissue region, and the sensor poses from sensor_layout.json.
Buttons at the top switch views; drag to orbit, wheel to zoom.

Run (main Python):  python sim/render/view_fields_web.py [--coarse] [--no-open]
Output: sim/render/out/view_fields.html (self-contained; double-click opens it in the browser)
"""
import argparse, json, pathlib, webbrowser
import numpy as np, meshio
import plotly.graph_objects as go

ROOT = pathlib.Path(__file__).resolve().parents[2]; OUT = ROOT / "sim" / "out"
REGIONS = {1: "angular artery", 2: "temporal artery", 3: "cornea", 4: "sclera", 5: "lens", 6: "aqueous", 7: "vitreous", 8: "eyelid", 9: "orbital fat",
           10: "skin", 11: "subcutaneous fat", 12: "muscle", 13: "bone", 14: "brain / core"}
REGION_COLOR = {1: "#c0392b", 2: "#e74c3c", 3: "#5dade2", 4: "#d5dbdb", 5: "#f5cba7", 6: "#aed6f1", 7: "#2874a6", 8: "#e59866", 9: "#f4d03f",
                10: "#dc7633", 11: "#fdebd0", 12: "#a04000", 13: "#bfc9ca", 14: "#af7ac5"}


def boundary_faces(tets):
    """Triangles that belong to exactly one tetrahedron, with the owning tet index."""
    f = np.concatenate([tets[:, [0, 1, 2]], tets[:, [0, 1, 3]], tets[:, [0, 2, 3]], tets[:, [1, 2, 3]]])
    owner = np.tile(np.arange(len(tets)), 4)
    fs = np.sort(f, axis=1)
    _, idx, cnt = np.unique(fs, axis=0, return_index=True, return_counts=True)
    keep = idx[cnt == 1]
    f, owner = f[keep], owner[keep]
    return f, owner


def orient_outward(pts, tets, f, owner):
    """Flip boundary triangles so their normals point away from the owning tetrahedron (consistent shading)."""
    c_tet = pts[tets[owner]].mean(1); c_f = pts[f].mean(1)
    n = np.cross(pts[f[:, 1]] - pts[f[:, 0]], pts[f[:, 2]] - pts[f[:, 0]])
    flip = np.einsum("ij,ij->i", n, c_f - c_tet) < 0
    f = f.copy(); f[flip] = f[flip][:, [0, 2, 1]]
    return f


def mesh_trace(pts, tri, intensity=None, colors=None, name="", cmin=None, cmax=None, visible=True, showscale=False):
    kw = dict(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], i=tri[:, 0], j=tri[:, 1], k=tri[:, 2], name=name, visible=visible, flatshading=False,
              lighting=dict(ambient=0.55, diffuse=0.7, specular=0.15, roughness=0.9), lightposition=dict(x=200, y=-300, z=300))
    if intensity is not None:
        kw.update(intensity=intensity, colorscale="Inferno", cmin=cmin, cmax=cmax, showscale=showscale, colorbar=dict(title="T (°C)", len=0.6))
    else:
        kw.update(facecolor=colors)
    return go.Mesh3d(**kw)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--coarse", action="store_true"); ap.add_argument("--no-open", action="store_true"); a = ap.parse_args()
    vtu = OUT / ("thermal3d_coarse.vtu" if a.coarse else "thermal3d_full.vtu")
    m = meshio.read(vtu); pts = m.points.astype(float); tets = m.cells_dict["tetra"]; T = m.point_data["T"].astype(float)
    region = m.cell_data["region"][0].astype(int) if isinstance(m.cell_data["region"], list) else m.cell_data["region"].astype(int)
    layout = json.loads((OUT / "sensor_layout.json").read_text(encoding="utf-8"))
    cmin, cmax = float(np.percentile(T, 1)), float(np.percentile(T, 99.5))
    cent = pts[tets].mean(1)

    traces, names = [], []
    tri, own = boundary_faces(tets); tri = orient_outward(pts, tets, tri, own)
    traces.append(mesh_trace(pts, tri, intensity=T, name="temperature, full head", cmin=cmin, cmax=cmax, showscale=True)); names.append("Temperature field")
    for x0 in (26.0, 31.5, 37.0):  # sagittal cuts through the left globe (globe centre x = 31.5 mm)
        keep = cent[:, 0] >= x0
        tri_c, own_c = boundary_faces(tets[keep]); tri_c = orient_outward(pts, tets[keep], tri_c, own_c)
        traces.append(mesh_trace(pts, tri_c, intensity=T, name=f"cut at x = {x0:.0f} mm", cmin=cmin, cmax=cmax, showscale=True, visible=False)); names.append(f"Sagittal cut x = {x0:.0f} mm")
    cols = np.array([REGION_COLOR[int(r)] for r in region[own]])
    traces.append(mesh_trace(pts, tri, colors=cols, name="tissue regions", visible=False)); names.append("Tissue regions")
    keep = cent[:, 0] >= 31.5; tri_c, own_c = boundary_faces(tets[keep]); tri_c = orient_outward(pts, tets[keep], tri_c, own_c); cols_c = np.array([REGION_COLOR[int(r)] for r in region[np.where(keep)[0][own_c]]])
    traces.append(mesh_trace(pts, tri_c, colors=cols_c, name="tissue regions, cut", visible=False)); names.append("Tissue regions, cut through the globe")
    n_mesh = len(traces)
    # sensors (always shown)
    sx, sy, sz, txt = [], [], [], []
    for s in layout["sensors"]:
        p = s["pos"]
        if p[0] >= -1:  # left half only
            sx.append(p[0]); sy.append(p[1]); sz.append(p[2]); txt.append(f"{s['name']} ({s['kind']}, {s.get('part', '')})")
    traces.append(go.Scatter3d(x=sx, y=sy, z=sz, mode="markers+text", marker=dict(size=5, color="#00e5ff", line=dict(color="black", width=1)), text=[t.split(" ")[0] for t in txt],
                               hovertext=txt, textfont=dict(size=9, color="#00e5ff"), name="sensors", textposition="top center"))
    buttons = []
    for k, nm in enumerate(names):
        vis = [i == k for i in range(n_mesh)] + [True]
        buttons.append(dict(label=nm, method="update", args=[{"visible": vis}]))
    fig = go.Figure(data=traces)
    fig.update_layout(template="plotly_dark", title=dict(text=f"NRDI periorbital finite-element model: steady temperature (Pennes), {len(pts):,} nodes, {len(tets):,} tetrahedra, ambient 25 °C", x=0.02),
                      scene=dict(aspectmode="data", xaxis_title="x lateral (mm)", yaxis_title="y anterior (mm)", zaxis_title="z up (mm)", camera=dict(eye=dict(x=1.6, y=-1.4, z=0.7))),
                      updatemenus=[dict(type="buttons", direction="right", x=0.02, y=0.985, xanchor="left", yanchor="top", buttons=buttons, bgcolor="#222", font=dict(color="white"))],
                      margin=dict(l=0, r=0, t=60, b=0), height=900)
    out = ROOT / "sim" / "render" / "out" / "view_fields.html"
    fig.write_html(out, include_plotlyjs=True, full_html=True)
    print("wrote", out, f"({out.stat().st_size/1e6:.1f} MB); {len(tri):,} surface triangles")
    if not a.no_open: webbrowser.open(out.as_uri())


if __name__ == "__main__":
    main()
