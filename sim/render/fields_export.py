"""
Bridge physics fields into renderable assets (one palette for every surface).

Outputs in sim/out/fields/
  skin_T.ply          exposed skin + cornea + conjunctiva surface, vertex colour =
                      temperature (cmcrameri 'lajolla', T_MIN..T_MAX), plus raw
                      'T' vertex property
  slice_T_z0.png      temperature on the z = 0 (horizontal, through the pupil) plane,
                      alpha = 0 outside tissue, with slice_T_z0.json (extent, range)
  slice_T_x31.png     temperature on the x = 31.5 (sagittal, through the globe) plane
  banana_880.npy      MCX measurement-density banana, normalised 0..1 (for VDB in Blender)
  banana_880.json     grid origin/res so Blender can place the volume
  leadfield.ply       skin surface coloured by the bioimpedance lead-field density
                      projected from the nearest tetra (cmcrameri 'batlow')
  palette.json        colormap names + ranges shared by matplotlib / Blender / Three.js

Run (nrdi-env):  python sim/render/fields_export.py
"""
import sys, json, pathlib
import numpy as np
from skfem import MeshTet, Basis, ElementTetP1
import cmcrameri.cm as cmc
from matplotlib import colormaps as mpl_cm
INFERNO = mpl_cm['inferno']
import imageio.v3 as iio

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "sim" / "forward_models"))
import periorbital_mesh as PM                                  # noqa: E402

OUT = ROOT / "sim" / "out"
FDIR = OUT / "fields"; FDIR.mkdir(exist_ok=True)
T_MIN, T_MAX = 32.5, 35.8          # skin surface range: shows the warm canthus / cool lids
PALETTE = {"thermal": {"cmap": "inferno", "vmin": T_MIN, "vmax": T_MAX, "unit": "degC"},
           "optical": {"cmap": "magma", "scale": "log", "unit": "a.u."},
           "electrical": {"cmap": "batlow", "scale": "log", "unit": "a.u."},
           "accent": "#b04a20"}


def write_ply(path, verts, faces, rgb, scalar=None, sname="T"):
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\ncomment NRDI field export\n")
        f.write(f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        if scalar is not None:
            f.write(f"property float {sname}\n")
        f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for i, v in enumerate(verts):
            line = f"{v[0]:.4f} {v[1]:.4f} {v[2]:.4f} {rgb[i,0]} {rgb[i,1]} {rgb[i,2]}"
            if scalar is not None:
                line += f" {scalar[i]:.4f}"
            f.write(line + "\n")
        for t in faces:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def surface_ply(m, values, names, path, cmap, vmin, vmax, sname):
    facets = np.concatenate([m.boundaries[n] for n in names if n in m.boundaries])
    tri = m.facets[:, facets].T                                  # (nf, 3) node ids
    used, inv = np.unique(tri, return_inverse=True)
    verts = m.p[:, used].T / 1e-3                                # mm
    faces = inv.reshape(-1, 3)
    val = values[used]
    rgb = (np.asarray(cmap((val - vmin) / (vmax - vmin)))[:, :3] * 255).astype(np.uint8)
    write_ply(path, verts, faces, rgb, val, sname)
    return len(verts), len(faces)


def slice_png(m, basis, T, plane, value, res, path, cmap, window=None):
    """Sample T on an axis-aligned plane; transparent outside the mesh.
    window: optional ((u0,u1),(v0,v1)) in mm to restrict the slice extent."""
    lo, hi = m.p.min(axis=1) / 1e-3, m.p.max(axis=1) / 1e-3
    if window is not None:
        ax_ = {"x": 0, "y": 1, "z": 2}[plane]; oth = [i for i in range(3) if i != ax_]
        lo = lo.copy(); hi = hi.copy()
        lo[oth[0]], hi[oth[0]] = window[0]; lo[oth[1]], hi[oth[1]] = window[1]
    ax = {"x": 0, "y": 1, "z": 2}[plane]
    others = [i for i in range(3) if i != ax]
    u = np.arange(lo[others[0]], hi[others[0]], res); v = np.arange(lo[others[1]], hi[others[1]], res)
    U, V = np.meshgrid(u, v, indexing="ij")
    P = np.zeros((U.size, 3)); P[:, others[0]] = U.ravel(); P[:, others[1]] = V.ravel(); P[:, ax] = value
    # inverse-distance interpolation from the 4 nearest nodes (basis.probes searches
    # every element and needs 70 GB here); inside/outside from the analytic domain test
    from scipy.spatial import cKDTree
    d, idx = cKDTree(m.p.T / 1e-3).query(P, k=4)
    wgt = 1.0 / np.maximum(d, 1e-6); vals = (T[idx] * wgt).sum(axis=1) / wgt.sum(axis=1)
    sys.path.insert(0, str(ROOT / "sim" / "forward_models"))
    from voxelize import in_domain
    inside = in_domain(P) & (d[:, 0] < 8.0)
    img = np.zeros((U.shape[0], U.shape[1], 4))
    lo_c, hi_c = 31.0, 38.6                                      # 37 C -> orange, not white                                      # slice range: where the gradient lives
    col = np.asarray(cmap(np.clip((vals - lo_c) / (hi_c - lo_c), 0, 1)))
    img[..., :3] = col[:, :3].reshape(U.shape + (3,))
    img[..., 3] = inside.reshape(U.shape)
    # image rows = v (top = v max), columns = u, so Blender's plane X = u, Y = v
    iio.imwrite(path, np.flipud((img * 255).astype(np.uint8).transpose(1, 0, 2)))
    return {"plane": plane, "value_mm": value, "u_axis": "xyz"[others[0]], "v_axis": "xyz"[others[1]],
            "u_range_mm": [float(u[0]), float(u[-1])], "v_range_mm": [float(v[0]), float(v[-1])],
            "res_mm": res, "T_range": [31.0, 38.6]}


def main():
    tag = "full" if (OUT / "thermal3d_T_full.npy").exists() else "coarse"
    m = MeshTet.load(str(OUT / f"periorbital{'' if tag == 'full' else '_coarse'}.msh")).scaled(1e-3)
    basis = Basis(m, ElementTetP1())
    T = np.load(OUT / f"thermal3d_T_{tag}.npy")
    print(f"T field from the {tag} mesh ({m.p.shape[1]:,} nodes)")
    nv, nf = surface_ply(m, T, ["skin_exposed", "cornea_exposed", "conj_exposed"],
                         FDIR / "skin_T.ply", INFERNO, T_MIN, T_MAX, "T")
    print(f"skin_T.ply: {nv} verts, {nf} faces, T {T.min():.2f}..{T.max():.2f}")
    meta = {"slice_T_z0": slice_png(m, basis, T, "z", 0.0, 0.25, FDIR / "slice_T_z0.png", INFERNO),
            "slice_T_x31": slice_png(m, basis, T, "x", PM.PX, 0.25, FDIR / "slice_T_x31.png", INFERNO,
                                     window=((-50.0, 8.0), (-32.0, 32.0)))}
    print("slices written")
    # optical banana (880 nm) -> normalised volume for Blender VDB
    vm = json.loads((OUT / "voxel_meta.json").read_text(encoding="utf-8"))["rois"]["temple"]
    b = np.load(OUT / "ppg_banana_880.npy").astype(np.float64)
    b = np.clip(np.log10(np.clip(b / b.max(), 1e-9, 1.0)) / 3.0 + 1.0, 0.0, 1.0)   # log scale over 3 decades, 0..1
    np.save(FDIR / "banana_880.npy", b.astype(np.float32))
    (FDIR / "banana_880.json").write_text(json.dumps({"origin_mm": vm["origin_mm"], "res_mm": vm["res_mm"],
                                                      "shape": list(b.shape), "scale": "log10, 3 decades"}), encoding="utf-8")
    # bioimpedance lead field -> skin surface colour (nearest element per surface node)
    lf = OUT / "bioimp_leadfield_coarse.npy"
    if lf.exists():
        m = MeshTet.load(str(OUT / "periorbital_coarse.msh")).scaled(1e-3)
        dens = np.abs(np.load(lf))
        node_val = np.zeros(m.p.shape[1]); cnt = np.zeros(m.p.shape[1])
        for k in range(4):
            np.add.at(node_val, m.t[k], dens); np.add.at(cnt, m.t[k], 1)
        node_val = np.log10(np.clip(node_val / np.maximum(cnt, 1) / dens.max(), 1e-5, 1)) / 5 + 1
        nv, nf = surface_ply(m, node_val, ["skin_exposed", "cornea_exposed", "conj_exposed"], FDIR / "leadfield.ply", cmc.batlow, 0, 1, "S")
        print(f"leadfield.ply: {nv} verts")
    (FDIR / "palette.json").write_text(json.dumps({**PALETTE, "slices": meta}, indent=2), encoding="utf-8")
    print("wrote sim/out/fields/*")


if __name__ == "__main__":
    main()
