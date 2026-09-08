"""
Thermography on the real face: project the 3D Pennes skin temperature onto the
Lee Perry-Smith head scan so `blender_physics.py --shot thermography` can render
"image 2" (what an IR camera would see) on the same head as the hero shots.

Steps
  1. load sim/assets/head/LeePerrySmith.glb (trimesh, glTF Y-up) and apply the
     SAME world pose blender_hero.py uses:
        Blender glTF import   (x, y, z)_glb -> (x, -z, y)      [Y-up -> Z-up]
        rotate 180 deg about Z, scale 42.4 mm/unit, translate (0, -85, -79) mm
     Verified by printing the nose tip (max y, x ~ 0, y ~ +25 mm) and the
     corneal plane at the pupils (y ~ 0 at x = +-31.5, z = 0).
  2. midpoint-subdivide the scan twice (9.3k -> ~148k vertices) so the vertex
     colours can carry the temperature gradients around the eye.
  3. for every scan vertex take the temperature of the nearest node of the
     physics skin surface (sim/out/fields/skin_T.ply, left half of the face) with
     a cKDTree on |x| (mirror symmetry). Beyond FADE_START mm from the modelled
     patch the colour blends to T_FAR (forehead/chin/neck are outside the
     periorbital mesh: that part of the picture is NOT a solver output and is
     flagged in the 'd_mm' vertex property).
  4. write sim/out/fields/face_T.ply: vertex colour = inferno over
     T_MIN..T_MAX (same range as skin_T.ply), plus raw 'T' and 'd_mm'.
     Also writes sim/out/fields/colorbar_inferno.png (horizontal gradient) for
     the render's colour bar.

Run (nrdi-env):  python sim/render/thermography_face.py
"""
import json, pathlib, sys
import numpy as np
import trimesh
from scipy.spatial import cKDTree
from matplotlib import colormaps as mpl_cm
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = pathlib.Path(__file__).resolve().parents[2]
ASSETS = ROOT / "sim" / "assets"; FDIR = ROOT / "sim" / "out" / "fields"
HEAD_SCALE, HEAD_OFFSET = 42.4, np.array([0.0, -85.0, -79.0])      # == blender_hero.py
T_MIN, T_MAX = 32.5, 35.8                                          # == fields_export.py skin range
FADE_START, FADE_LEN, T_FAR = 6.0, 14.0, 33.2                      # mm, mm, degC (outside the modelled patch)
SUBDIV = 2
INFERNO = mpl_cm["inferno"]


def read_ply_ascii(path):
    """Minimal reader for the ASCII PLY written by fields_export.py (x y z r g b T)."""
    with open(path, "r", encoding="ascii") as f:
        n = 0; props = []
        while True:
            line = f.readline().strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property") and n and "list" not in line:
                props.append(line.split()[-1])
            elif line == "end_header":
                break
        data = np.loadtxt(f, max_rows=n)
    return data[:, :3], data[:, props.index("T")]


def write_ply(path, verts, faces, rgb, T, d):
    with open(path, "w", encoding="ascii") as f:
        f.write("ply\nformat ascii 1.0\ncomment NRDI thermography: Pennes skin T projected on the LeePerrySmith scan (CC BY 3.0)\n")
        f.write(f"element vertex {len(verts)}\nproperty float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nproperty float T\nproperty float d_mm\n")
        f.write(f"element face {len(faces)}\nproperty list uchar int vertex_indices\nend_header\n")
        for i, v in enumerate(verts):
            f.write(f"{v[0]:.3f} {v[1]:.3f} {v[2]:.3f} {rgb[i,0]} {rgb[i,1]} {rgb[i,2]} {T[i]:.3f} {d[i]:.2f}\n")
        for t in faces:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def head_in_project_frame():
    scene = trimesh.load(str(ASSETS / "head" / "LeePerrySmith.glb"))
    m = scene.geometry["LeePerrySmith"] if isinstance(scene, trimesh.Scene) else scene
    v = np.asarray(m.vertices, dtype=np.float64)
    b = np.c_[v[:, 0], -v[:, 2], v[:, 1]]                # Blender glTF import (Y-up -> Z-up)
    r = np.c_[-b[:, 0], -b[:, 1], b[:, 2]]               # rotation 180 deg about Z
    w = r * HEAD_SCALE + HEAD_OFFSET                     # mm, project frame
    return trimesh.Trimesh(w, np.asarray(m.faces), process=False)


def colorbar_png(path, width=512, height=48):
    grad = np.linspace(0, 1, width)[None, :].repeat(height, axis=0)
    fig = plt.figure(figsize=(width / 100, height / 100), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1]); ax.imshow(grad, cmap="inferno", aspect="auto"); ax.set_axis_off()
    fig.savefig(path, dpi=100); plt.close(fig)


def main():
    head = head_in_project_frame()
    w = head.vertices
    tip = w[w[:, 1].argmax()]
    sel = (np.abs(np.abs(w[:, 0]) - 31.5) < 3) & (np.abs(w[:, 2]) < 3)
    print(f"head pose check: nose tip x={tip[0]:.1f} y={tip[1]:.1f} z={tip[2]:.1f} mm; "
          f"corneal plane y at the pupils = {w[sel][:, 1].max():.1f} mm (expect ~0); "
          f"z extent {w[:, 2].min():.0f}..{w[:, 2].max():.0f}")
    assert 15 < tip[1] < 30 and abs(tip[0]) < 8, "head pose does not match blender_hero.py"
    for _ in range(SUBDIV):
        head = head.subdivide()
    print(f"scan subdivided x{SUBDIV}: {len(head.vertices):,} vertices, {len(head.faces):,} faces")

    sv, sT = read_ply_ascii(FDIR / "skin_T.ply")
    tree = cKDTree(sv)
    q = head.vertices.copy(); q[:, 0] = np.abs(q[:, 0])   # mirror: the physics mesh is the left half
    d, idx = tree.query(q, k=1)
    T = sT[idx]
    wfar = np.clip((d - FADE_START) / FADE_LEN, 0.0, 1.0)
    T = (1 - wfar) * T + wfar * T_FAR
    rgb = (np.asarray(INFERNO(np.clip((T - T_MIN) / (T_MAX - T_MIN), 0, 1)))[:, :3] * 255).astype(np.uint8)
    inside = d <= FADE_START
    print(f"projection: {inside.mean()*100:.0f}% of scan vertices within {FADE_START} mm of the modelled skin; "
          f"T on-patch {T[inside].min():.2f}..{T[inside].max():.2f} degC; far fade -> {T_FAR} degC")
    out = FDIR / "face_T.ply"
    write_ply(out, head.vertices, head.faces, rgb, T, d)
    colorbar_png(FDIR / "colorbar_inferno.png")
    meta = {"T_min": T_MIN, "T_max": T_MAX, "cmap": "inferno", "fade_start_mm": FADE_START, "fade_len_mm": FADE_LEN,
            "T_far": T_FAR, "n_vertices": int(len(head.vertices)), "source": "sim/out/fields/skin_T.ply (thermal3d full mesh)",
            "head_pose": {"scale": HEAD_SCALE, "rot_z_deg": 180, "offset_mm": HEAD_OFFSET.tolist()}}
    (FDIR / "face_T.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB) and colorbar_inferno.png")


if __name__ == "__main__":
    main()
