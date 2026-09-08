"""
Bake the NRDI digital-twin dashboard: one self-contained HTML file with every
physics result and the device geometry embedded (no backend, works offline and
as a Claude Artifact).

Embeds
  parts geometry   sim/out/parts_stl/*.stl -> base64 float32 positions + uint32 indices
  parts.json       explode vectors, mount points, callouts, colours
  sensor_layout.json
  thermal3d_full.json / thermal3d_coarse.json, optical_results.json, opto_thermal.json,
  bioimpedance3d_coarse.json, pupil_model.json, verification.json,
  surrogate_thermal.json (if present), banana slice (2D, log) from ppg_banana_880.npy,
  slice_T_x31.png (data URI), hero renders (data URI, downscaled)

Run (nrdi-env):  python sim/dashboard/build.py
"""
import json, base64, pathlib, io
import numpy as np
import trimesh
import imageio.v3 as iio

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "sim" / "out"
HERE = pathlib.Path(__file__).resolve().parent


def b64(arr):
    return base64.b64encode(np.ascontiguousarray(arr).tobytes()).decode("ascii")


TOTAL_TRIS = sum(len(trimesh.load(p, force="mesh").faces) for p in (OUT / "parts_stl_web").glob("*.stl"))


def geometry():
    parts = json.loads((OUT / "parts.json").read_text(encoding="utf-8"))
    out = []
    ntri = 0
    for p in parts["parts"]:
        path = OUT / "parts_stl_web" / f"{p['label']}.stl"
        if not path.exists():
            continue
        m = trimesh.load(path, force="mesh")
        m.merge_vertices()
        # decimate for the web: ~80k triangles for the whole assembly, proportional per part
        budget = int(max(400, 80_000 * len(m.faces) / TOTAL_TRIS))
        if len(m.faces) > budget:
            import fast_simplification as fs
            v, f = fs.simplify(m.vertices.astype(np.float32), m.faces.astype(np.int32), target_count=budget)
            m = trimesh.Trimesh(v, f, process=True)
        ntri += len(m.faces)
        out.append({"label": p["label"], "category": p["category"], "callout": p["callout"],
                    "part_number": p.get("part_number", ""), "color": p["color"],
                    "explode": p["explode"], "pos": p["pos_mm"],
                    "mount": p.get("mount_mm", p["pos_mm"]),          # attachment point on the frame (leader line target)
                    "nv": int(len(m.vertices)), "nf": int(len(m.faces)),
                    "pos_b64": b64(m.vertices.astype(np.float32)),
                    "idx_b64": b64(m.faces.astype(np.uint32))})
    print(f"geometry: {len(out)} parts, {ntri:,} triangles")
    return {"parts": out, "bbox": parts["assembled_bbox_mm"]}


def load(name, optional=False):
    p = OUT / name
    if not p.exists():
        if optional:
            return None
        raise FileNotFoundError(p)
    return json.loads(p.read_text(encoding="utf-8"))


def banana_slice():
    """2D slice of the 880 nm measurement density through the source-detector plane."""
    meta = json.loads((OUT / "voxel_meta.json").read_text(encoding="utf-8"))["rois"]["temple"]
    b = np.load(OUT / "ppg_banana_880.npy").astype(np.float64)
    lab = np.load(OUT / "labels_temple.npy")
    res = meta["res_mm"]; origin = meta["origin_mm"]
    # plane z = 12 mm (the PPG pose height): index along z
    kz = int(round((12.0 - origin[2]) / res - 0.5))
    sl = b[:, :, kz]; lab2 = lab[:, :, kz]
    v = np.log10(np.clip(sl / sl.max(), 1e-6, 1)) / 6 + 1
    v[lab2 == 0] = np.nan
    # crop to x >= 55 mm (skin at ~69.5) and y around the sensor (-90..-66)
    ix0 = int((55 - origin[0]) / res); iy0 = int((-92 - origin[1]) / res); iy1 = int((-64 - origin[1]) / res)
    crop = v[ix0:, iy0:iy1]
    art = (lab2[ix0:, iy0:iy1] == 2)
    return {"x0_mm": 55.0, "y0_mm": -92.0, "res_mm": res, "shape": list(crop.shape),
            "values": np.where(np.isnan(crop), -1, crop).round(3).tolist(),
            "artery_mask_b64": b64(art.astype(np.uint8))}


def img_data_uri(path, max_w=1280):
    im = iio.imread(path)
    if im.shape[1] > max_w:
        f = max_w / im.shape[1]
        h = int(im.shape[0] * f)
        ys = (np.arange(h) / f).astype(int); xs = (np.arange(max_w) / f).astype(int)
        im = im[ys][:, xs]
    buf = io.BytesIO(); iio.imwrite(buf, im[..., :3], extension=".jpg", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def main():
    data = {
        "geometry": geometry(),
        "layout": load("sensor_layout.json"),
        "thermal": load("thermal3d_full.json", optional=True) or load("thermal3d_coarse.json"),
        "optical": load("optical_results.json"),
        "opto_thermal": load("opto_thermal.json"),
        "bioimpedance": load("bioimpedance3d_coarse.json"),
        "pupil": load("pupil_model.json"),
        "verification": load("verification.json", optional=True),
        "surrogate": load("surrogate_thermal.json", optional=True),
        "tissue_db_meta": {k: v for k, v in json.loads((OUT / "tissue_db.json").read_text(encoding="utf-8"))["meta"].items()
                           if k in ("generated", "n_values", "sources")} if (OUT / "tissue_db.json").exists() else {},
        "banana": banana_slice(),
        "images": {},
    }
    rd = ROOT / "sim" / "render" / "out"
    for key, fn in (("hero_face", "hero_face.png"), ("hero_product", "hero_product.png"),
                    ("physics_thermal", "physics_thermal.png"), ("physics_thermography", "physics_thermography.png"),
                    ("explode", "explode_cycles_f090.png")):
        if (rd / fn).exists():
            data["images"][key] = img_data_uri(rd / fn, 1100)
    tpl = (HERE / "template.html").read_text(encoding="utf-8")
    html = tpl.replace("/*__DATA__*/", "const DATA = " + json.dumps(data, separators=(",", ":")) + ";")
    out = HERE / "nrdi_twin.html"
    out.write_text(html, encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
