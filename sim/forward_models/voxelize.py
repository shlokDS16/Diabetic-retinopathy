"""
Voxel label volumes for Monte Carlo (MCX) from the SAME anatomy as the mesh.

periorbital_mesh.classify_volume() is analytic, so the label grid is exact
at any resolution and needs no mesh. Two regions of interest:
    temple  -- around the MAX30102 PPG site over the superficial temporal artery
    eye     -- around the left orbit for the 940 nm illuminator / cornea check
Label 0 = outside the head (air, tracked as a medium so LED light can cross
the gap to the cornea); 1..14 = periorbital_mesh.REGIONS.

Also writes the per-label optical property table for each device wavelength
from sim/out/tissue_db.json:  [mu_a, mu_s, g, n] in 1/mm (MCX convention,
mu_s = mu_s' / (1 - g)).

Run (nrdi-env):  python sim/forward_models/voxelize.py
"""
import sys, json, pathlib
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cad"))
import periorbital_mesh as PM                                  # noqa: E402
import sensor_layout as L                                      # noqa: E402

OUT = HERE.parent / "out"
DB = json.loads((OUT / "tissue_db.json").read_text(encoding="utf-8"))["tissues"]

OPT_SRC = {   # mesh region -> tissue_db optical entry
    "skin": "skin", "fat_subcutaneous": "fat_subcutaneous", "muscle": "muscle",
    "bone_cortical": "bone_cortical", "brain_grey": "brain_grey", "eyelid": "eyelid",
    "orbital_fat": "orbital_fat", "cornea": "cornea", "aqueous": "aqueous", "lens": "lens",
    "vitreous": "vitreous", "sclera": "sclera", "artery_angular": "blood", "artery_temporal": "blood",
}
ROIS = {
    "temple": {"lo": (44.0, -100.0, -10.0), "hi": (80.0, -56.0, 34.0), "res": 0.25},
    "eye":    {"lo": (6.0, -32.0, -26.0),   "hi": (58.0, 22.0, 26.0),  "res": 0.4},
}


def _v(x):
    return float(x["value"]) if isinstance(x, dict) else float(x)


def optical_table(wavelength):
    """Rows indexed by label: row 0 = air, rows 1..14 = REGIONS."""
    rows = [[0.0, 0.0, 1.0, 1.0]]
    for nm in PM.REGIONS:
        o = DB[OPT_SRC[nm]]["optical"][str(wavelength)]
        mua, musp, g, n = _v(o["mu_a"]), _v(o["mu_s_prime"]), _v(o["g"]), _v(o["n"])
        rows.append([mua, musp / max(1.0 - g, 1e-3), g, n])
    return np.array(rows, dtype=np.float32)


def in_domain(p):
    """Head ∪ lid pad ∪ brow ∪ nose ∪ globe, minus the palpebral-fissure air (same rules as the CAD)."""
    x, y, z = p.T
    head = PM._ell(p, PM.HEAD_C, PM.HEAD_A) <= 1.0
    brow, nose = PM._bulge(p, 0.0)                       # anatomy v2 bulges
    in_cone = (y <= PM.ORBIT_FRONT_Y) & (y >= PM.ORBIT_APEX_Y) & (np.hypot(x - PM.PX, z) <= PM._cone_r(y))
    pad = in_cone & (PM._ell(p, PM.HEAD_C, (PM.HEAD_A[0], PM.HEAD_A[1] + PM.PAD_DY, PM.HEAD_A[2])) <= 1.0) & (x >= PM.PAD_XMIN)
    dg = np.sqrt((x - PM.PX) ** 2 + (y - PM.GLOBE_CY) ** 2 + z ** 2)
    dc = np.sqrt((x - PM.PX) ** 2 + (y - PM.CORNEA_CY) ** 2 + z ** 2)
    globe = (dg <= PM.GLOBE_R) | (dc <= PM.CORNEA_R)
    fissure = (((x - PM.PX) / PM.FISSURE_RX) ** 2 + (z / PM.FISSURE_RZ) ** 2 <= 1.0) & (y >= PM.LID_Y)
    air = fissure & ~globe
    return (head | pad | brow | nose | globe) & ~air


def voxelize(name, roi):
    lo, hi, res = np.array(roi["lo"]), np.array(roi["hi"]), roi["res"]
    n = np.ceil((hi - lo) / res).astype(int)
    ax = [lo[i] + (np.arange(n[i]) + 0.5) * res for i in range(3)]
    X, Y, Z = np.meshgrid(*ax, indexing="ij")
    P = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    ang_pts = [(PM.ANG_X, PM.surface_y(PM.ANG_X, z) - PM.VESSEL_DEPTH, z) for z in (-15.0, 15.0)]
    tmp_pts = [(PM.surface_x(PM.TMP_Y, z) - PM.VESSEL_DEPTH, PM.TMP_Y, z) for z in (-5.0, 30.0)]
    lab = np.zeros(len(P), dtype=np.uint8)
    dom = in_domain(P)
    lab[dom] = PM.classify_volume(P[dom], ang_pts, tmp_pts).astype(np.uint8)
    vol = lab.reshape(n)
    np.save(OUT / f"labels_{name}.npy", vol)
    meta = {"name": name, "origin_mm": lo.tolist(), "res_mm": res, "shape": n.tolist(),
            "labels": {0: "air", **{i + 1: nm for i, nm in enumerate(PM.REGIONS)}},
            "counts": {nm: int((vol == i + 1).sum()) for i, nm in enumerate(PM.REGIONS)}}
    return vol, meta


if __name__ == "__main__":
    meta_all = {"rois": {}, "optical": {}}
    for name, roi in ROIS.items():
        vol, meta = voxelize(name, roi)
        meta_all["rois"][name] = meta
        print(f"{name:7s} {vol.shape} @ {roi['res']} mm  tissue voxels {int((vol>0).sum()):,}  "
              f"artery {meta['counts'].get('artery_temporal', 0) + meta['counts'].get('artery_angular', 0):,}  "
              f"cornea {meta['counts']['cornea']:,}")
    for wl in (660, 880, 940):
        t = optical_table(wl)
        meta_all["optical"][str(wl)] = t.tolist()
        print(f"{wl} nm: skin mua {t[10][0]:.4f} mus {t[10][1]:.2f} | blood mua {t[2][0]:.3f} mus {t[2][1]:.1f}")
    (OUT / "voxel_meta.json").write_text(json.dumps(meta_all, indent=1), encoding="utf-8")
    print("wrote labels_*.npy + voxel_meta.json")
