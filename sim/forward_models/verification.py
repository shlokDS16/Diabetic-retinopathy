"""
Solution verification for the thermal solver (ASME V&V 20 / V&V 40 "code and
solution verification").

1. Method of manufactured solutions (MMS) on the periorbital meshes:
   pick T_m = 37 + A sin(kx) cos(ky) e^{-z/l}, derive the source that makes
   it satisfy -div(k grad T) + w (T - T_a) = q with UNIFORM k and w (a smooth
   manufactured field cannot satisfy the flux-jump conditions of piecewise
   properties, so code verification uses uniform coefficients on the real
   geometry), impose T_m on every boundary, solve, and report the L2 / Linf
   error. A P1 scheme must converge at second order in h; we check the error
   ratio between the coarse and fine meshes.

2. Grid convergence (Roache GCI) on the quantities of interest: corneal apex
   temperature and the DCI thermopile reading, using the coarse and fine
   meshes (r = 2) with the assumed order p = 2 (P1 elements) and Fs = 3 for a
   two-grid study.

Run (nrdi-env):  python sim/forward_models/verification.py
"""
import sys, json, pathlib, time
import numpy as np
from skfem import MeshTet, Basis, ElementTetP1, BilinearForm, LinearForm, asm, condense, solve
from skfem.helpers import dot, grad

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cad"))
import thermal3d as T3                                       # noqa: E402
import sensor_layout as L                                    # noqa: E402
OUT = HERE.parent / "out"

A_M, K_M, L_M = 2.0, 60.0, 0.03          # amplitude [K], wavenumber [1/m], decay length [m]


def T_manufactured(x, y, z):
    return 37.0 + A_M * np.sin(K_M * x) * np.cos(K_M * y) * np.exp(-z / L_M)


def laplacian_manufactured(x, y, z):
    s = np.sin(K_M * x) * np.cos(K_M * y) * np.exp(-z / L_M)
    return A_M * (-2 * K_M ** 2 + 1.0 / L_M ** 2) * s


def mms(mesh_path):
    th = T3.Thermal3D(mesh_path)
    m, basis = th.m, th.basis
    # uniform coefficients for code verification (see header)
    k_el = np.full(m.t.shape[1], 0.5); w_el = np.full(m.t.shape[1], 5000.0)

    @BilinearForm
    def a(u, v, w):
        return w["k"] * dot(grad(u), grad(v)) + w["wb"] * u * v

    @LinearForm
    def rhs(v, w):
        x, y, z = w.x
        Tm = T_manufactured(x, y, z)
        q = -w["k"] * laplacian_manufactured(x, y, z) + w["wb"] * (Tm - 37.0)
        return (q + w["wb"] * 37.0) * v          # bilinear form carries w*T, so move w*T_a to the rhs

    nq = basis.X.shape[1]
    kq = np.repeat(k_el[:, None], nq, axis=1); wq = np.repeat(w_el[:, None], nq, axis=1)
    A = asm(a, basis, k=kq, wb=wq)
    b = asm(rhs, basis, k=kq, wb=wq)
    Tex = T_manufactured(*m.p)
    D = basis.get_dofs().all()                                  # all boundary dofs
    T = solve(*condense(A, b, x=Tex, D=D))
    err = T - Tex
    pairs = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    edge = np.stack([np.linalg.norm(m.p[:, m.t[i]] - m.p[:, m.t[j]], axis=0) for i, j in pairs])
    h = float(edge.max(axis=0).mean())                          # mean over elements of the longest edge
    return {"nodes": int(m.p.shape[1]), "h_mm": h * 1e3,
            "L2_rel": float(np.linalg.norm(err) / np.linalg.norm(Tex - 37.0)),
            "Linf_K": float(np.abs(err).max())}


def gci(f_fine, f_coarse, r=2.0, p=2.0, Fs=3.0):
    """Roache grid convergence index on the fine grid, relative."""
    eps = (f_coarse - f_fine) / f_fine
    return {"fine": f_fine, "coarse": f_coarse,
            "richardson_extrapolated": f_fine + (f_fine - f_coarse) / (r ** p - 1),
            "GCI_fine_percent": 100.0 * Fs * abs(eps) / (r ** p - 1)}


def main():
    res = {}
    print("MMS (manufactured solution, Dirichlet everywhere):")
    for tag in ("coarse", "full"):
        path = OUT / ("periorbital_coarse.msh" if tag == "coarse" else "periorbital.msh")
        t = time.time(); r = mms(path); r["time_s"] = time.time() - t
        res[f"mms_{tag}"] = r
        print(f"  {tag:6s}: {r['nodes']:>8,} nodes, h ~ {r['h_mm']:.2f} mm, L2 rel {r['L2_rel']:.3e}, "
              f"Linf {r['Linf_K']:.3e} K  ({r['time_s']:.0f}s)")
    ratio = res["mms_coarse"]["L2_rel"] / res["mms_full"]["L2_rel"]
    hr = res["mms_coarse"]["h_mm"] / res["mms_full"]["h_mm"]
    p_obs = float(np.log(ratio) / np.log(hr))
    res["observed_order"] = p_obs
    print(f"  observed order p = {p_obs:.2f} (h ratio {hr:.2f}; P1 expects ~2)")

    print("\nGCI on quantities of interest (coarse vs fine runs of thermal3d):")
    c = json.loads((OUT / "thermal3d_coarse.json").read_text(encoding="utf-8"))
    f = json.loads((OUT / "thermal3d_full.json").read_text(encoding="utf-8"))
    qoi = {"corneal_apex_C": (f["corneal_apex_C"], c["corneal_apex_C"]),
           "canthus_skin_C": (f["medial_canthus_skin_C"], c["medial_canthus_skin_C"]),
           "thermopile_DCI_C": (f["thermopile_DCI_5deg"]["reading_C"], c["thermopile_DCI_5deg"]["reading_C"]),
           "deficit_50pct_choroid_C": (f["choroid_sweep"][5]["d_apex_C"], c["choroid_sweep"][5]["d_apex_C"])}
    res["gci"] = {}
    for k, (ff, fc) in qoi.items():
        g = gci(ff, fc, r=hr, p=max(p_obs, 1.0))
        res["gci"][k] = g
        print(f"  {k:26s} fine {ff:8.4f}  coarse {fc:8.4f}  extrap {g['richardson_extrapolated']:8.4f}  GCI {g['GCI_fine_percent']:.3f} %")
    (OUT / "verification.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("\nwrote verification.json")


if __name__ == "__main__":
    main()
