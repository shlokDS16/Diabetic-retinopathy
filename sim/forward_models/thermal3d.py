"""
3D Pennes bioheat model on the shared periorbital mesh, with the thermopile
sensor forward model.

    -div(k grad T) + w (T - T_art) = q_m          in each tissue
    -k dT/dn = h (T - T_amb) + E                   on exposed surfaces
    T = T_art                                      on artery lumen walls
    zero flux                                      symmetry plane, far cut planes

w = rho_b c_b w_b  [W/m3/K] is the volumetric perfusion coefficient. The
choroid is not meshed separately: the 1 mm sclera shell carries a perfusion
term calibrated so that its heat exchange equals Scott's (1988) posterior
boundary coefficient h_bl = 65 W/m2/K (the value the validated axisymmetric
model uses) -- that is w_choroid = h_bl / shell_thickness = 65e3 W/m3/K.
"Choroidal perfusion fraction" scales that number: it is the disease handle.

Tissue properties come from sim/out/tissue_db.json when present (cited,
with uncertainty ranges); otherwise the provisional table below is used
and the run is flagged PROVISIONAL.

Sensor forward model (thermopile): the reading is the emissivity-weighted
radiometric average over the boundary facets inside the sensor's field-of-view
cone, each weighted by projected solid angle (area * cos(theta) / d^2), plus
an ambient reflection term (1 - eps). This is what the MLX90614 actually
measures, not the temperature at one point.

Run:  python sim/forward_models/thermal3d.py [--mesh sim/out/periorbital_coarse.msh]
"""
import sys, json, pathlib, time, argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import (MeshTet, Basis, FacetBasis, ElementTetP1, BilinearForm,
                   LinearForm, asm, condense, solve)
from skfem.helpers import dot, grad

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cad"))
import sensor_layout as L                                    # noqa: E402

OUT = HERE.parent / "out"
MM = 1e-3
SIGMA = 5.670e-8

# ---- provisional properties (used only if tissue_db.json is absent) -------
# k [W/m/K], w [W/m3/K] = rho_b c_b w_b, q_m [W/m3]. Values follow the IT'IS
# tissue database / Scott 1988 conventions; the cited db supersedes them.
RHO_B, C_B = 1050.0, 3617.0
def _w(ml_min_kg, rho):
    return RHO_B * C_B * ml_min_kg * rho / 60.0 / 1e6
PROVISIONAL = {
    "skin":             {"k": 0.37, "w": _w(106.0, 1109.0), "q_m": 1.65 * 1109.0},
    "fat_subcutaneous": {"k": 0.21, "w": _w(33.0, 911.0),   "q_m": 0.51 * 911.0},
    "muscle":           {"k": 0.49, "w": _w(37.0, 1090.0),  "q_m": 0.91 * 1090.0},
    "bone_cortical":    {"k": 0.32, "w": _w(10.0, 1908.0),  "q_m": 0.15 * 1908.0},
    "brain_grey":       {"k": 0.55, "w": _w(760.0, 1045.0), "q_m": 15.5 * 1045.0},
    "eyelid":           {"k": 0.40, "w": _w(70.0, 1100.0),  "q_m": 1.2 * 1100.0},
    "orbital_fat":      {"k": 0.21, "w": _w(33.0, 911.0),   "q_m": 0.51 * 911.0},
    "cornea":           {"k": 0.58, "w": 0.0, "q_m": 0.0},
    "aqueous":          {"k": 0.58, "w": 0.0, "q_m": 0.0},
    "lens":             {"k": 0.40, "w": 0.0, "q_m": 0.0},
    "vitreous":         {"k": 0.603, "w": 0.0, "q_m": 0.0},
    "sclera":           {"k": 1.0042, "w": 65.0e3, "q_m": 0.0},   # includes choroid, see header
    "artery_angular":   {"k": 0.52, "w": 0.0, "q_m": 0.0},
    "artery_temporal":  {"k": 0.52, "w": 0.0, "q_m": 0.0},
}
BOUNDARY = {
    "T_amb": 25.0, "T_art": 37.0,
    "h_cornea": 10.0, "eps_cornea": 0.975, "E_tear": 40.0,      # Scott 1988
    "h_skin": 5.0, "eps_skin": 0.98, "E_sweat": 10.0,            # still air, insensible loss
}


BOUNDARY_MAP = {          # tissue_db.json key -> solver key
    "ambient_temperature_C": "T_amb", "arterial_blood_temperature_C": "T_art",
    "cornea_convection_h": "h_cornea", "cornea_emissivity": "eps_cornea",
    "tear_evaporation_heat_loss": "E_tear", "skin_convection_h_still_air": "h_skin",
    "skin_emissivity": "eps_skin", "skin_insensible_evaporation_facial": "E_sweat",
}


def load_properties(db_path=OUT / "tissue_db.json"):
    """Return (props, boundary, provenance) using the cited DB when available.

    DB schema (sim/out/tissue_db.json): tissues[name].thermal.{k,rho,c,w_b_s,q_m_vol}
    each as {"value", "unit", "source", "confidence", "uncertainty"}.
    """
    if not db_path.exists():
        return PROVISIONAL, BOUNDARY, "PROVISIONAL (tissue_db.json not found)"
    db = json.loads(db_path.read_text(encoding="utf-8"))
    tissues = db["tissues"]
    props = {}
    for name, dflt in PROVISIONAL.items():
        src = "blood" if name.startswith("artery") else name
        t = tissues.get(src, {}).get("thermal", {})
        k = _num(t.get("k"), dflt["k"])
        wb = _num(t.get("w_b_s"), None)
        qm = _num(t.get("q_m_vol"), None)
        w = RHO_B * C_B * wb if wb is not None else dflt["w"]
        if name == "sclera":
            # 1 mm shell stands in for sclera+choroid: calibrated to Scott's
            # h_bl = 65 W/m2/K (see header). The DB choroid value gives ~79e3
            # W/m3/K for a 0.2 mm choroid averaged over the shell -- consistent.
            w = dflt["w"]
        if name.startswith("artery"):
            w, qm = 0.0, 0.0
        props[name] = {"k": k, "w": w, "q_m": qm if qm is not None else dflt["q_m"]}
    bnd = dict(BOUNDARY)
    for key, skey in BOUNDARY_MAP.items():
        v = _num(db.get("boundary", {}).get(key), None)
        if v is not None:
            bnd[skey] = v
    return props, bnd, f"tissue_db.json ({db.get('meta', {}).get('generated', 'cited DB')})"


def _num(v, default):
    if isinstance(v, dict):
        v = v.get("value", default)
    return default if v is None else float(v)


def h_rad(eps, T0_C=34.0):
    return 4.0 * eps * SIGMA * (T0_C + 273.15) ** 3


class Thermal3D:
    def __init__(self, mesh_path):
        t = time.time()
        m = MeshTet.load(str(mesh_path))
        self.m = m.scaled(MM)
        self.basis = Basis(self.m, ElementTetP1())
        self.props, self.bnd, self.provenance = load_properties()
        self.regions = {nm: elems for nm, elems in self.m.subdomains.items()}
        self.load_time = time.time() - t
        self._cache = {}

    # -------------------------------------------------------------- assembly
    def _region_matrices(self):
        """Per-region stiffness K_r, mass M_r and load vectors, assembled once."""
        if self._cache:
            return self._cache
        @BilinearForm
        def stiff(u, v, w):
            return dot(grad(u), grad(v))
        @BilinearForm
        def mass(u, v, w):
            return u * v
        @LinearForm
        def one(v, w):
            return 1.0 * v
        for nm, elems in self.regions.items():
            sb = Basis(self.m, ElementTetP1(), elements=elems)
            self._cache[nm] = (asm(stiff, sb), asm(mass, sb), asm(one, sb))
        # boundary facet bases
        for nm in ("skin_exposed", "cornea_exposed", "conj_exposed"):
            if nm in self.m.boundaries:
                fb = FacetBasis(self.m, ElementTetP1(), facets=self.m.boundaries[nm])
                self._cache["fb_" + nm] = (asm(mass, fb), asm(one, fb))
        return self._cache

    def solve(self, choroid_frac=1.0, skin_perf_frac=1.0, T_amb=None, T_art=None,
              h_skin=None, extra_source=None, eyelid_perf_frac=1.0, orbit_perf_frac=1.0,
              E_tear=None, k_scale=None):
        """Steady solution. extra_source: nodal volumetric heat [W/m3] (e.g. LED).
        k_scale: optional dict region -> multiplier on conductivity (UQ)."""
        c = self._region_matrices()
        b = self.bnd
        T_amb = b["T_amb"] if T_amb is None else T_amb
        T_art = b["T_art"] if T_art is None else T_art
        h_skin = b["h_skin"] if h_skin is None else h_skin
        E_tear = b["E_tear"] if E_tear is None else E_tear
        k_scale = k_scale or {}
        N = self.basis.N
        A = sp.csr_matrix((N, N))
        rhs = np.zeros(N)
        for nm, (K, M, one) in ((k, v) for k, v in c.items() if not k.startswith("fb_")):
            p = self.props[nm]
            w = p["w"]
            if nm == "sclera":
                w *= choroid_frac
            if nm == "skin":
                w *= skin_perf_frac
            if nm == "eyelid":
                w *= skin_perf_frac * eyelid_perf_frac
            if nm == "orbital_fat":
                w *= orbit_perf_frac
            A = A + p["k"] * k_scale.get(nm, 1.0) * K + w * M
            rhs = rhs + w * T_art * one + p["q_m"] * one
        # exposed surfaces
        for nm, h, eps, E in (("skin_exposed", h_skin, b["eps_skin"], b["E_sweat"]),
                              ("cornea_exposed", b["h_cornea"], b["eps_cornea"], E_tear),
                              ("conj_exposed", b["h_cornea"], b["eps_cornea"], E_tear)):
            if "fb_" + nm not in c:
                continue
            Mf, onef = c["fb_" + nm]
            htot = h + h_rad(eps)
            A = A + htot * Mf
            rhs = rhs + (htot * T_amb - E) * onef
        if extra_source is not None:
            _, M0, _ = c["skin"]      # any mass matrix has the right sparsity; build global
            Mg = sum(v[1] for k, v in c.items() if not k.startswith("fb_"))
            rhs = rhs + Mg @ extra_source
        # Dirichlet on artery walls
        D = np.array([], dtype=np.int64)
        for nm in ("artery_angular_wall", "artery_temporal_wall"):
            if nm in self.m.boundaries:
                D = np.union1d(D, self.basis.get_dofs(facets=self.m.boundaries[nm]).all())
        x0 = np.full(N, T_art)
        T = solve(*condense(A.tocsr(), rhs, x=x0, D=D))
        return T

    # ------------------------------------------------------------- probes
    def point_T(self, T, xyz_mm):
        """Temperature at the mesh node nearest to a point given in mm."""
        p = np.asarray(xyz_mm) * MM
        i = np.argmin(np.sum((self.m.p.T - p) ** 2, axis=1))
        return float(T[i]), self.m.p[:, i] / MM

    def corneal_apex(self, T):
        return self.point_T(T, (L.IPD / 2, 0.0, 0.0))[0]

    def boundary_facets(self, names):
        f = np.concatenate([self.m.boundaries[n] for n in names if n in self.m.boundaries])
        tri = self.m.facets[:, f]                       # (3, nf)
        P = self.m.p
        a, b_, c_ = P[:, tri[0]], P[:, tri[1]], P[:, tri[2]]
        n = np.cross((b_ - a).T, (c_ - a).T)
        area = 0.5 * np.linalg.norm(n, axis=1)
        n = n / (2 * area[:, None] + 1e-30)
        cen = (a + b_ + c_).T / 3.0
        # outward orientation: away from the domain centroid of the element that owns it
        t2f = self.m.f2t[0, f]
        ec = P[:, self.m.t[:, t2f]].mean(axis=1).T
        flip = np.einsum("ij,ij->i", n, cen - ec) < 0
        n[flip] *= -1
        return tri, cen, n, area

    def thermopile_reading(self, T, sensor_name, fov_deg=None, eps=0.98, T_amb=None):
        """Radiometric reading of a thermopile at a sensor_layout pose."""
        s = next(x for x in L.SENSORS if x.name == sensor_name)
        fov = s.fov_deg if fov_deg is None else fov_deg
        T_amb = self.bnd["T_amb"] if T_amb is None else T_amb
        pos = np.asarray(s.pos) * MM
        axis = np.asarray(s.unit_axis())
        tri, cen, n, area = self.boundary_facets(["skin_exposed", "cornea_exposed", "conj_exposed"])
        v = cen - pos
        d = np.linalg.norm(v, axis=1)
        u = v / d[:, None]
        in_cone = np.arccos(np.clip(u @ axis, -1, 1)) <= np.radians(fov / 2)
        facing = np.einsum("ij,ij->i", n, -u) > 0
        sel = in_cone & facing
        if not sel.any():
            # narrow cone (spot smaller than a facet): interpolate T where the
            # sensor axis hits the surface instead of returning NaN
            cand = (np.arccos(np.clip(u @ axis, -1, 1)) <= np.radians(max(fov, 10.0))) & facing
            hit = self._ray_hit(pos, axis, tri[:, cand])
            if hit is None:
                return float("nan"), 0.0
            k, bary, dist = hit
            Tf = float(T[tri[:, cand][:, k]] @ bary) + 273.15
            rad = eps * Tf ** 4 + (1 - eps) * (T_amb + 273.15) ** 4
            return float(rad ** 0.25 - 273.15), float(2 * dist * np.tan(np.radians(fov / 2)) / MM)
        wgt = area[sel] * np.einsum("ij,ij->i", n[sel], -u[sel]) / d[sel] ** 2
        Tf = T[tri[:, sel]].mean(axis=0) + 273.15
        rad = (wgt * (eps * Tf ** 4 + (1 - eps) * (T_amb + 273.15) ** 4)).sum() / wgt.sum()
        spot_mm = 2 * np.median(d[sel]) * np.tan(np.radians(fov / 2)) / MM
        return float(rad ** 0.25 - 273.15), float(spot_mm)

    def _ray_hit(self, pos, axis, tri):
        """Moller-Trumbore: first triangle of `tri` (3, n) hit by the ray pos + t*axis.
        Returns (index, barycentric weights, distance) or None."""
        P = self.m.p
        v0, v1, v2 = P[:, tri[0]].T, P[:, tri[1]].T, P[:, tri[2]].T
        e1, e2 = v1 - v0, v2 - v0
        h = np.cross(axis, e2)
        a = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(a) > 1e-30
        f = np.where(ok, 1.0 / np.where(ok, a, 1.0), 0.0)
        s_ = pos - v0
        uu = f * np.einsum("ij,ij->i", s_, h)
        q = np.cross(s_, e1)
        vv = f * (q @ axis)
        t = f * np.einsum("ij,ij->i", e2, q)
        ok &= (uu >= -1e-9) & (vv >= -1e-9) & (uu + vv <= 1 + 1e-9) & (t > 1e-9)
        if not ok.any():
            return None
        k = int(np.argmin(np.where(ok, t, np.inf)))
        return k, np.array([1 - uu[k] - vv[k], uu[k], vv[k]]), float(t[k])

    # -------------------------------------------------------------- export
    def export_vtu(self, T, path, extra=None):
        import meshio
        pd = {"T": T}
        if extra:
            pd.update(extra)
        region = np.zeros(self.m.t.shape[1], dtype=np.int32)
        names = list(self.regions)
        for i, nm in enumerate(names):
            region[self.regions[nm]] = i + 1
        meshio.write(str(path), meshio.Mesh(self.m.p.T / MM, [("tetra", self.m.t.T)],
                                            point_data=pd, cell_data={"region": [region]}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=str(OUT / "periorbital.msh"))
    ap.add_argument("--no-sweep", action="store_true")
    args = ap.parse_args()
    mesh_path = pathlib.Path(args.mesh)
    if not mesh_path.exists():
        mesh_path = OUT / "periorbital_coarse.msh"

    th = Thermal3D(mesh_path)
    print(f"mesh {mesh_path.name}: {th.m.p.shape[1]:,} nodes, {th.m.t.shape[1]:,} tets "
          f"(load {th.load_time:.1f}s)  properties: {th.provenance}")
    t = time.time(); th._region_matrices(); print(f"assembly {time.time()-t:.1f}s")
    t = time.time(); T = th.solve(); print(f"solve {time.time()-t:.1f}s")

    canthus = (L.IPD / 2 - L.MEDIAL_CANTHUS_DX, 2.0, L.MEDIAL_CANTHUS_DZ)
    apex = th.corneal_apex(T)
    canth_T, canth_p = th.point_T(T, canthus)
    temple_T, _ = th.point_T(T, (L.TEMPLE_ARTERY_X, L.TEMPLE_ARTERY_Y, L.TEMPLE_ARTERY_Z))
    baa, spot_baa = th.thermopile_reading(T, "thermo_L")
    dci, spot_dci = th.thermopile_reading(T, "thermo_L", fov_deg=5.0)

    res = {
        "mesh": str(mesh_path), "properties": th.provenance,
        "T_range_C": [float(T.min()), float(T.max())],
        "corneal_apex_C": apex,
        "medial_canthus_skin_C": canth_T,
        "temple_skin_C": temple_T,
        "thermopile_BAA_90deg": {"reading_C": baa, "spot_mm": spot_baa},
        "thermopile_DCI_5deg": {"reading_C": dci, "spot_mm": spot_dci},
        "validation": {
            "corneal_apex_published_C": [34.0, 35.0],
            "inner_canthus_published_C": [35.0, 36.5],
        },
    }
    print(f"\nT range        {T.min():.2f} .. {T.max():.2f} C")
    print(f"corneal apex   {apex:.2f} C   (published 34-35)")
    print(f"medial canthus {canth_T:.2f} C   (published ~35-36.5, node at {np.round(canth_p,1)})")
    print(f"temple skin    {temple_T:.2f} C")
    print(f"thermopile BAA 90deg: {baa:.2f} C over {spot_baa:.1f} mm spot")
    print(f"thermopile DCI  5deg: {dci:.2f} C over {spot_dci:.1f} mm spot")

    if not args.no_sweep:
        rows = []
        print(f"\n{'choroid frac':>12s} {'apex':>7s} {'dT apex':>8s} {'canthus':>8s} {'DCI':>7s} {'BAA':>7s}")
        for frac in (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4):
            Ts = th.solve(choroid_frac=frac)
            a = th.corneal_apex(Ts)
            cT = th.point_T(Ts, canthus)[0]
            d_, _ = th.thermopile_reading(Ts, "thermo_L", fov_deg=5.0)
            b_, _ = th.thermopile_reading(Ts, "thermo_L")
            rows.append({"choroid_frac": frac, "apex_C": a, "d_apex_C": a - apex,
                         "canthus_C": cT, "dci_C": d_, "baa_C": b_})
            print(f"{frac:12.2f} {a:7.2f} {a-apex:8.3f} {cT:8.2f} {d_:7.2f} {b_:7.2f}")
        res["choroid_sweep"] = rows
        rows = []
        print(f"\n{'skin perf frac':>14s} {'canthus':>8s} {'DCI':>7s} {'temple':>7s}")
        for frac in (1.0, 0.7, 0.5):
            Ts = th.solve(skin_perf_frac=frac)
            cT = th.point_T(Ts, canthus)[0]
            d_, _ = th.thermopile_reading(Ts, "thermo_L", fov_deg=5.0)
            tT = th.point_T(Ts, (L.TEMPLE_ARTERY_X, L.TEMPLE_ARTERY_Y, L.TEMPLE_ARTERY_Z))[0]
            rows.append({"skin_perf_frac": frac, "canthus_C": cT, "dci_C": d_, "temple_C": tT})
            print(f"{frac:14.2f} {cT:8.2f} {d_:7.2f} {tT:7.2f}")
        res["skin_perfusion_sweep"] = rows

    tag = "coarse" if "coarse" in mesh_path.name else "full"
    (OUT / f"thermal3d_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    np.save(OUT / f"thermal3d_T_{tag}.npy", T)
    th.export_vtu(T, OUT / f"thermal3d_{tag}.vtu")
    print(f"\nwrote thermal3d_{tag}.json / .npy / .vtu")


if __name__ == "__main__":
    main()
