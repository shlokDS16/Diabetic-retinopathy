"""
Periorbital bioimpedance forward model on the shared mesh (quasi-static,
complex conductivity, 1-100 kHz).

    div( sigma*(f) grad u ) = 0,   sigma* = sigma + i w eps0 eps_r   per tissue
    -sigma* du/dn = +-I / A_l       on drive-electrode skin patches (gap model)
    U_l = (1/A_l) int_{E_l} u dS    measured electrode potential

Electrode patches are the skin_exposed facets within PAD_R of each pad centre
from sensor_layout v3 (2026-09-04): four contacts per eye where the eyewear
already touches the face -- brow_med / brow_lat on the brow bumper (supra-
orbital ridge, anatomy-v2 brow bulge) and nose_sup / nose_inf on the nose pad
(nasal sidewall, anatomy-v2 nose bulge). Left eye only; the reference node
(deepest node of the mesh) is grounded. The half-head mesh's x = 0 Neumann
plane is equivalent to an in-phase mirrored drive on the right eye; for the
nose contacts 7.5-8.5 mm from the midline this overstates |Z| slightly
(no current crosses the midline). Noted, not corrected.

Configurations
  bipolar_L          drive brow_med -> nose_sup, measure the same pair
                     (+ 2 x contact impedance z_c/A): what a 2-electrode AD5933
                     reading actually contains.
  tetrapolar_local   drive brow_lat -> nose_inf (I = 1 mA), sense
                     brow_med - nose_sup. The current path crosses the orbit.
                     Lead field (Geselowitz): Z = (1/I^2) int sigma* grad u_d . grad u_m,
                     computed per tissue -> which tissue the measurement "sees".
                     Two internal checks: the reciprocity residual
                     |Z_recip - Z| / |Z| (Z from the drive solve vs Z from the
                     reciprocal solve driving the sense pair) and the Geselowitz
                     residual |sum_r Z_r - Z| / |Z|.

Disease / coupling handles
  edema   conductivity of orbital_fat and eyelid x (1 + delta): extracellular
          water expansion (DME / periorbital edema). Reported as dZ, d(phase).
  sigma(T)  sigma x (1 + 0.02 (T - 37)) with T from thermal3d -> the
          thermo-electrical coupling term.

Run (nrdi-env):  python sim/forward_models/bioimpedance3d.py [--mesh ...]
"""
import sys, json, pathlib, time, argparse
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from skfem import MeshTet, Basis, FacetBasis, ElementTetP1, BilinearForm, LinearForm, asm
from skfem.helpers import dot, grad

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cad"))
import sensor_layout as L                                   # noqa: E402

OUT = HERE.parent / "out"
MM = 1e-3
EPS0 = 8.854e-12
PAD_R = 3.0e-3                    # electrode contact radius [m] (6 mm contact, sensor_layout v3)
Z_CONTACT = 0.05                  # dry-electrode contact impedance [Ohm m^2] (= 0.5 kOhm cm^2), estimate
I_DRIVE = 1e-3                    # 1 mA rms (AD5933 excitation is voltage; 1 mA is the design current)
FREQS = [1e3, 10e3, 50e3, 100e3]
TEMP_COEFF = 0.02                 # 1/K conductivity temperature coefficient (electrolytes ~2 %/K)
EDEMA_TISSUES = ("orbital_fat", "eyelid")
ORBIT_TISSUES = ("orbital_fat", "eyelid", "sclera", "vitreous", "aqueous", "lens", "cornea")

DRIVE = ("elec_L_brow_lat", "elec_L_nose_inf")     # local tetrapolar: current pair
SENSE = ("elec_L_brow_med", "elec_L_nose_sup")     # local tetrapolar: voltage pair
BIPOLAR = ("elec_L_brow_med", "elec_L_nose_sup")   # 2-electrode comparison

DIEL_SRC = {"skin": "skin", "fat_subcutaneous": "fat_subcutaneous", "muscle": "muscle",
            "bone_cortical": "bone_cortical", "brain_grey": "brain_grey", "eyelid": "eyelid",
            "orbital_fat": "orbital_fat", "cornea": "cornea", "aqueous": "aqueous", "lens": "lens",
            "vitreous": "vitreous", "sclera": "sclera", "artery_angular": "blood", "artery_temporal": "blood"}


def _v(x, d=None):
    return float(x["value"]) if isinstance(x, dict) else (d if x is None else float(x))


def dielectric_table():
    db = json.loads((OUT / "tissue_db.json").read_text(encoding="utf-8"))["tissues"]
    tab = {}
    for reg, src in DIEL_SRC.items():
        de = db[src]["dielectric"]
        sig0 = _v(de.get("sigma_lowfreq_recommended"), None)
        per = de.get("per_frequency", {})
        rows = {}
        for f in FREQS:
            e = per.get(str(int(f)), {})
            sig = _v(e.get("sigma"), sig0)
            epsr = None
            for k in ("eps_r", "epsr", "eps_r_gabriel_cole_cole", "epsr_gabriel", "eps_r_gabriel"):
                if k in e:
                    epsr = _v(e[k]); break
            if epsr is None:
                cc = e.get("eps_r_cole_cole") or e.get("permittivity")
                epsr = _v(cc, 1e4) if cc is not None else 1e4      # fallback: typical kHz-range tissue eps_r
            rows[f] = complex(sig, 2 * np.pi * f * EPS0 * epsr)
        tab[reg] = rows
    return tab


class Bioimpedance3D:
    def __init__(self, mesh_path, T_nodal=None):
        t0 = time.time()
        self.m = MeshTet.load(str(mesh_path)).scaled(MM)
        self.basis = Basis(self.m, ElementTetP1())
        self.diel = dielectric_table()
        self.regions = dict(self.m.subdomains)
        self.T = T_nodal
        self._K = {}
        self._Kt = {}
        self._assemble()
        self.load_time = time.time() - t0

    def _assemble(self):
        @BilinearForm
        def stiff(u, v, w):
            return dot(grad(u), grad(v))
        @BilinearForm
        def stiff_w(u, v, w):
            return w["s"] * dot(grad(u), grad(v))
        Tm = None
        if self.T is not None:
            Tm = self.T[self.m.t].mean(axis=0)                   # element-mean temperature
        for nm, elems in self.regions.items():
            sb = Basis(self.m, ElementTetP1(), elements=elems)
            self._K[nm] = asm(stiff, sb)
            if Tm is not None:
                f = 1.0 + TEMP_COEFF * (Tm[elems] - 37.0)
                self._Kt[nm] = asm(stiff_w, sb, s=np.repeat(f[:, None], sb.X.shape[1], axis=1))
        # electrode patches on the skin
        self.el = {}
        facets = self.m.boundaries["skin_exposed"]
        cen = self.m.p[:, self.m.facets[:, facets]].mean(axis=1).T
        @LinearForm
        def one(v, w):
            return 1.0 * v
        for s in L.SENSORS:
            if s.kind != "electrode" or not s.name.startswith("elec_L"):
                continue
            c = np.asarray(s.pos) * MM
            sel = facets[np.linalg.norm(cen - c, axis=1) <= PAD_R]
            if len(sel) == 0:
                raise RuntimeError(f"{s.name}: no skin_exposed facet within {PAD_R*1e3:.1f} mm of {s.pos}")
            fb = FacetBasis(self.m, ElementTetP1(), facets=sel)
            ell = asm(one, fb)
            self.el[s.name] = {"facets": sel, "ell": ell, "area": float(ell.sum()), "centre": c,
                               "skin_gap_mm": float(np.linalg.norm(cen - c, axis=1).min() / MM)}
        # deep reference node (ground)
        self.ref = int(np.argmin(self.m.p[1]))

    def system(self, f, edema=0.0, thermal=False):
        A = None
        for nm in self.regions:
            s = self._sigma(nm, f, edema)
            K = self._Kt[nm] if (thermal and nm in self._Kt) else self._K[nm]
            blk = s * K
            A = blk if A is None else A + blk
        return A.tocsc().astype(np.complex128)

    def _sigma(self, nm, f, edema):
        s = self.diel[nm][f]
        return s * (1.0 + edema) if nm in EDEMA_TISSUES else s

    def _solve(self, A, b, dirichlet):
        N = A.shape[0]
        free = np.setdiff1d(np.arange(N), dirichlet)
        u = np.zeros(N, dtype=np.complex128)
        u[free] = spla.spsolve(A[free][:, free], b[free])
        return u

    def _current(self, pair):
        a, b_ = self.el[pair[0]], self.el[pair[1]]
        return I_DRIVE * (a["ell"] / a["area"] - b_["ell"] / b_["area"])

    def potential(self, u, name):
        e = self.el[name]
        return complex(e["ell"] @ u / e["area"])

    def _dU(self, u, pair):
        return self.potential(u, pair[0]) - self.potential(u, pair[1])

    def bipolar(self, f, **kw):
        A = self.system(f, **kw)
        a, b_ = self.el[BIPOLAR[0]], self.el[BIPOLAR[1]]
        u = self._solve(A, self._current(BIPOLAR), np.array([self.ref]))
        Zt = self._dU(u, BIPOLAR) / I_DRIVE
        Zc = Z_CONTACT / a["area"] + Z_CONTACT / b_["area"]
        return {"Z_tissue": Zt, "Z_contact": Zc, "Z_total": Zt + Zc}, u

    def tetrapolar(self, f, **kw):
        """Local tetrapolar: drive DRIVE, sense SENSE, reference node grounded."""
        A = self.system(f, **kw)
        ground = np.array([self.ref])
        u_d = self._solve(A, self._current(DRIVE), ground)     # drive pair energised
        u_m = self._solve(A, self._current(SENSE), ground)     # reciprocal: sense pair energised
        Z = self._dU(u_d, SENSE) / I_DRIVE
        Z_recip = self._dU(u_m, DRIVE) / I_DRIVE
        contrib = {}
        for nm in self.regions:
            s = self._sigma(nm, f, kw.get("edema", 0.0))
            K = self._Kt[nm] if (kw.get("thermal") and nm in self._Kt) else self._K[nm]
            contrib[nm] = complex(s * (u_d @ (K @ u_m)) / I_DRIVE ** 2)
        return {"Z": Z, "Z_recip": Z_recip, "contrib": contrib,
                "reciprocity_residual": abs(Z_recip - Z) / abs(Z),
                "geselowitz_residual": abs(sum(contrib.values()) - Z) / abs(Z)}, (u_d, u_m)

    def lead_field(self, u_d, u_m, f):
        """Element-wise sensitivity density sigma grad u_d . grad u_m (real part), for rendering."""
        dens = np.zeros(self.m.t.shape[1])
        for nm, elems in self.regions.items():
            sb = Basis(self.m, ElementTetP1(), elements=elems)
            gd = sb.interpolate(u_d).grad; gm = sb.interpolate(u_m).grad
            s = self.diel[nm][f]
            val = (s * np.einsum("ijk,ijk->jk", gd, gm)).real.mean(axis=1)
            dens[elems] = val
        return dens


def cplx(z):
    return {"re": float(np.real(z)), "im": float(np.imag(z)), "abs": float(abs(z)),
            "phase_deg": float(np.degrees(np.angle(z)))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mesh", default=str(OUT / "periorbital_coarse.msh"))
    a = ap.parse_args()
    mesh_path = pathlib.Path(a.mesh)
    tag = "coarse" if "coarse" in mesh_path.name else "full"
    Tfile = OUT / f"thermal3d_T_{tag}.npy"
    T = np.load(Tfile) if Tfile.exists() else None

    bi = Bioimpedance3D(mesh_path, T)
    print(f"mesh {mesh_path.name}: {bi.m.p.shape[1]:,} nodes; setup {bi.load_time:.1f}s\nelectrodes:")
    for k, v in bi.el.items():
        print(f"  {k:18s} {v['area']*1e6:5.1f} mm2  {len(v['facets']):3d} facets  nearest facet {v['skin_gap_mm']:.2f} mm")
    print(f"drive {DRIVE[0]} -> {DRIVE[1]} ({I_DRIVE*1e3:.0f} mA), sense {SENSE[0]} - {SENSE[1]}; "
          f"bipolar {BIPOLAR[0]} -> {BIPOLAR[1]}")
    res = {"mesh": str(mesh_path), "I_drive_A": I_DRIVE, "pad_radius_mm": PAD_R * 1e3,
           "z_contact_ohm_m2": Z_CONTACT,
           "configuration": {"tetrapolar_local": {"drive": DRIVE, "sense": SENSE}, "bipolar": BIPOLAR,
                             "note": "single eye, reference node grounded; half-head mesh (x=0 Neumann)"},
           "electrodes": {k: {"area_mm2": v["area"] * 1e6, "n_facets": int(len(v["facets"])),
                              "nearest_facet_mm": v["skin_gap_mm"]} for k, v in bi.el.items()},
           "sweep": {}}
    print(f"\n{'f (kHz)':>8s} {'|Z| bip tissue':>14s} {'|Z| bip total':>13s} {'|Z| tetra':>10s} {'phase tetra':>12s} "
          f"{'recip.res':>10s} {'gesel.res':>10s}")
    for f in FREQS:
        t = time.time()
        b, _ = bi.bipolar(f)
        q, (u_d, u_m) = bi.tetrapolar(f)
        res["sweep"][str(int(f))] = {"bipolar": {k: cplx(v) for k, v in b.items()},
                                    "tetrapolar": cplx(q["Z"]), "tetrapolar_reciprocal": cplx(q["Z_recip"]),
                                    "tetrapolar_contrib": {k: cplx(v) for k, v in q["contrib"].items()},
                                    "reciprocity_residual": q["reciprocity_residual"],
                                    "geselowitz_residual": q["geselowitz_residual"]}
        print(f"{f/1e3:8.0f} {abs(b['Z_tissue']):14.2f} {abs(b['Z_total']):13.1f} {abs(q['Z']):10.3f} "
              f"{np.degrees(np.angle(q['Z'])):12.2f} {q['reciprocity_residual']:10.2e} {q['geselowitz_residual']:10.2e}"
              f"   ({time.time()-t:.1f}s)")
        if f == 50e3:
            u50 = (u_d, u_m); Z50 = q["Z"]; c50 = q["contrib"]; b50 = b
    # what the local tetrapolar measurement sees, at 50 kHz
    share = {k: float(v.real / Z50.real) for k, v in c50.items()}
    orbit_share = float(sum(share[k] for k in ORBIT_TISSUES if k in share))
    res["tissue_share_50kHz"] = share
    res["orbit_share_50kHz"] = orbit_share
    print("\nlocal tetrapolar 50 kHz: share of Re(Z) by tissue")
    for k, v in sorted(share.items(), key=lambda kv: -abs(kv[1])):
        if abs(v) > 0.005:
            print(f"  {k:18s} {100*v:6.1f} %")
    print(f"  {'orbit (fat+lid+globe)':18s} {100*orbit_share:6.1f} %")
    # edema sensitivity
    print("\nedema (orbital fat + eyelid conductivity x (1+delta)) at 50 kHz:")
    res["edema_50kHz"] = {}
    for d in (0.1, 0.2, 0.3):
        q, _ = bi.tetrapolar(50e3, edema=d)
        b, _ = bi.bipolar(50e3, edema=d)
        dZt = abs(q["Z"]) - abs(Z50)
        dZb = abs(b["Z_tissue"]) - abs(b50["Z_tissue"])
        res["edema_50kHz"][str(d)] = {
            "tetrapolar": cplx(q["Z"]), "dZ_abs_ohm": float(dZt), "dZ_abs_percent": float(100 * dZt / abs(Z50)),
            "dphase_deg": float(np.degrees(np.angle(q["Z"]) - np.angle(Z50))),
            "bipolar_tissue_dZ_abs_ohm": float(dZb),
            "bipolar_tissue_dZ_abs_percent": float(100 * dZb / abs(b50["Z_tissue"])),
            "bipolar_total_dZ_abs_percent": float(100 * dZb / abs(b50["Z_total"]))}
        print(f"  delta {d:.1f}: tetra |Z| {abs(q['Z']):.3f} Ohm ({dZt:+.4f}, {100*dZt/abs(Z50):+.2f} %), phase "
              f"{np.degrees(np.angle(q['Z'])):.2f} deg ({np.degrees(np.angle(q['Z'])-np.angle(Z50)):+.3f}); "
              f"bipolar tissue {100*dZb/abs(b50['Z_tissue']):+.2f} % (of total {100*dZb/abs(b50['Z_total']):+.2f} %)")
    # thermo-electrical coupling
    if T is not None:
        q, _ = bi.tetrapolar(50e3, thermal=True)
        res["thermal_coupling_50kHz"] = {"Z_with_sigma_T": cplx(q["Z"]),
                                         "dZ_abs_percent": float(100 * (abs(q["Z"]) - abs(Z50)) / abs(Z50))}
        print(f"\nsigma(T) coupling at 50 kHz: |Z| {abs(q['Z']):.3f} vs {abs(Z50):.3f} Ohm "
              f"({res['thermal_coupling_50kHz']['dZ_abs_percent']:+.2f} %)")
    dens = bi.lead_field(*u50, 50e3)
    np.save(OUT / f"bioimp_leadfield_{tag}.npy", dens)
    (OUT / f"bioimpedance3d_{tag}.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(f"\nwrote bioimpedance3d_{tag}.json + bioimp_leadfield_{tag}.npy")


if __name__ == "__main__":
    main()
