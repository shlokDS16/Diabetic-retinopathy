"""
Steady-state Pennes bioheat model of the human eye, 2D axisymmetric.

    div(k grad T) + wb*rho_b*cb*(Ta - T) + Qm = 0

Solved on the Scott-style schematic eye mesh from eye_mesh.py. In the
axisymmetric weak form every integral carries a factor r from the volume
element 2*pi*r dr dz.

Boundary conditions follow the standard ocular thermal formulation:
  anterior  (cornea, exposed to air)
        -k dT/dn = h_conv (T - T_amb) + h_rad (T - T_amb) + E_evap
     radiation is linearised about the surface temperature,
        h_rad = 4 * eps * sigma * T0^3
  posterior (sclera, backed by choroidal blood)
        -k dT/dn = h_bl (T - T_body)

h_bl is the physiological handle for this project: choroidal perfusion falls
in diabetic retinopathy, so h_bl is the parameter we sweep to predict the
ocular surface temperature deficit the wearable would have to detect.

    PROVENANCE: tissue properties and BC coefficients below are the standard
    values reused across the ocular thermal literature. They MUST be checked
    against Scott (1988) and Ng & Ooi before publication. Marked TO-VERIFY.

Run:  python sim/forward_models/thermal.py
"""
import pathlib, json
import numpy as np
from skfem import (Basis, FacetBasis, ElementTriP1, BilinearForm, LinearForm,
                   asm, solve, condense)
from skfem.helpers import dot, grad
from skfem import Mesh

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE.parent / "out"

# --- material properties, SI ------------------------------------  TO-VERIFY
K = {"cornea": 0.58, "aqueous": 0.58, "lens": 0.40,
     "vitreous": 0.603, "sclera": 1.0042}          # W/m/K

# --- boundary coefficients ---------------------------------------  TO-VERIFY
T_AMB   = 25.0      # degC, ambient
T_BODY  = 37.0      # degC, core
H_CONV  = 10.0      # W/m2/K, corneal convection
EPS     = 0.975     # corneal emissivity
SIGMA   = 5.670e-8  # W/m2/K4
E_EVAP  = 40.0      # W/m2, tear evaporation
H_BL_0  = 65.0      # W/m2/K, baseline choroidal blood convection

MM = 1e-3           # mesh is in mm


def h_radiation(T0_C=34.0):
    """Linearised radiative coefficient about a surface temperature."""
    return 4.0 * EPS * SIGMA * (T0_C + 273.15) ** 3


def solve_eye(h_bl=H_BL_0, t_amb=T_AMB, mesh_path=None, mesh=None):
    m = mesh if mesh is not None else Mesh.load(str(mesh_path or OUT / "eye_axisym.msh"))
    basis = Basis(m, ElementTriP1())
    h_rad = h_radiation()
    h_ant = H_CONV + h_rad

    @BilinearForm
    def conduction(u, v, w):
        return w["k"] * dot(grad(u), grad(v)) * w.x[0]

    @BilinearForm
    def robin(u, v, w):
        return w["h"] * u * v * w.x[0]

    @LinearForm
    def robin_rhs(v, w):
        return w["q"] * v * w.x[0]

    # --- conduction, assembled per subdomain so k is exact per region -------
    A = None
    for name, kval in K.items():
        if name not in m.subdomains:
            raise RuntimeError(f"subdomain '{name}' missing from mesh")
        sb = Basis(m, ElementTriP1(), elements=m.subdomains[name])
        blk = asm(conduction, sb, k=kval)
        A = blk if A is None else A + blk

    # --- Robin boundaries ---------------------------------------------------
    fb_ant = FacetBasis(m, ElementTriP1(), facets=m.boundaries["anterior"])
    fb_post = FacetBasis(m, ElementTriP1(), facets=m.boundaries["posterior"])

    A = A + asm(robin, fb_ant, h=h_ant) + asm(robin, fb_post, h=h_bl)
    b = (asm(robin_rhs, fb_ant, q=h_ant * t_amb - E_EVAP)
         + asm(robin_rhs, fb_post, q=h_bl * T_BODY))

    T = solve(A, b)
    return m, basis, T


def corneal_apex_temperature(m, T):
    """Temperature at the corneal apex: r = 0, minimum z."""
    r, z = m.p[0], m.p[1]
    on_axis = np.where(r < 1e-9)[0]
    apex = on_axis[np.argmin(z[on_axis])]
    return float(T[apex]), float(r[apex]), float(z[apex])


def surface_profile(m, T, max_z=4.0):
    """Anterior surface temperature vs radial position, for the OST figure."""
    facets = m.boundaries["anterior"]
    nodes = np.unique(m.facets[:, facets])
    r, z = m.p[0, nodes], m.p[1, nodes]
    keep = z < max_z
    order = np.argsort(r[keep])
    return r[keep][order], T[nodes][keep][order]


if __name__ == "__main__":
    # scale mesh mm -> m before solving
    m0 = Mesh.load(str(OUT / "eye_axisym.msh"))
    m0 = m0.scaled(MM)

    print(f"mesh: {m0.p.shape[1]} nodes, {m0.t.shape[1]} elements")
    print(f"subdomains: {sorted(m0.subdomains)}")
    print(f"boundaries: {sorted(m0.boundaries)}")
    print(f"h_rad (linearised) = {h_radiation():.2f} W/m2/K\n")

    m, basis, T = solve_eye(mesh=m0)
    apex_T, _, _ = corneal_apex_temperature(m, T)

    print(f"{'':22s} {'T_apex (degC)':>14s}")
    print(f"{'baseline':22s} {apex_T:14.3f}")
    print(f"  range over eye: {T.min():.2f} to {T.max():.2f} degC")

    # ---- VERIFICATION: published corneal centre temperature is ~34-35 degC
    ok = 33.0 <= apex_T <= 36.0
    print(f"\nVERIFICATION vs published corneal surface temperature (34-35 degC): "
          f"{'PASS' if ok else 'FAIL'}")

    # ---- choroidal perfusion sweep -> predicted OST deficit ---------------
    print(f"\n{'h_bl (W/m2/K)':>14s} {'T_apex':>9s} {'dT vs baseline':>16s}")
    rows = []
    for frac in (1.00, 0.90, 0.80, 0.70, 0.60, 0.50):
        _, _, Ts = solve_eye(h_bl=H_BL_0 * frac, mesh=m0)
        Ta, _, _ = corneal_apex_temperature(m0, Ts)
        rows.append({"perfusion_fraction": frac, "h_bl": H_BL_0 * frac,
                     "T_apex_C": Ta, "delta_C": Ta - apex_T})
        print(f"{H_BL_0*frac:14.1f} {Ta:9.3f} {Ta-apex_T:16.3f}")

    (OUT / "thermal_sweep.json").write_text(json.dumps({
        "baseline_T_apex_C": apex_T,
        "h_rad_linearised": h_radiation(),
        "verification_pass": bool(ok),
        "sweep": rows,
    }, indent=2), encoding="utf-8")
    np.save(OUT / "thermal_T.npy", T)
    print(f"\nwrote {OUT/'thermal_sweep.json'} and thermal_T.npy")
