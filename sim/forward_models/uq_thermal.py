"""
Uncertainty quantification for the thermal model (ASME V&V 40 credibility):
global sensitivity (Sobol' indices, Saltelli scheme on a scrambled Sobol'
sequence) and a polynomial surrogate for the live digital twin.

Inputs (uniform over the stated ranges; ranges from tissue_db uncertainty
where available, otherwise the physiological/clinical envelope):
    choroid_frac      0.3 .. 1.0   choroidal perfusion fraction (disease handle)
    skin_perf_frac    0.5 .. 1.5   skin perfusion (vasomotion, sympathetic tone)
    eyelid_perf_frac  0.5 .. 3.0   eyelid perfusion multiplier (DB estimate)
    orbit_perf_frac   0.5 .. 3.0   orbital fat perfusion multiplier (DB estimate)
    T_amb            18   .. 28    room temperature [degC]
    h_skin            3.0 .. 8.0   skin convection [W/m2/K] (still air .. draught)
    E_tear           20   .. 60    tear evaporation heat loss [W/m2]
    k_sclera_scale    0.6 .. 1.7   sclera/choroid conductivity (0.58 vs Scott 1.0042)

Outputs: corneal apex T, medial canthus skin T, DCI thermopile reading aimed at
the nasal limbus (finding #3), DCI aimed at the canthus (current layout).

Surrogate: full quadratic polynomial in the 8 scaled inputs (45 terms) fitted by
least squares on the Saltelli samples, validated on a held-out 20 %. Written to
sim/out/surrogate_thermal.json for the dashboard (evaluates in microseconds).

Run (nrdi-env):  python sim/forward_models/uq_thermal.py [--n 64]
"""
import sys, json, pathlib, time, argparse, copy
import numpy as np
from scipy.stats import qmc

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cad"))
import thermal3d as T3                                       # noqa: E402
import sensor_layout as L                                    # noqa: E402
OUT = HERE.parent / "out"

PARAMS = [("choroid_frac", 0.3, 1.0), ("skin_perf_frac", 0.5, 1.5), ("eyelid_perf_frac", 0.5, 3.0),
          ("orbit_perf_frac", 0.5, 3.0), ("T_amb", 18.0, 28.0), ("h_skin", 3.0, 8.0),
          ("E_tear", 20.0, 60.0), ("k_sclera_scale", 0.6, 1.7)]
OUTPUTS = ["apex_C", "canthus_C", "dci_limbus_C", "dci_canthus_C"]


def make_probe(th):
    s = next(x for x in L.SENSORS if x.name == "thermo_L")
    pos = np.array(s.pos); limbus = np.array([L.IPD / 2 - 6.0, -1.5, 0.0])
    s2 = copy.copy(s); s2.name = "probe_limbus"; s2.axis = tuple((limbus - pos) / np.linalg.norm(limbus - pos))
    L.SENSORS.append(s2)
    canthus = (L.IPD / 2 - L.MEDIAL_CANTHUS_DX, 2.0, L.MEDIAL_CANTHUS_DZ)

    def evaluate(x):
        kw = dict(zip([p[0] for p in PARAMS], x))
        ks = kw.pop("k_sclera_scale")
        T = th.solve(k_scale={"sclera": ks}, **kw)
        return [th.corneal_apex(T), th.point_T(T, canthus)[0],
                th.thermopile_reading(T, "probe_limbus", fov_deg=5.0)[0],
                th.thermopile_reading(T, "thermo_L", fov_deg=5.0)[0]]
    return evaluate


def saltelli(n, d, seed=7):
    """A, B and the d AB_i matrices (Saltelli 2010), unit hypercube."""
    s = qmc.Sobol(d=2 * d, scramble=True, seed=seed).random(n)
    A, B = s[:, :d], s[:, d:]
    AB = [A.copy() for _ in range(d)]
    for i in range(d):
        AB[i][:, i] = B[:, i]
    return A, B, AB


def sobol_indices(fA, fB, fAB):
    """First-order (Saltelli 2010, Jansen) and total (Jansen) indices."""
    var = np.var(np.concatenate([fA, fB]), axis=0)
    S1 = np.array([np.mean(fB * (fABi - fA), axis=0) / var for fABi in fAB])
    ST = np.array([0.5 * np.mean((fA - fABi) ** 2, axis=0) / var for fABi in fAB])
    return S1, ST


def quad_features(X):
    n, d = X.shape
    cols = [np.ones(n)] + [X[:, i] for i in range(d)]
    for i in range(d):
        for j in range(i, d):
            cols.append(X[:, i] * X[:, j])
    return np.stack(cols, axis=1)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=64)
    a = ap.parse_args()
    d = len(PARAMS)
    lo = np.array([p[1] for p in PARAMS]); hi = np.array([p[2] for p in PARAMS])
    A, B, AB = saltelli(a.n, d)
    th = T3.Thermal3D(OUT / "periorbital_coarse.msh"); th._region_matrices()
    evaluate = make_probe(th)
    total = a.n * (d + 2)
    print(f"Sobol/Saltelli: {a.n} base samples x ({d}+2) = {total} solves on the coarse mesh")
    t0 = time.time()

    def run(M, tag):
        out = []
        for i, u in enumerate(M):
            out.append(evaluate(lo + u * (hi - lo)))
            if (i + 1) % 16 == 0:
                el = time.time() - t0
                print(f"  {tag} {i+1}/{len(M)}  ({el:.0f}s elapsed)", flush=True)
        return np.array(out)

    fA, fB = run(A, "A"), run(B, "B")
    fAB = [run(ABi, f"AB{i}") for i, ABi in enumerate(AB)]
    S1, ST = sobol_indices(fA, fB, fAB)

    # surrogate on all samples (scaled inputs in [-1, 1])
    X = np.concatenate([A, B] + AB); Y = np.concatenate([fA, fB] + fAB)
    Xs = 2 * X - 1
    rng = np.random.default_rng(0); idx = rng.permutation(len(X)); ntest = len(X) // 5
    te, tr = idx[:ntest], idx[ntest:]
    F = quad_features(Xs)
    coef, *_ = np.linalg.lstsq(F[tr], Y[tr], rcond=None)
    pred = F[te] @ coef
    rmse = np.sqrt(np.mean((pred - Y[te]) ** 2, axis=0)); rng_y = Y.max(axis=0) - Y.min(axis=0)

    print(f"\n{'parameter':18s}" + "".join(f"{o:>16s}" for o in OUTPUTS) + "   (S1 / ST)")
    for i, (nm, *_r) in enumerate(PARAMS):
        print(f"{nm:18s}" + "".join(f"{S1[i,j]:7.2f}/{ST[i,j]:5.2f}   " for j in range(len(OUTPUTS))))
    print(f"\noutput mean/sd over the input ranges:")
    for j, o in enumerate(OUTPUTS):
        print(f"  {o:14s} {Y[:, j].mean():7.2f} +- {Y[:, j].std():5.2f} C   surrogate RMSE {rmse[j]:.3f} C "
              f"({100*rmse[j]/rng_y[j]:.1f} % of range)")
    res = {"n_base": a.n, "n_solves": total, "params": PARAMS, "outputs": OUTPUTS,
           "S1": S1.tolist(), "ST": ST.tolist(),
           "output_stats": {o: {"mean": float(Y[:, j].mean()), "sd": float(Y[:, j].std()),
                                "p05": float(np.percentile(Y[:, j], 5)), "p95": float(np.percentile(Y[:, j], 95))}
                            for j, o in enumerate(OUTPUTS)},
           "surrogate": {"type": "quadratic polynomial, inputs scaled to [-1,1] via params ranges",
                         "feature_order": "1, x_i, x_i*x_j (i<=j)", "coef": coef.tolist(),
                         "rmse_holdout": rmse.tolist()},
           "time_s": time.time() - t0}
    (OUT / "surrogate_thermal.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    np.savez(OUT / "uq_thermal_samples.npz", X=X, Y=Y)
    print(f"\nwrote surrogate_thermal.json + uq_thermal_samples.npz  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
