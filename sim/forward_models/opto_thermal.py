"""
Opto-thermal coupling: absorbed 940 nm illuminator power (MCX, W/mm^3 per W)
becomes a volumetric heat source in the Pennes model.

    q_led(x) = P_led * a(x),   a from sim/out/led_absorbed_940.npy (eye ROI grid)

Reports the steady temperature rise at the corneal apex, the eyelid skin and
the medial canthus for the IEC-limited LED power (from optical_results.json)
and for 20 mW continuous. Answers "does the illuminator confound the
thermopile?" with a number.

Run (nrdi-env):  python sim/forward_models/opto_thermal.py
"""
import sys, json, pathlib
import numpy as np
from scipy.interpolate import RegularGridInterpolator

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parent / "cad"))
import thermal3d as T3                                       # noqa: E402
import sensor_layout as L                                    # noqa: E402

OUT = HERE.parent / "out"


def main():
    meta = json.loads((OUT / "voxel_meta.json").read_text(encoding="utf-8"))["rois"]["eye"]
    a = np.load(OUT / "led_absorbed_940.npy")                # W/mm3 per W, MCX flux*mua (rate units)
    a = a * 5e-9                                             # CW: flux is a rate -> multiply by tstep
    origin, res = np.array(meta["origin_mm"]), meta["res_mm"]
    axes = [origin[i] + (np.arange(a.shape[i]) + 0.5) * res for i in range(3)]
    interp = RegularGridInterpolator(axes, a, bounds_error=False, fill_value=0.0)

    th = T3.Thermal3D(OUT / "periorbital_coarse.msh")
    th._region_matrices()
    P = th.m.p.T / T3.MM                                     # nodes in mm
    a_nodes = interp(P) * 1e9                                # W/mm3 -> W/m3 per W of LED power
    opt = json.loads((OUT / "optical_results.json").read_text(encoding="utf-8"))["led"]
    p_iec = opt["max_led_optical_power_mW"] * 1e-3
    print(f"absorbed-power field: max {a_nodes.max():.3e} W/m3 per W; IEC-limited LED power {p_iec*1e3:.1f} mW")

    T0 = th.solve()
    canthus = (L.IPD / 2 - L.MEDIAL_CANTHUS_DX, 2.0, L.MEDIAL_CANTHUS_DZ)
    lid = (L.IPD / 2, -0.5, 9.0)
    out = {"led_power_mW": {}, "absorbed_field_max_W_m3_per_W": float(a_nodes.max())}
    print(f"\n{'LED power':>10s} {'dT apex':>9s} {'dT lid':>8s} {'dT canthus':>11s} {'dT max':>8s}")
    for p in (0.005, 0.020, p_iec):
        T = th.solve(extra_source=p * a_nodes)
        d = T - T0
        row = {"apex": th.corneal_apex(T) - th.corneal_apex(T0),
               "lid": th.point_T(T, lid)[0] - th.point_T(T0, lid)[0],
               "canthus": th.point_T(T, canthus)[0] - th.point_T(T0, canthus)[0],
               "max": float(d.max())}
        out["led_power_mW"][f"{p*1e3:.1f}"] = row
        print(f"{p*1e3:8.1f} mW {row['apex']:9.3f} {row['lid']:8.3f} {row['canthus']:11.3f} {row['max']:8.3f}")
    (OUT / "opto_thermal.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("\nwrote opto_thermal.json")


if __name__ == "__main__":
    main()
