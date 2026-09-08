"""
Pupil light reflex (PLR) forward model + camera projection for the pupillography
branch. Two parts:

1. Longtin-Milton delay-differential model of the iris (Longtin & Milton 1989,
   Math Biosci 90:183). Retinal illuminance phi drives the Edinger-Westphal
   nucleus through a Hill-type activation with latency tau; the iris constricts
   with a first-order time constant. Autonomic neuropathy is represented by a
   single "autonomic gain" g in [0, 1] that scales the parasympathetic drive
   (constriction amplitude/velocity) and lengthens latency, matching the
   published diabetic deficits (PROJECT_MASTER table P1-P5: amplitude 2.44 ->
   2.27 mm, MCV 7.24 -> 6.68 mm/s, latency longer, DV6 abnormal in 45 %).

2. Camera projection: the OV7670 sits on the bridge at a vergence angle to the
   pupil axis (sensor_layout: ~50 deg). A circular pupil of diameter D projects
   to an ellipse with minor axis D cos(theta); naive diameter = minor axis under-
   reads by (1 - cos theta). The correction factor is derived here so the branch
   reports true diameter.

Also: the stimulus problem. The BOM only has 940 nm illumination, which is
outside the photopic/melanopic sensitivity — the model shows zero reflex
unless a visible stimulus LED is added (design finding #2); the script
quantifies the required stimulus (log photopic lux) for a 2.3 mm constriction.

Run (nrdi-env):  python sim/forward_models/pupil.py
"""
import sys, json, pathlib, math
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cad"))
import sensor_layout as L                                   # noqa: E402
OUT = HERE.parent / "out"

# ---- Longtin-Milton parameters (healthy adult, from the 1989 paper's fits) ----
D0 = 6.0                          # mm, resting diameter (published healthy 6.02 +- 0.48)
AMP_MAX = 2.6                     # mm, saturating constriction for a 1 s flash
TAU_LAT = 0.25                    # s, afferent + efferent latency (healthy 200-280 ms)
TAU_C = 0.35                      # s, constriction time constant -> MCV ~ 7 mm/s
TAU_D = 1.20                      # s, redilation time constant (slower, sympathetic)
PHI_HALF = 20.0                   # lux-equivalent illuminance for half-maximal drive
HILL_N = 1.5
# Calibration: healthy amplitude 2.44 mm and MCV 7.24 mm/s at the reference flash
# (PMID 8462391); the diabetic group there (2.27 mm, 6.68 mm/s) corresponds to g ~ 0.92.


def simulate(phi_of_t, g=1.0, dt=0.002, t_end=6.0, d0=D0):
    """Integrate the DDE by history buffer. g = autonomic gain (1 healthy).
    Returns t, D(t) [mm]."""
    n = int(t_end / dt)
    t = np.arange(n) * dt
    D = np.full(n, d0)
    lat = TAU_LAT * (1.0 + 0.6 * (1.0 - g))       # neuropathy lengthens latency (up to +60 %)
    k_lat = int(round(lat / dt))
    for i in range(1, n):
        phi = phi_of_t(t[i - k_lat]) if i - k_lat >= 0 else phi_of_t(0.0)
        drive = g * phi ** HILL_N / (PHI_HALF ** HILL_N + phi ** HILL_N)      # 0..g
        D_target = d0 - AMP_MAX * drive
        tau = TAU_C if D_target < D[i - 1] else TAU_D * (1.0 + 0.5 * (1.0 - g))
        D[i] = D[i - 1] + dt * (D_target - D[i - 1]) / tau
    return t, D


def flash(lux, t_on=1.0, dur=1.0, background=5.0):
    return lambda tt: lux if t_on <= tt < t_on + dur else background


def metrics(t, D, t_on=1.0):
    """Clinical PLR parameters: baseline, amplitude, latency, max constriction
    velocity (MCV), average dilation velocity, DV6 (dilation velocity at 6 s)."""
    base = D[t < t_on].mean()
    i_on = np.searchsorted(t, t_on)
    dD = np.gradient(D, t)
    i_min = int(np.argmin(D))
    amp = base - D[i_min]
    # latency: first time after stimulus where constriction velocity exceeds 0.3 mm/s
    moving = np.where((dD[i_on:] < -0.3))[0]
    lat = float(t[i_on + moving[0]] - t_on) if len(moving) else float("nan")
    mcv = float(-dD[i_on:i_min + 1].min()) if i_min > i_on else 0.0
    # redilation: mean velocity over the first 1.5 s after the minimum
    i_r1 = min(len(t) - 1, i_min + int(1.5 / (t[1] - t[0])))
    adv = float((D[i_r1] - D[i_min]) / (t[i_r1] - t[i_min])) if i_r1 > i_min else 0.0
    i6 = min(len(t) - 1, int(np.searchsorted(t, 6.0)))
    dv6 = float(dD[i6 - 1])
    return {"baseline_mm": float(base), "amplitude_mm": float(amp), "latency_s": lat,
            "max_constriction_velocity_mm_s": mcv, "avg_dilation_velocity_mm_s": adv, "dv6_mm_s": dv6,
            "min_diameter_mm": float(D[i_min])}


def camera_projection():
    """Off-axis viewing: ellipse minor/major axis ratio = cos(vergence)."""
    cam = next(s for s in L.SENSORS if s.name == "cam_L")
    pupil = np.array([L.IPD / 2, 0.0, 0.0])
    view = pupil - np.array(cam.pos)
    d = float(np.linalg.norm(view))
    theta = math.degrees(math.acos(abs(view[1]) / d))         # angle to the pupil axis (+Y)
    ratio = math.cos(math.radians(theta))
    # OV7670: 640x480, 1/6" sensor 3.6 x 2.7 mm; with a 25 deg FOV lens the pixel pitch at the
    # working distance is the diameter quantisation of a naive measurement
    fov = cam.fov_deg
    px_mm = 2 * d * math.tan(math.radians(fov / 2)) / 640.0
    return {"working_distance_mm": d, "vergence_deg": theta, "minor_over_major": ratio,
            "naive_underread_percent": 100 * (1 - ratio), "pixel_pitch_at_pupil_mm": px_mm,
            "correction": "true D = major axis of the fitted ellipse (or minor / cos theta)"}


def main():
    t, Dh = simulate(flash(300.0), g=1.0)
    t, Dn = simulate(flash(300.0), g=0.92)
    mh, mn = metrics(t, Dh), metrics(t, Dn)
    print(f"{'metric':32s} {'healthy':>9s} {'g=0.92':>9s}   published (healthy -> diabetic)")
    pub = {"amplitude_mm": "2.44 -> 2.27", "max_constriction_velocity_mm_s": "7.24 -> 6.68",
           "latency_s": "longer in DAN", "avg_dilation_velocity_mm_s": "slower (P4)", "dv6_mm_s": "abnormal 45 % (P5)"}
    for k in ("baseline_mm", "amplitude_mm", "latency_s", "max_constriction_velocity_mm_s",
              "avg_dilation_velocity_mm_s", "dv6_mm_s"):
        print(f"{k:32s} {mh[k]:9.3f} {mn[k]:9.3f}   {pub.get(k, '')}")
    # gain sweep: the concept the fusion model must recover
    sweep = []
    for g in np.linspace(1.0, 0.3, 8):
        _, D = simulate(flash(300.0), g=g)
        m = metrics(t, D)
        sweep.append({"gain": float(g), **m})
    # stimulus requirement: 940 nm has ~zero photopic efficacy
    stim = []
    for lux in (0.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0):
        _, D = simulate(flash(lux), g=1.0)
        stim.append({"stimulus_lux": lux, "amplitude_mm": metrics(t, D)["amplitude_mm"]})
    cam = camera_projection()
    print(f"\ncamera: distance {cam['working_distance_mm']:.1f} mm, vergence {cam['vergence_deg']:.1f} deg, "
          f"naive diameter under-reads by {cam['naive_underread_percent']:.1f} %, pixel pitch {cam['pixel_pitch_at_pupil_mm']*1e3:.0f} um")
    print("stimulus -> amplitude: " + ", ".join(f"{s['stimulus_lux']:.0f} lx: {s['amplitude_mm']:.2f} mm" for s in stim))
    res = {"model": "Longtin-Milton DDE, autonomic gain g", "healthy": mh, "gain_0.92": mn,
           "gain_sweep": sweep, "stimulus_sweep": stim, "camera": cam,
           "traces": {"t": t[::10].tolist(), "D_healthy": Dh[::10].tolist(), "D_gain_0.92": Dn[::10].tolist()}}
    (OUT / "pupil_model.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("wrote pupil_model.json")


if __name__ == "__main__":
    main()
