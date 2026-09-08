"""
Monte Carlo photon transport (MCX, GPU) on the voxelised periorbital anatomy.

Experiments
  ppg   MAX30102 over the superficial temporal artery, 660 and 880 nm.
        Source = LED disk on the skin, detector = photodiode 3.0 mm along the
        temple. Outputs: detected-photon partial path length per tissue, the
        share of attenuation owed to arterial blood (the PPG "signal fraction"),
        and the photon-measurement-density banana = Phi_source * Phi_detector
        (reciprocity), saved for rendering.
  led   SFH 4726AS 940 nm illuminator aimed at the pupil, 60 deg cone.
        Output: corneal irradiance per watt of LED optical power and the
        maximum LED power that meets the IEC 62471 corneal IR limit
        (100 W/m2 for exposure > 1000 s, 780-3000 nm).

Run (nrdi13, has pmcx 0.7.1):  python sim/forward_models/mcx_run.py [--nphoton 5e7]
"""
import sys, json, pathlib, time, argparse
import numpy as np
import pmcx

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "cad"))
import sensor_layout as L                                      # noqa: E402
OUT = HERE.parent / "out"
META = json.loads((OUT / "voxel_meta.json").read_text(encoding="utf-8"))
LABELS = {int(k): v for k, v in META["rois"]["temple"]["labels"].items()}
IEC_CORNEA_IR_W_M2 = 100.0        # IEC 62471 IR cornea/lens hazard, t > 1000 s
PD_SPACING_MM = 3.0               # MAX30102 LED-to-photodiode centre spacing (package 5.6 x 3.3 mm)
DERMAL_BLOOD_FRACTION = 0.03      # blood volume fraction of skin (dermal plexus), estimate 2-5 %
PULSE_FRACTION = 0.10             # fractional blood-volume change over a cardiac cycle


def roi(name):
    m = META["rois"][name]
    vol = np.load(OUT / f"labels_{name}.npy")
    return vol, np.array(m["origin_mm"]), m["res_mm"]


def to_grid(p_mm, origin, res):
    """mm -> MCX grid coordinates (1-based, voxel units)."""
    return (np.asarray(p_mm) - origin) / res + 1.0


def surface_entry(vol, p_grid, d):
    """March from p_grid along d until the first tissue voxel; return the air
    voxel just before it (source placed in air touching the skin)."""
    p = np.array(p_grid, float)
    d = np.asarray(d, float); d /= np.linalg.norm(d)
    # back out into air first
    while True:
        i = tuple(np.clip(np.floor(p - 1).astype(int), 0, np.array(vol.shape) - 1))
        if vol[i] == 0:
            break
        p -= d * 0.5
    while True:
        q = p + d * 0.5
        i = tuple(np.clip(np.floor(q - 1).astype(int), 0, np.array(vol.shape) - 1))
        if vol[i] != 0:
            return p
        p = q


def run(cfg, tag):
    t = time.time()
    res = pmcx.run(cfg)
    nd = int(np.asarray(res["detp"]).shape[-1]) if "detp" in res else 0
    print(f"  {tag}: {time.time()-t:5.1f} s, detected {nd:,} photons")
    return res


def ppg_experiment(nphoton):
    vol, origin, res = roi("temple")
    s = next(x for x in L.SENSORS if x.name == "ppg_L")
    axis = np.array(s.unit_axis())                           # (-1, 0, 0) into the skin
    src = surface_entry(vol, to_grid(s.pos, origin, res), axis)
    det_mm = np.array(s.pos) + np.array([0.0, PD_SPACING_MM, 0.0])   # along the temple
    det = surface_entry(vol, to_grid(det_mm, origin, res), axis)
    out = {"pd_spacing_mm": PD_SPACING_MM, "src_grid": src.tolist(), "det_grid": det.tolist(),
           "res_mm": res, "wavelengths": {}}
    for wl in (660, 880):
        prop = np.array(META["optical"][str(wl)], dtype=np.float32)
        base = dict(nphoton=int(nphoton), vol=vol, prop=prop, tstart=0, tend=5e-9, tstep=5e-9,
                    unitinmm=res, srctype="disk", srcparam1=[0.5 / res, 0, 0, 0],
                    isreflect=1, autopilot=1, gpuid=1, issrcfrom0=0, isnormalized=1)
        # forward: LED -> tissue, photodiode as detector
        cfg = dict(base, srcpos=src.tolist(), srcdir=axis.tolist(),
                   detpos=[[*det.tolist(), 1.0 / res]], savedetflag="dp", maxdetphoton=2_000_000)
        r_src = run(cfg, f"{wl} nm forward")
        # adjoint: photodiode as source (reciprocity) for the measurement-density banana
        cfg_adj = dict(base, srcpos=det.tolist(), srcdir=axis.tolist())
        r_det = run(cfg_adj, f"{wl} nm adjoint")
        phi_s = r_src["flux"][..., 0]; phi_d = r_det["flux"][..., 0]
        banana = phi_s * phi_d
        np.save(OUT / f"ppg_banana_{wl}.npy", banana.astype(np.float32))
        np.save(OUT / f"ppg_fluence_{wl}.npy", phi_s.astype(np.float32))

        detp = pmcx.detphoton(r_src["detp"], prop.shape[0] - 1, "dp")   # parse raw columns
        ppath = np.asarray(detp["ppath"]) * res                   # mm, per medium (labels 1..N)
        w = np.exp(-(ppath * prop[1:, 0]).sum(axis=1))           # Beer-Lambert weight
        n_det = ppath.shape[0]
        mean_pp = (ppath * w[:, None]).sum(axis=0) / w.sum()
        att = mean_pp * prop[1:, 0]                               # attenuation share per medium
        i_art = [i for i, nm in LABELS.items() if nm == "artery_temporal"][0] - 1
        i_skin = [i for i, nm in LABELS.items() if nm == "skin"][0] - 1
        mua_blood = float(prop[[i for i, nm in LABELS.items() if nm == "artery_temporal"][0], 0])
        # penetration: banana-weighted depth below the skin surface at the source
        X = (np.arange(vol.shape[0]) + 0.5) * res + origin[0]
        x_surf = origin[0] + (src[0] - 1.0) * res
        depth_w = (banana.sum(axis=(1, 2)) * np.clip(x_surf - X, 0, None)).sum() / banana.sum()
        # PPG pulse = dermal-plexus blood volume pulsation + the artery itself
        att_skin_blood = DERMAL_BLOOD_FRACTION * mua_blood * mean_pp[i_skin]
        d_att = PULSE_FRACTION * (att[i_art] + att_skin_blood)
        ac_dc = float(1 - np.exp(-d_att))
        out["wavelengths"][str(wl)] = {
            "n_detected": int(n_det),
            "detected_fraction": float(n_det / nphoton),
            "mean_partial_path_mm": {LABELS[i + 1]: float(mean_pp[i]) for i in range(len(mean_pp)) if mean_pp[i] > 0},
            "attenuation_share": {LABELS[i + 1]: float(att[i] / att.sum()) for i in range(len(att)) if att[i] > 0},
            "arterial_attenuation_share": float(att[i_art] / att.sum()),
            "dermal_blood_attenuation_share": float(att_skin_blood / (att.sum() + att_skin_blood)),
            "ac_dc_perfusion_index": ac_dc,
            "ac_dc_artery_only": float(1 - np.exp(-PULSE_FRACTION * att[i_art])),
            "banana_weighted_depth_mm": float(depth_w),
        }
        print(f"  {wl} nm: AC/DC {ac_dc*100:.2f} % (artery alone {100*(1-np.exp(-PULSE_FRACTION*att[i_art])):.3f} %), "
              f"banana depth {depth_w:.2f} mm, skin path {mean_pp[i_skin]:.1f} / artery {mean_pp[i_art]:.3f} of {mean_pp.sum():.1f} mm")
    # --- LED-to-photodiode spacing sweep (880 nm): dermal plexus vs artery sampling ---
    prop = np.array(META["optical"]["880"], dtype=np.float32)
    i_art = [i for i, nm in LABELS.items() if nm == "artery_temporal"][0] - 1
    i_skin = [i for i, nm in LABELS.items() if nm == "skin"][0] - 1
    mua_blood = float(prop[i_art + 1, 0])
    sweep = []
    for sp in (3.0, 6.0, 9.0, 12.0):
        det = surface_entry(vol, to_grid(np.array(s.pos) + np.array([0.0, sp, 0.0]), origin, res), axis)
        cfg = dict(nphoton=int(nphoton), vol=vol, prop=prop, tstart=0, tend=5e-9, tstep=5e-9,
                   unitinmm=res, srctype="disk", srcparam1=[0.5 / res, 0, 0, 0], isreflect=1,
                   autopilot=1, gpuid=1, issrcfrom0=0, isnormalized=1, srcpos=src.tolist(),
                   srcdir=axis.tolist(), detpos=[[*det.tolist(), 1.0 / res]], savedetflag="dp",
                   maxdetphoton=2_000_000)
        r = run(cfg, f"880 nm spacing {sp:.0f} mm")
        pp = np.asarray(pmcx.detphoton(r["detp"], prop.shape[0] - 1, "dp")["ppath"]) * res
        w = np.exp(-(pp * prop[1:, 0]).sum(axis=1)); mp = (pp * w[:, None]).sum(axis=0) / w.sum()
        att = mp * prop[1:, 0]; att_sb = DERMAL_BLOOD_FRACTION * mua_blood * mp[i_skin]
        sweep.append({"spacing_mm": sp, "detected_fraction": float(pp.shape[0] / nphoton),
                      "ac_dc_total": float(1 - np.exp(-PULSE_FRACTION * (att[i_art] + att_sb))),
                      "ac_dc_artery": float(1 - np.exp(-PULSE_FRACTION * att[i_art])),
                      "artery_share_of_pulse": float(att[i_art] / (att[i_art] + att_sb))})
        print(f"    spacing {sp:4.1f} mm: AC/DC {100*sweep[-1]['ac_dc_total']:.2f} %, artery share of pulse "
              f"{100*sweep[-1]['artery_share_of_pulse']:.1f} %, detected {100*sweep[-1]['detected_fraction']:.2f} %")
    out["spacing_sweep_880"] = sweep
    return out


def led_experiment(nphoton):
    vol, origin, res = roi("eye")
    s = next(x for x in L.SENSORS if x.name == "irled_L")
    axis = np.array(s.unit_axis())
    src = to_grid(s.pos, origin, res)
    prop = np.array(META["optical"]["940"], dtype=np.float32)
    cfg = dict(nphoton=int(nphoton), vol=vol, prop=prop, tstart=0, tend=5e-9, tstep=5e-9,
               unitinmm=res, srctype="cone", srcpos=src.tolist(), srcdir=axis.tolist(),
               srcparam1=[np.radians(s.fov_deg / 2), 0, 0, 0],
               isreflect=1, autopilot=1, gpuid=1, issrcfrom0=0, isnormalized=1)
    r = run(cfg, "940 nm LED cone")
    phi = r["flux"][..., 0]
    np.save(OUT / "led_fluence_940.npy", phi.astype(np.float32))
    # Absorbed power per tissue per watt of LED output (for the opto-thermal coupling):
    # MCX CW fluence [1/mm2 per W]; absorbed density = mua * phi; sum over voxel volume.
    TSTEP = 5e-9                     # MCX normalised flux is a fluence RATE; CW fluence = flux * tstep
    absorbed = {}
    for lid, nm in LABELS.items():
        if lid == 0:
            continue
        m = vol == lid
        if m.any():
            absorbed[nm] = float((phi[m] * prop[lid, 0]).sum() * res ** 3 * TSTEP)
    np.save(OUT / "led_absorbed_940.npy", (phi * prop[vol, 0]).astype(np.float32))   # W/mm3 per W
    # Corneal irradiance across the AIR gap is analytic (no scattering): a cone of
    # half-angle theta_h with uniform intensity gives E = P / (2 pi d^2 (1 - cos theta_h)).
    d = float(np.linalg.norm(np.array(s.pos) - np.array([L.IPD / 2, 0, 0]))) * 1e-3   # m
    th = np.radians(s.fov_deg / 2)
    e_per_w = 1.0 / (2 * np.pi * d ** 2 * (1 - np.cos(th)))                            # W/m2 per W
    e_mean = e_max = e_per_w
    out = {"led_to_cornea_mm": d * 1e3, "cone_half_angle_deg": s.fov_deg / 2,
           "irradiance_W_m2_per_W_led": e_per_w,
           "iec62471_cornea_limit_W_m2": IEC_CORNEA_IR_W_M2,
           "max_led_optical_power_mW": 1e3 * IEC_CORNEA_IR_W_M2 / e_per_w,
           "absorbed_fraction_per_tissue": absorbed,
           "absorbed_total_fraction": float(sum(absorbed.values()))}
    print(f"  cornea irradiance {e_per_w:.0f} W/m2 per W at {d*1e3:.1f} mm -> max LED optical power "
          f"{out['max_led_optical_power_mW']:.1f} mW for the IEC 62471 corneal limit; "
          f"absorbed in tissue {100*out['absorbed_total_fraction']:.1f} % "
          f"(eyelid {100*absorbed.get('eyelid',0):.1f} %, cornea {100*absorbed.get('cornea',0):.2f} %, sclera {100*absorbed.get('sclera',0):.2f} %)")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--nphoton", type=float, default=5e7)
    a = ap.parse_args()
    print("GPU:", pmcx.gpuinfo()[0]["name"])
    res = {"nphoton": a.nphoton, "ppg": ppg_experiment(a.nphoton), "led": led_experiment(a.nphoton)}
    (OUT / "optical_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("wrote optical_results.json + banana/fluence volumes")
