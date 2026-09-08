"""
Single source of truth for device geometry and sensor pose.

Everything downstream reads from here:
  - frame.py            builds the CAD solid from these numbers
  - thermal model       reads thermopile position + field of view
  - optical model       reads LED/photodiode separation and optical axes
  - bioimpedance model  reads electrode positions
  - demonstrator        reads the same JSON as the solvers

Coordinate system (right-handed, millimetres):
    origin  = midpoint between the two pupils, at the corneal plane
    +X      = wearer's LEFT   (subject's own left)
    +Y      = anterior, away from the face (toward the lens)
    +Z      = superior (up)

Anthropometry is adult mean unless noted. Sources are cited per value so the
paper can state provenance rather than "we picked reasonable numbers".
"""
from dataclasses import dataclass, field, asdict
from typing import Tuple
import math

# ----------------------------------------------------------------------------
# Head / face anthropometry
# ----------------------------------------------------------------------------
IPD                 = 63.0   # interpupillary distance, adult mean (Dodgson 2004)
VERTEX_DISTANCE     = 13.0   # cornea -> back of lens, standard fitting value
PANTOSCOPIC_TILT    = 9.0    # deg, lens plane tilted bottom-in
FACE_FORM_WRAP      = 6.0    # deg, frame front wrap around the face
MEDIAL_CANTHUS_DX   = 15.0   # pupil centre -> medial canthus, lateral (Farkas)
MEDIAL_CANTHUS_DZ   = -2.0   # slightly inferior to pupil axis
TEMPLE_ARTERY_X     = 68.0   # superficial temporal artery, lateral from midline
TEMPLE_ARTERY_Y     = -78.0  # posterior from corneal plane, anterior to tragus
TEMPLE_ARTERY_Z     = 12.0   # superior

# ----------------------------------------------------------------------------
# Frame geometry
# ----------------------------------------------------------------------------
LENS_W              = 52.0
LENS_H              = 40.0
BRIDGE_W            = 18.0   # DBL, distance between lenses
RIM_T               = 3.2    # rim cross-section thickness (anterior-posterior)
RIM_W               = 4.0    # rim cross-section width (radial)
TEMPLE_LEN          = 145.0
TEMPLE_SECTION      = (12.0, 4.5)   # w x h, houses the 12x60mm PCB
EARHOOK_DROP        = 32.0

# PCB / battery envelopes (from the bill of materials)
PCB_SIZE            = (60.0, 12.0, 1.6)     # 4-layer, per temple arm
BATTERY_SIZE        = (34.0, 12.0, 5.0)     # 3.7 V 250 mAh LiPo


@dataclass
class Sensor:
    """A sensor with a pose. `axis` is the outward optical/sensing direction."""
    name:    str
    kind:    str
    part:    str
    pos:     Tuple[float, float, float]
    axis:    Tuple[float, float, float]
    fov_deg: float = 0.0          # full cone angle; 0 = not a cone sensor
    note:    str = ""

    def unit_axis(self):
        n = math.sqrt(sum(c * c for c in self.axis))
        return tuple(c / n for c in self.axis)


def _aim(frm, to):
    """Unit vector from `frm` toward `to`."""
    v = tuple(b - a for a, b in zip(frm, to))
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def build_sensors():
    S = []
    half = IPD / 2.0

    for side, sgn in (("L", +1.0), ("R", -1.0)):
        pupil   = (sgn * half, 0.0, 0.0)
        canthus = (sgn * (half - MEDIAL_CANTHUS_DX), 2.0, MEDIAL_CANTHUS_DZ)

        # --- Pupillography camera, on the bridge, aimed at the pupil ---------
        # Sits just lateral of the bridge centre, forward of the corneal plane.
        cam_pos = (sgn * (BRIDGE_W / 2.0 + 2.5), VERTEX_DISTANCE + 4.0, 6.0)
        S.append(Sensor(
            f"cam_{side}", "camera", "OV7670",
            cam_pos, _aim(cam_pos, pupil), fov_deg=25.0,
            note="DVP/SCCB. Working distance set by vertex distance + bridge offset."))

        # --- 940 nm IR illuminator, flanking the camera ----------------------
        led_pos = (sgn * (BRIDGE_W / 2.0 + 8.0), VERTEX_DISTANCE + 3.0, 6.0)
        S.append(Sensor(
            f"irled_{side}", "ir_led", "SFH 4726AS",
            led_pos, _aim(led_pos, pupil), fov_deg=60.0,
            note="940 nm. Invisible to the eye so it does not itself drive the reflex. "
                 "Irradiance at the corneal plane must be checked against IEC 62471."))

        # --- Thermopile at the medial canthus --------------------------------
        # Looks medially and posteriorly at the skin over the angular artery.
        th_pos = (sgn * (half - MEDIAL_CANTHUS_DX + 1.0), VERTEX_DISTANCE - 2.0,
                  MEDIAL_CANTHUS_DZ + 1.0)
        S.append(Sensor(
            f"thermo_{side}", "thermopile", "MLX90614ESF-BAA",
            th_pos, _aim(th_pos, canthus), fov_deg=90.0,
            note="I2C, address collision -> TCA9548A. 90 deg FOV means the spot "
                 "grows fast with standoff; this drives the standoff constraint."))

        # --- PPG over the superficial temporal artery ------------------------
        ppg_pos = (sgn * TEMPLE_ARTERY_X, TEMPLE_ARTERY_Y, TEMPLE_ARTERY_Z)
        S.append(Sensor(
            f"ppg_{side}", "ppg", "MAX30102",
            ppg_pos, (sgn * -1.0, 0.0, 0.0), fov_deg=0.0,
            note="660 nm + 880 nm. Faces medially into the skin. The LED-photodiode "
                 "separation inside this package is what the Monte Carlo model sizes."))

        # --- Bioimpedance electrodes: local tetrapolar set, 4 per eye ---------
        # v3 (2026-09-04). Contacts only where eyewear already touches the face:
        #   brow bumper (silicone strip on the posterior face of the upper rim)
        #       carries brow_med and brow_lat; the supraorbital ridge skin is
        #       modelled at y ~ +10 (medial) .. +4 (lateral) at z = +21
        #   nose pad carries nose_sup and nose_inf on the nasal sidewall
        #       (skin at y ~ +11, facing the pad medially)
        # Drive brow_lat -> nose_inf, sense brow_med -> nose_sup: the current
        # path crosses the orbit, which is what the edema measurement needs
        # (finding #6). No sprung arms: nothing protrudes above or below the frame.
        n_ax = (-sgn * 0.944, 0.33, 0.0)             # into the nasal sidewall
        for tag, pos, ax in (("brow_med", (sgn * (half - 10.0), 10.0, LENS_H / 2.0 + 1.0), (0.0, -1.0, 0.0)),
                             ("brow_lat", (sgn * (half + 10.0),  4.1, LENS_H / 2.0 + 1.0), (0.0, -1.0, 0.0)),
                             ("nose_sup", (sgn * 7.5, 10.8, -2.0), n_ax),
                             ("nose_inf", (sgn * 8.5, 11.3, -10.0), n_ax)):
            S.append(Sensor(
                f"elec_{side}_{tag}", "electrode", "Ag/AgCl",
                pos, ax,
                note="Dry Ag/AgCl-coated silicone contact, 6 mm. Local tetrapolar set "
                     "per eye (drive brow_lat->nose_inf, sense brow_med->nose_sup)."))

    # --- Non-sensing components ---------------------------------------------
    S.append(Sensor("mcu", "processor", "ESP32-S3",
                    (TEMPLE_ARTERY_X, -40.0, TEMPLE_ARTERY_Z), (1.0, 0.0, 0.0),
                    note="Right temple PCB."))
    S.append(Sensor("haptic", "actuator", "LRA C10-100",
                    (TEMPLE_ARTERY_X, -120.0, 4.0), (-1.0, 0.0, 0.0),
                    note="Near the ear hook, where skin coupling is best."))
    S.append(Sensor("battery", "power", "LiPo 250mAh",
                    (-TEMPLE_ARTERY_X, -100.0, TEMPLE_ARTERY_Z), (-1.0, 0.0, 0.0),
                    note="Left temple, counterweights the right-side PCB."))
    return S


SENSORS = build_sensors()


def derived_metrics():
    """Quantities the physics models need, computed rather than assumed."""
    half = IPD / 2.0
    out = {}

    cam = next(s for s in SENSORS if s.name == "cam_L")
    pupil = (half, 0.0, 0.0)
    d = math.dist(cam.pos, pupil)
    out["camera_working_distance_mm"] = round(d, 2)
    # angle between the camera axis and the straight-ahead (+Y) direction
    ax = cam.unit_axis()
    out["camera_vergence_deg"] = round(math.degrees(math.acos(abs(ax[1]))), 2)

    th = next(s for s in SENSORS if s.name == "thermo_L")
    canthus = (half - MEDIAL_CANTHUS_DX, 2.0, MEDIAL_CANTHUS_DZ)
    standoff = math.dist(th.pos, canthus)
    out["thermopile_standoff_mm"] = round(standoff, 2)
    # 90 deg full cone -> spot diameter equals 2 * standoff * tan(45 deg)
    out["thermopile_spot_diameter_mm"] = round(
        2.0 * standoff * math.tan(math.radians(th.fov_deg / 2.0)), 2)

    led = next(s for s in SENSORS if s.name == "irled_L")
    out["led_to_cornea_mm"] = round(math.dist(led.pos, pupil), 2)

    eL = next(s for s in SENSORS if s.name == "elec_L_brow_lat")
    eL2 = next(s for s in SENSORS if s.name == "elec_L_nose_inf")
    out["electrode_drive_separation_mm"] = round(math.dist(eL.pos, eL2.pos), 2)
    eL = next(s for s in SENSORS if s.name == "elec_L_brow_med")
    eL2 = next(s for s in SENSORS if s.name == "elec_L_nose_sup")
    out["electrode_sense_separation_mm"] = round(math.dist(eL.pos, eL2.pos), 2)
    return out


def to_dict():
    return {
        "coordinate_system": {
            "origin": "midpoint between pupils, at the corneal plane",
            "x": "wearer's left", "y": "anterior", "z": "superior",
            "units": "mm",
        },
        "anthropometry": {
            "ipd_mm": IPD, "vertex_distance_mm": VERTEX_DISTANCE,
            "pantoscopic_tilt_deg": PANTOSCOPIC_TILT,
            "face_form_wrap_deg": FACE_FORM_WRAP,
            "medial_canthus_offset_mm": MEDIAL_CANTHUS_DX,
        },
        "frame": {
            "lens_w_mm": LENS_W, "lens_h_mm": LENS_H, "bridge_w_mm": BRIDGE_W,
            "rim_thickness_mm": RIM_T, "temple_length_mm": TEMPLE_LEN,
            "pcb_mm": list(PCB_SIZE), "battery_mm": list(BATTERY_SIZE),
        },
        "derived": derived_metrics(),
        "sensors": [asdict(s) for s in SENSORS],
    }


if __name__ == "__main__":
    import json, pathlib
    d = to_dict()
    out = pathlib.Path(__file__).resolve().parents[1] / "out" / "sensor_layout.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(d, indent=2), encoding="utf-8")

    print(f"{len(SENSORS)} components placed -> {out}")
    print("\nderived constraints the physics models will consume:")
    for k, v in d["derived"].items():
        print(f"  {k:34s} {v}")
    print("\nby modality:")
    kinds = {}
    for s in SENSORS:
        kinds.setdefault(s.kind, []).append(s.name)
    for k, names in sorted(kinds.items()):
        print(f"  {k:12s} {len(names)}  {', '.join(names)}")
