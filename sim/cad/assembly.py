"""
Per-part assembly of the NRDI glasses for exploded views and rendering.

frame.py fuses everything into one solid (what the FEM/voxeliser want).
This file keeps every body SEPARATE, with a unique label and a colour, and
adds the electronics that frame.py omits, so that a Three.js exploded view
and a Blender render can address each part by name.

Geometry sources
    sensor_layout.py   every pose (never edit numbers here, edit there)
    frame.py           rim / bridge / nosepad / temple / sensor-housing builders

Exports (all in project-native coordinates: millimetres, +Z up, +Y anterior)
    out/glasses.glb          one mesh + one node per part, node name == label
    out/glasses.step         same Compound as a STEP assembly
    out/parts_stl/<label>.stl  one STL per part (voxeliser, fallback importer)
    out/parts.json           explode manifest: category, side, centroid, bbox,
                             explode direction/distance/order, callout, P/N
    out/glasses_occt.glb     (reference only) build123d's own export_gltf --
                             it writes one primitive PER FACE, so it is NOT
                             used by the viewer; kept so the two can be diffed

Overlap policy
    Electronics are placed inside the temple envelope at their sensor_layout
    poses WITHOUT cutting pockets. The temple section is 12 x 4.5 mm and the
    PCB is 12 mm wide, so a pocket would leave no side wall; the ESP32 module
    (18 mm) is wider than the temple anyway. Temple<->electronics overlap is
    therefore accepted (hidden inside the arm when assembled). Electronics do
    not overlap EACH OTHER, and lenses/sensor pods only touch the frame.

Run:  python sim/cad/assembly.py
"""
import sys, pathlib, json, math

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import numpy as np
from build123d import (
    Box, Cylinder, RectangleRounded, Plane, Pos, Vector, Align, Color,
    Compound, extrude, export_step, export_stl, export_gltf,
)
import sensor_layout as L
import frame as F

OUT = F.OUT
STL_DIR = OUT / "parts_stl"
STL_DIR.mkdir(parents=True, exist_ok=True)
WEB_DIR = OUT / "parts_stl_web"
WEB_DIR.mkdir(parents=True, exist_ok=True)

NOTES = [
    "sensor_layout.py places 'mcu' at x=+68 and 'battery' at x=-68. Under its own "
    "convention (+X = wearer's left) that is temple_L for the PCB and temple_R for "
    "the battery, although the notes there say 'Right temple PCB' / 'Left temple'. "
    "Poses are the source of truth, so the bodies sit where the numbers put them; "
    "labels pcb_R / battery follow the bill of materials; 'side' is geometric.",
    "Lenses are cut around the nasal sensor pods (camera, IR LED, thermopile + its "
    "leads) and the temple hinge corner because those poses lie in the lens plane.",
    "v3 contacts (2026-09-04): no electrode arms. browpad_L/R is a silicone strip on "
    "the posterior face of the upper rim carrying elec_*_brow_med/brow_lat; nosepad_L/R "
    "sits on the nasal sidewall through elec_*_nose_sup/nose_inf. The electrode discs "
    "are embedded in those pads (disc body inside the pad, face + button exposed). "
    "Rims are centred on FRONT_Y (y 11.4..14.6) so their posterior face is at 11.4.",
    "Residual overlaps inherited from frame.py's pod placement: rim<->cam barrel "
    "(camera pose is in the rim's nasal corner), temple<->ppg package/flex (inside the "
    "temple, only the skin face shows), cam<->irled. Electrodes overlap only their pads.",
]

# Tessellation used for the GLB and the per-part STLs (mm)
LIN_DEFL, ANG_DEFL = 0.02, 0.1

# ----------------------------------------------------------------------------
# Colour palette by category (build123d Color, RGB 0..1)
# ----------------------------------------------------------------------------
COL = {
    "frame":     (0.12, 0.12, 0.13),
    "silicone":  (0.10, 0.10, 0.11),
    "lens":      (0.85, 0.90, 0.95),
    "pcb":       (0.05, 0.35, 0.15),
    "chip":      (0.05, 0.05, 0.05),
    "battery":   (0.70, 0.70, 0.72),
    "electrode": (0.16, 0.16, 0.17),
    "window":    (0.20, 0.20, 0.25),
    "led":       (0.50, 0.10, 0.10),
    "flex":      (0.72, 0.42, 0.12),
    "gold":      (0.85, 0.68, 0.25),
}
LENS_ALPHA = 0.45          # lenses are rendered translucent in the GLB

# Part-number / callout table keyed by label prefix. Order matters: a longer
# prefix must come before the shorter one it extends (thermo_pins before thermo).
_META = {
    "rim":         ("frame",       "Lens rim (PLA)",                              "NRDI-F01"),
    "bridge":      ("frame",       "Bridge (PLA)",                                "NRDI-F02"),
    "nosepad":     ("frame",       "Silicone nose pad with 2 dry electrodes",     "NRDI-NP01"),
    "browpad":     ("frame",       "Silicone brow bumper with 2 dry electrodes",  "NRDI-BP01"),
    "temple":      ("frame",       "Temple arm (PLA)",                            "NRDI-F04"),
    "hinge":       ("frame",       "Stainless hinge + screws",                    "NRDI-H01"),
    "lens":        ("lens",        "Plano lens, CR-39, 1.8 mm",                   "CR39-PLANO-52x40"),
    "cam_fpc":     ("electronics", "Camera FPC, 4 mm, to the bridge",             "OV7670-FPC"),
    "cam":         ("sensor",      "OV7670 pupillography camera, M7 lens",        "OV7670"),
    "camboard":    ("electronics", "OV7670 module board (8x8 mm)",                "OV7670-MOD"),
    "irled_pkg":   ("sensor",      "SFH 4726AS package, 3.75 mm OSLON",           "SFH 4726AS"),
    "irled":       ("sensor",      "SFH 4726AS 940 nm IR illuminator dome",       "SFH 4726AS"),
    "thermo_pins": ("electronics", "TO-39 gold leads (4)",                        "MLX90614ESF-BAA"),
    "thermo":      ("sensor",      "MLX90614 thermopile, TO-39",                  "MLX90614ESF-BAA"),
    "ppg_flex":    ("electronics", "PPG flex tab, 8x6 mm",                        "NRDI-FLEX-PPG"),
    "ppg":         ("sensor",      "MAX30102 PPG module",                         "MAX30102"),
    "elec":        ("sensor",      "Ag/AgCl dry electrode, 6 mm disc",            "Ag/AgCl 6 mm"),
    "pcb":         ("electronics", "Main PCB, 4-layer FR4, 60x12 mm",             "NRDI-PCB-R revA"),
    "mcu":         ("electronics", "ESP32-S3-WROOM-1 module",                     "ESP32-S3-WROOM-1"),
    "i2c_mux":     ("electronics", "TCA9548A I2C multiplexer, TSSOP-24",          "TCA9548APWR"),
    "afe":         ("electronics", "AD5933 impedance AFE, SSOP-16",               "AD5933YRSZ"),
    "battery":     ("power",       "LiPo 250 mAh, 34x12x5 mm",                    "LP401234"),
    "haptic":      ("actuator",    "Coin LRA 10 mm",                              "C10-100"),
}


def _sensor(name):
    return next(s for s in L.SENSORS if s.name == name)


def _side_of(part):
    """Side from geometry, using sensor_layout's convention (+X = wearer's left),
    the same rule frame.py uses for its _L/_R labels. Derived rather than read
    from the label because sensor_layout's mcu/battery notes say "right"/"left"
    temple while their x signs put them in the +X/-X arms (see NOTES)."""
    x = part.center().X
    return "C" if abs(x) < 1.0 else ("L" if x > 0 else "R")


def _meta(label):
    for prefix, m in _META.items():
        if label == prefix or label.startswith(prefix + "_"):
            return m
    raise KeyError(label)


# ----------------------------------------------------------------------------
# Geometry helpers
# ----------------------------------------------------------------------------
def temple_tangent(sgn, y):
    """Unit vector along the temple centreline at station y (same polyline as
    frame.temple_arm: hinge -> p1 is angled, p1 -> p2 runs straight in -Y)."""
    x_hinge = sgn * (F.LENS_CX + L.LENS_W / 2 - 1.0)
    p0 = Vector(x_hinge, F.FRONT_Y, L.LENS_H / 2 - 8.0)
    p1 = Vector(sgn * L.TEMPLE_ARTERY_X, -60.0, L.TEMPLE_ARTERY_Z)
    if y > p1.Y:
        return (p1 - p0).normalized()
    return Vector(0, -1, 0)


def lens_plate(cx, rim, cutters=()):
    """Thin rounded-rect plate filling the rim opening, centred in the rim plane.
    0.1 mm radial clearance so the lens touches but does not overlap the rim.
    `cutters` (nasal sensor pods, temple hinge) are subtracted: their poses put
    them in the lens plane, so the real lens is edged around them."""
    w = L.LENS_W - 2 * L.RIM_W - 0.2
    h = L.LENS_H - 2 * L.RIM_W - 0.2
    plate = extrude(Plane.XZ * RectangleRounded(w, h, radius=6.9), amount=1.8)
    c = plate.bounding_box().center()
    target = rim.bounding_box().center()
    plate = Pos(cx - c.X, target.Y - c.Y, -c.Z) * plate
    for cut in cutters:
        plate = plate - cut
    return plate


def cam_board(s):
    """OV7670 module board: 8x8x1.2 mm plate sitting flush behind the barrel."""
    pl = Plane(origin=Vector(*s.pos), z_dir=Vector(*s.unit_axis()))
    return pl * Box(8.0, 8.0, 1.2, align=(Align.CENTER, Align.CENTER, Align.MAX))


def pcb_plane(s):
    """Local frame on the temple at the MCU pose: x along the arm, z up.
    The PCB top face sits at the pose height."""
    sgn = 1.0 if s.pos[0] > 0 else -1.0
    t = temple_tangent(sgn, s.pos[1])
    return Plane(origin=Vector(*s.pos), x_dir=t, z_dir=Vector(0, 0, 1))


def pcb(pl):
    w, h, t = L.PCB_SIZE                       # 60 x 12 x 1.6
    return pl * Box(w, h, t, align=(Align.CENTER, Align.CENTER, Align.MAX))


def mcu_module(pl):
    """ESP32-S3-WROOM-1: 18 x 25.5 x 0.8 substrate + shield can, one antenna
    end (6 mm) left bare. Long axis along the PCB. Fused into one body."""
    sub = Box(25.5, 18.0, 0.8, align=(Align.CENTER, Align.CENTER, Align.MIN))
    can = Pos(-3.0, 0, 0.8) * F._fillet_safe(Box(19.5, 16.0, 2.3, align=(Align.CENTER, Align.CENTER, Align.MIN)), 0.3)
    return pl * (sub + can)


def smd(pl, along, length, width, height):
    """Small package on the PCB top face at station `along` (mm along the arm)."""
    return pl * Pos(along, 0, 0) * Box(length, width, height,
                                       align=(Align.CENTER, Align.CENTER, Align.MIN))


def battery_pouch(s):
    """LiPo pouch, long axis along the straight temple run (Y)."""
    w, h, t = L.BATTERY_SIZE                   # 34 x 12 x 5
    return Pos(*s.pos) * F._fillet_safe(Box(h, w, t), 1.2)


def haptic_coin(s):
    """10 mm coin LRA, vibration axis == pose axis (into the skin)."""
    pl = Plane(origin=Vector(*s.pos), z_dir=Vector(*s.unit_axis()))
    return F._fillet_safe(pl * Cylinder(5.0, 3.6), 0.4)


# ----------------------------------------------------------------------------
# Explode rules (direction, distance mm, order) -- see task brief
# ----------------------------------------------------------------------------
def _unit(v):
    n = math.sqrt(sum(c * c for c in v))
    return [round(c / n, 6) for c in v]


def explode_rule(part, s=None):
    label = part.label
    sgn = +1.0 if _side_of(part) == "L" else -1.0
    if label.startswith("lens"):
        return _unit((0, 1, 0)), 25.0, 0
    if label.startswith("hinge"):
        return _unit((sgn, 0.4, 0)), 10.0, 1
    if label.startswith("rim"):
        return _unit((0, 1, 0)), 12.0, 1
    if label == "bridge":
        return _unit((0, 18, 6)), math.hypot(18, 6), 1
    if label.startswith("nosepad"):
        return _unit((0, -1, 0)), 8.0, 3
    if label.startswith("browpad"):
        return _unit((0, 0, 1)), 16.0, 2
    if label.startswith("temple"):
        return _unit((sgn * 20, -10, 0)), math.hypot(20, 10), 1
    if label.startswith(("camboard", "cam_fpc")):
        return _unit([-c for c in s.unit_axis()]), 8.0, 3
    if label.startswith(("cam_", "thermo", "irled")):        # incl. thermo_pins, irled_pkg
        return _unit(s.unit_axis()), 14.0, 2
    if label.startswith("ppg_flex"):
        return _unit((sgn, 0, 0)), 22.0, 3
    if label.startswith("ppg"):
        return _unit((sgn, 0, 0)), 18.0, 2
    if label.startswith("elec"):                             # away from the skin
        return _unit([-c for c in s.unit_axis()]), 10.0, 2
    z = {"pcb_R": 22, "mcu": 30, "i2c_mux": 36, "afe": 36, "battery": 22, "haptic": 18}[label]
    return _unit((0, 0, 1)), float(z), 3


# ----------------------------------------------------------------------------
# Build
# ----------------------------------------------------------------------------
def build():
    """Return (list of labelled+coloured solids, dict label -> Sensor or None)."""
    parts, src = [], {}

    def add(label, shape, colour, s=None):
        assert label.replace("_", "").isalnum() and label.isascii(), label
        assert label not in src, f"duplicate label {label}"
        shape.label = label
        shape.color = Color(*colour)
        parts.append(shape)
        src[label] = s

    # --- frame (from frame.py builders) --------------------------------------
    rim_L, rim_R = F.lens_rim(+F.LENS_CX), F.lens_rim(-F.LENS_CX)
    brd = F.bridge()
    add("rim_L", rim_L, COL["frame"])
    add("rim_R", rim_R, COL["frame"])
    add("bridge", brd, COL["frame"])
    add("nosepad_L", F.nose_pad(+1, rim_L), COL["silicone"])
    add("nosepad_R", F.nose_pad(-1, rim_R), COL["silicone"])
    add("browpad_L", F.brow_bumper(+1), COL["silicone"])
    add("browpad_R", F.brow_bumper(-1), COL["silicone"])
    temples = {"L": F.temple_arm(+1), "R": F.temple_arm(-1)}
    add("temple_L", temples["L"], COL["frame"])
    add("temple_R", temples["R"], COL["frame"])
    add("hinge_L", F.hinge(+1), COL["battery"])
    add("hinge_R", F.hinge(-1), COL["battery"])

    # --- sensor housings (same builders frame.py fuses) ----------------------
    nasal = {"L": [], "R": []}          # bodies that cross the lens plane
    for s in L.SENSORS:
        side = s.name[-1] if s.kind != "electrode" else s.name.split("_")[1]
        if s.kind == "camera":
            pod = F.camera_pod(s)
            add(s.name, pod, COL["window"], s)
            add(f"camboard_{side}", cam_board(s), COL["chip"], s)
            add(f"cam_fpc_{side}", F.camera_fpc(s, brd), COL["flex"], s)
            nasal[side].append(pod)
        elif s.kind == "ir_led":
            pod, pkg = F.emitter_pod(s), F.emitter_package(s)
            add(s.name, pod, COL["led"], s)
            add(f"irled_pkg_{side}", pkg, COL["chip"], s)
            nasal[side] += [pod, pkg]
        elif s.kind == "thermopile":
            pod, pins = F.thermopile_pod(s), F.thermopile_pins(s)
            add(s.name, pod, COL["window"], s)
            add(f"thermo_pins_{side}", pins, COL["gold"], s)
            nasal[side] += [pod, pins]
        elif s.kind == "electrode":
            add(s.name, F.electrode_disc(s), COL["electrode"], s)
        elif s.kind == "ppg":
            add(s.name, F.ppg_window(s), COL["chip"], s)
            add(f"ppg_flex_{side}", F.ppg_flex(s), COL["flex"], s)

    # --- lenses, edged around the nasal pods and the hinge corner ------------
    add("lens_L", lens_plate(+F.LENS_CX, rim_L, nasal["L"] + [temples["L"]]), COL["lens"])
    add("lens_R", lens_plate(-F.LENS_CX, rim_R, nasal["R"] + [temples["R"]]), COL["lens"])

    # --- electronics on the main PCB (at sensor_layout's mcu pose) -----------
    mcu = _sensor("mcu")
    pl = pcb_plane(mcu)
    add("pcb_R", pcb(pl), COL["pcb"], mcu)
    add("mcu", mcu_module(pl), COL["chip"], mcu)
    add("i2c_mux", smd(pl, +20.0, 7.8, 4.4, 1.0), COL["chip"], mcu)   # toward the ear
    add("afe", smd(pl, -20.0, 6.2, 5.3, 1.75), COL["chip"], mcu)      # toward the hinge

    # --- power / actuator ----------------------------------------------------
    add("battery", battery_pouch(_sensor("battery")), COL["battery"], _sensor("battery"))
    add("haptic", haptic_coin(_sensor("haptic")), COL["battery"], _sensor("haptic"))
    return parts, src


# ----------------------------------------------------------------------------
# GLB writer (trimesh): one mesh + one node per part, names preserved
# ----------------------------------------------------------------------------
def write_glb(parts, path):
    import trimesh
    from trimesh.visual.material import PBRMaterial
    from trimesh.visual import TextureVisuals

    scene = trimesh.Scene()
    for p in parts:
        verts, faces = p.tessellate(LIN_DEFL, ANG_DEFL)
        mesh = trimesh.Trimesh(
            vertices=np.array([[v.X, v.Y, v.Z] for v in verts], dtype=np.float64),
            faces=np.array(faces, dtype=np.int64), process=False)
        r, g, b = tuple(p.color)[:3]
        a = LENS_ALPHA if p.label.startswith("lens") else 1.0
        mat = PBRMaterial(name=f"mat_{p.label}",
                          baseColorFactor=[r, g, b, a],
                          metallicFactor=0.9 if p.label in ("battery", "haptic") else 0.0,
                          roughnessFactor=0.35 if p.label.startswith("lens") else 0.7,
                          alphaMode="BLEND" if a < 1.0 else "OPAQUE")
        mesh.visual = TextureVisuals(material=mat)
        # no transform: keep project Z-up millimetre coordinates verbatim
        scene.add_geometry(mesh, node_name=p.label, geom_name=p.label)
    scene.export(str(path))
    return scene


# ----------------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------------
def manifest(parts, src, asm):
    rows = []
    for p in parts:
        bb = p.bounding_box()
        c = p.center()
        d, dist, order = explode_rule(p, src[p.label])
        cat, callout, pn = _meta(p.label)
        # mount point for the dashboard's leader lines: the sensor pose for
        # sensors and electrodes, the part centroid for everything else
        s = src[p.label]
        mount = list(s.pos) if (cat == "sensor" and s is not None) else [c.X, c.Y, c.Z]
        rows.append({
            "label": p.label,
            "category": cat,
            "side": _side_of(p),
            "pos_mm": [round(c.X, 3), round(c.Y, 3), round(c.Z, 3)],
            "mount_mm": [round(float(v), 3) for v in mount],
            "bbox_mm": {"min": [round(bb.min.X, 3), round(bb.min.Y, 3), round(bb.min.Z, 3)],
                        "max": [round(bb.max.X, 3), round(bb.max.Y, 3), round(bb.max.Z, 3)]},
            "explode": {"dir": d, "dist_mm": round(dist, 3), "order": order},
            "callout": callout,
            "part_number": pn,
            "color": [round(x, 3) for x in tuple(p.color)[:3]],
        })
    abb = asm.bounding_box()
    return {
        "name": asm.label,
        "units": "mm",
        "up": "+Z",
        "coordinate_system": L.to_dict()["coordinate_system"],
        "note": ("GLB is written in project coordinates (mm, +Z up, +Y anterior). "
                 "glTF viewers assume metres and +Y up: scale by 0.001 and rotate "
                 "-90 deg about X on the viewer side. Node name == geometry name == label."),
        "assembled_bbox_mm": {
            "min": [round(abb.min.X, 3), round(abb.min.Y, 3), round(abb.min.Z, 3)],
            "max": [round(abb.max.X, 3), round(abb.max.Y, 3), round(abb.max.Z, 3)]},
        "overlap_policy": ("electronics sit inside the temple envelope at their sensor_layout "
                           "poses without pockets; temple<->electronics overlap accepted; "
                           "electronics do not overlap each other"),
        "notes": NOTES,
        "explode_orders": {"0": "lenses", "1": "rims, bridge, temples",
                           "2": "sensors, electrodes (away from the skin), brow bumpers (+Z)",
                           "3": "electronics, flex/FPC, power, actuator, nosepads"},
        "parts": rows,
    }


# ----------------------------------------------------------------------------
if __name__ == "__main__":
    parts, src = build()
    asm = Compound(children=parts, label="nrdi_glasses")
    print(f"built {len(parts)} bodies: {', '.join(p.label for p in parts)}")

    # canonical GLB (trimesh writer, one node per part)
    glb = OUT / "glasses.glb"
    write_glb(parts, glb)
    print(f"wrote {glb}  ({glb.stat().st_size/1024:.0f} kB)")

    # STEP assembly
    export_step(asm, str(OUT / "glasses.step"))
    print(f"wrote {OUT/'glasses.step'}")

    # per-part STL
    for p in parts:
        export_stl(p, str(STL_DIR / f"{p.label}.stl"), tolerance=LIN_DEFL, angular_tolerance=ANG_DEFL)
        # coarse copies for the web viewer (~10x fewer triangles)
        export_stl(p, str(WEB_DIR / f"{p.label}.stl"), tolerance=0.12, angular_tolerance=0.45)
    print(f"wrote {len(parts)} STLs -> {STL_DIR} (+ web copies in {WEB_DIR})")

    # build123d's native glTF writer, for reference only (one primitive per face)
    try:
        export_gltf(asm, str(OUT / "glasses_occt.glb"), binary=True,
                    linear_deflection=LIN_DEFL, angular_deflection=ANG_DEFL)
        print(f"wrote {OUT/'glasses_occt.glb'}  (reference; per-face primitives)")
    except Exception as e:                                   # noqa: BLE001
        print(f"export_gltf failed (non-fatal): {e!r}")

    # manifest
    man = manifest(parts, src, asm)
    (OUT / "parts.json").write_text(json.dumps(man, indent=2), encoding="utf-8")
    print(f"wrote {OUT/'parts.json'}")
    bb = man["assembled_bbox_mm"]
    print(f"assembled bbox = {bb['min']} .. {bb['max']} mm")
