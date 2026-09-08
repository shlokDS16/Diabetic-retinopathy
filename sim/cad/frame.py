"""
Parametric spectacle frame with sensor mounts -- v2 (product-render quality).

Reads geometry from sensor_layout.py so the CAD and the physics models can
never disagree about where a sensor is.

v2 changes (2026-09-03): every body is filleted (acetate look), temples are
swept with a rounded section and taper into a tip, hinges are real bodies
with screw heads, nose pads sit on arms, the thermopile can has a window
ring, the camera barrel has a lens dome and the IR LED a dome. Fillets are
wrapped in `_fillet_safe` so an OCC fillet failure degrades to sharp edges
instead of killing the build.

v3 changes (2026-09-04): electrodes are contacts only where eyewear already
touches the face -- no sprung arms above/below the rims. A silicone brow
bumper on the posterior face of each upper rim carries brow_med / brow_lat;
the nose pad (repositioned onto the nasal sidewall at the electrode poses)
carries nose_sup / nose_inf. Each electrode is a 6 mm x 0.8 mm disc whose
face is at the sensor pose with a 0.3 mm proud Ag/AgCl button. The rims are
now centred on FRONT_Y (y 11.4..14.6) so their posterior face is at
RIM_BACK_Y = 11.4, which is where the bridge back and the bumper front sit.
Sensor pods are drawn as the real packages: TO-39 can with pins, MAX30102
on a flex tab, OV7670 M7 barrel with an FPC, SFH 4726AS with a clear dome.

Exports (when run directly)
    out/frame.step   CAD interchange -> patent figures, downstream FEM meshing
    out/frame.stl    tessellated     -> gmsh / MCX voxelisation / three.js
    out/fov.stl      sensor field-of-view cones, as separate visualisation geometry

Run:  python sim/cad/frame.py
"""
import sys, pathlib, math

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from build123d import (
    Box, Cylinder, Cone, Sphere, Rectangle, RectangleRounded, Plane, Pos, Rot,
    Axis, Location, Vector, extrude, fillet, chamfer, sweep, FilletPolyline, Mode,
    export_step, export_stl, Compound, Align, Circle, loft, Polygon,
)
import sensor_layout as L

OUT = HERE.parents[1] / "sim" / "out" if (HERE.parents[1] / "sim").exists() else HERE.parent / "out"
OUT.mkdir(parents=True, exist_ok=True)

LENS_CX = (L.BRIDGE_W + L.LENS_W) / 2.0          # 35.0 mm
DECENTRATION = LENS_CX - L.IPD / 2.0             # 3.5 mm nasal
FRONT_Y = L.VERTEX_DISTANCE                      # frame front plane (rim mid-depth)
RIM_BACK_Y = FRONT_Y - L.RIM_T / 2.0             # 11.4 mm, posterior face of the rims

# Brow bumper (v3): silicone strip on the posterior face of the upper rim.
BROW_X_MED, BROW_X_LAT = 15.0, 50.0              # |x| extent of the strip
BROW_H = 6.0                                     # body height (z 14..20, top flush with the rim)
BROW_LIP_T, BROW_LIP_H = 2.0, 2.0                # thin rear lip: depth (y) and rise above the rim (z)
BROW_TAPER_END = 0.8                             # residual depth where the lateral end meets the rim

# Electrode contact (v3): 6 mm disc, face at the pose, Ag/AgCl button proud
ELEC_R, ELEC_T, ELEC_BTN_R, ELEC_BTN_PROUD = 3.0, 0.8, 2.0, 0.3


def _fillet_safe(solid, radius, edge_filter=None):
    """Fillet all (or filtered) edges; fall back to the unfilleted solid."""
    try:
        edges = solid.edges() if edge_filter is None else edge_filter(solid.edges())
        if len(edges) == 0:
            return solid
        return fillet(edges, radius=radius)
    except Exception:
        try:
            return fillet(solid.edges(), radius=radius * 0.5)
        except Exception:
            return solid


# ---------------------------------------------------------------- frame ----
def lens_rim(cx):
    """Acetate-style rim: rounded-rect annulus, softened outer edges, small
    inner bevel where the lens seats."""
    outer = RectangleRounded(L.LENS_W, L.LENS_H, radius=12.0)
    inner = RectangleRounded(L.LENS_W - 2 * L.RIM_W, L.LENS_H - 2 * L.RIM_W, radius=9.5)
    solid = extrude(Plane.XZ * (outer - inner), amount=L.RIM_T)    # extrudes toward -Y
    solid = _fillet_safe(solid, 1.0, lambda e: e.filter_by(lambda ed: ed.length > 8.0))
    # centred on FRONT_Y: y in [RIM_BACK_Y, FRONT_Y + RIM_T/2] = [11.4, 14.6]
    return Pos(cx, FRONT_Y + L.RIM_T / 2.0, 0) * solid


def bridge():
    """Keyhole-style bridge bar, filleted."""
    bar = Box(L.BRIDGE_W + 2 * L.RIM_W + 1.0, L.RIM_T + 0.6, 7.0)
    bar = _fillet_safe(bar, 1.4)
    return Pos(0, FRONT_Y + 0.3, L.LENS_H / 2 - 7.5) * bar


def _elec(side, tag):
    return next(s for s in L.SENSORS if s.name == f"elec_{side}_{tag}")


def nose_pad_plane(sgn):
    """Local frame of the nose pad's SKIN face: origin = midpoint of the two
    nose electrodes, z = sensor_layout's nasal axis tilted about the in-plane
    horizontal just enough that BOTH electrode centres lie on the face (their
    poses are ~0.8 mm apart along the nasal axis, so an untilted plane cannot
    be flush with both), x = in-plane anterior-posterior direction."""
    side = "L" if sgn > 0 else "R"
    sup, inf = _elec(side, "nose_sup"), _elec(side, "nose_inf")
    c = (Vector(*sup.pos) + Vector(*inf.pos)) / 2.0
    d = Vector(*inf.pos) - Vector(*sup.pos)
    n = Vector(*sup.unit_axis())
    n = (n - d * (n.dot(d) / d.dot(d))).normalized()
    u = Vector(0, 1, 0)
    u = (u - n * u.dot(n)).normalized()
    return Plane(origin=c, x_dir=u, z_dir=n)


def nose_pad(sgn, rim=None):
    """v3: 9 x 16 x 2 mm rounded silicone pad whose skin face sits on the nasal
    sidewall through the two nose-electrode poses (see nose_pad_plane), plus a
    short 0.9 mm arm from the rim's posterior nasal corner to the pad's back.
    Pass the same-side `rim` to notch the pad where it seats against the rim
    so the two only touch. The arm ends inside the pad, 1 mm behind the
    contact face, so it never crosses an electrode."""
    pl = nose_pad_plane(sgn)
    pad = pl * extrude(RectangleRounded(9.0, 16.0, radius=3.5), amount=-2.0)   # body behind the face
    pad = _fillet_safe(pad, 0.8)
    if rim is not None:
        pad = pad - rim
    start = Vector(sgn * 10.0, RIM_BACK_Y, -3.0)                # rim back face, nasal corner
    end = pl.origin - pl.z_dir * 1.0                            # mid-thickness of the pad
    v = end - start
    arm = Plane(origin=start, z_dir=v) * Cylinder(0.45, v.length, align=(Align.CENTER, Align.CENTER, Align.MIN))
    return pad + arm


def brow_rear_y(sgn, x):
    """Skin (rear) face of the brow bumper at x: the straight line through the
    two brow-electrode poses, held constant medially of brow_med."""
    side = "L" if sgn > 0 else "R"
    med, lat = _elec(side, "brow_med"), _elec(side, "brow_lat")
    xm, ym, xl, yl = abs(med.pos[0]), med.pos[1], abs(lat.pos[0]), lat.pos[1]
    ax = abs(x)
    if ax <= xm:
        return ym
    return ym + (yl - ym) * (ax - xm) / (xl - xm)


def _brow_rear_pts(sgn):
    """Rear-face polyline (x, y) of the bumper, medial -> lateral: held at the
    brow_med depth medially, through both electrode poses, held past the
    lateral electrode so its disc stays embedded, then tapered into the rim."""
    side = "L" if sgn > 0 else "R"
    med, lat = _elec(side, "brow_med"), _elec(side, "brow_lat")
    x_hold = abs(lat.pos[0]) + ELEC_R + 0.5                      # past the lateral disc
    pts = [(BROW_X_MED, brow_rear_y(sgn, BROW_X_MED)),
           (abs(med.pos[0]), med.pos[1]),
           (abs(lat.pos[0]), lat.pos[1]),
           (x_hold, lat.pos[1]),
           (BROW_X_LAT, RIM_BACK_Y - BROW_TAPER_END)]           # tapered into the rim back
    return [(sgn * x, y) for x, y in pts]


def brow_bumper(sgn):
    """v3.1: soft silicone strip BEHIND the upper rim, never above it. Body
    z 14..20 (top flush with the rim top), front face on the rim back
    (y = RIM_BACK_Y), rear face follows the brow through the two brow-electrode
    poses and tapers into the rim laterally. A thin 2 mm rear lip rises to
    z = 22 so the electrode discs (centred at z = 21) stay embedded except for
    their top 2 mm."""
    z_top = L.LENS_H / 2.0                                       # 20, rim top
    rear = _brow_rear_pts(sgn)
    body_pts = [(rear[0][0], RIM_BACK_Y), (rear[-1][0], RIM_BACK_Y)] + list(reversed(rear))
    body = extrude(Plane.XY * Polygon(*body_pts, align=None), amount=BROW_H)
    body = Pos(0, 0, z_top - BROW_H - body.bounding_box().min.Z) * body     # z 14..20
    lip_pts = list(rear) + [(x, min(y + BROW_LIP_T, RIM_BACK_Y)) for x, y in reversed(rear)]
    lip = extrude(Plane.XY * Polygon(*lip_pts, align=None), amount=BROW_LIP_H + 0.5)
    lip = Pos(0, 0, z_top - 0.5 - lip.bounding_box().min.Z) * lip           # z 19.5..22
    return _fillet_safe(body + lip, 0.5)


def temple_arm(sgn):
    """Hinge -> straight run -> ear hook. Rounded 12 x 4.5 section (houses the
    PCB) swept along a filleted polyline, with a softened tip cap."""
    w, h = L.TEMPLE_SECTION
    x_hinge = sgn * (LENS_CX + L.LENS_W / 2 - 1.0)
    p0 = Vector(x_hinge, FRONT_Y, L.LENS_H / 2 - 8.0)
    p1 = Vector(sgn * L.TEMPLE_ARTERY_X, -60.0, L.TEMPLE_ARTERY_Z)
    p2 = Vector(sgn * L.TEMPLE_ARTERY_X, -L.TEMPLE_LEN + 20.0, L.TEMPLE_ARTERY_Z)
    p3 = Vector(sgn * L.TEMPLE_ARTERY_X, -L.TEMPLE_LEN, L.TEMPLE_ARTERY_Z - L.EARHOOK_DROP)
    path = FilletPolyline(p0, p1, p2, p3, radius=12.0)
    tangent = (p1 - p0).normalized()
    profile = Plane(origin=p0, z_dir=tangent) * RectangleRounded(w, h, radius=1.8)
    arm = sweep(profile, path)
    # soft tip: a rounded cap at the end of the ear hook
    tip = Pos(p3.X, p3.Y, p3.Z) * Sphere(min(w, h) / 2.0 * 0.98)
    try:
        arm = arm + tip
    except Exception:
        pass
    return arm


def hinge(sgn):
    """Metal hinge barrel at the rim/temple corner with two screw heads."""
    x = sgn * (LENS_CX + L.LENS_W / 2 - 0.5)
    z = L.LENS_H / 2 - 8.0
    barrel = Pos(x, FRONT_Y - 1.2, z) * Cylinder(2.3, 7.5)          # vertical barrel
    barrel = _fillet_safe(barrel, 0.4)
    screw_top = Pos(x, FRONT_Y - 1.2, z + 3.75) * Cylinder(1.2, 0.5, align=(Align.CENTER, Align.CENTER, Align.MIN))
    screw_bot = Pos(x, FRONT_Y - 1.2, z - 3.75) * Cylinder(1.2, 0.5, align=(Align.CENTER, Align.CENTER, Align.MAX))
    return barrel + screw_top + screw_bot


# ------------------------------------------------------------ sensor pods ----
_MIN = (Align.CENTER, Align.CENTER, Align.MIN)
_MAX = (Align.CENTER, Align.CENTER, Align.MAX)


def _frame_at(s, x_hint=None):
    """Local frame at the sensor pose: z = sensing axis. With `x_hint`, local x
    is that global direction projected into the pose plane."""
    ax = Vector(*s.unit_axis())
    if x_hint is None:
        return Plane(origin=Vector(*s.pos), z_dir=ax)
    h = Vector(*x_hint)
    h = (h - ax * h.dot(ax)).normalized()
    return Plane(origin=Vector(*s.pos), x_dir=h, z_dir=ax)


def _ring(r_out, r_in, h):
    return Cylinder(r_out, h, align=_MIN) - Cylinder(r_in, h, align=_MIN)


def camera_pod(s):
    """OV7670 lens assembly: 8 x 8 mm holder base on the module board, M7
    barrel (three shallow thread grooves) and a glass lens dome. The barrel
    points along the pose axis (toward the pupil)."""
    pl = _frame_at(s, x_hint=(0, 0, 1))
    holder = _fillet_safe(Box(8.0, 8.0, 2.5, align=_MIN), 0.6)
    barrel = Pos(0, 0, 2.5) * Cylinder(3.5, 5.5, align=_MIN)             # M7 x 0.35 look
    for z in (5.4, 6.2, 7.0):
        barrel = barrel - Pos(0, 0, z) * _ring(3.6, 3.3, 0.3)
    dome = Pos(0, 0, 8.0) * (Sphere(2.6) & Cylinder(2.6, 2.6, align=_MIN))
    return pl * (holder + barrel + dome)


def camera_fpc(s, cutter=None):
    """4 mm wide flexible printed circuit on the back (anterior face) of the
    OV7670 module board, trailing 5 mm in-plane toward the bridge. Pass the
    bridge as `cutter` so the strip stops at the bridge surface."""
    pl = _frame_at(s, x_hint=(0, 0, 1))                                  # local x = in-plane "up"
    tab = pl * Pos(4.0 + 2.5, 0, -1.2 - 0.075) * Box(5.0, 4.0, 0.15)
    if cutter is not None:
        tab = tab - cutter
    return tab


def emitter_package(s):
    """SFH 4726AS (OSLON Black) body: 3.75 x 3.75 x 2.3 mm black package."""
    pl = _frame_at(s, x_hint=(0, 0, 1))
    return pl * _fillet_safe(Box(3.75, 3.75, 2.3, align=_MIN), 0.2)


def emitter_pod(s):
    """SFH 4726AS clear silicone dome on top of the package (the part that is
    rendered as the IR emitter)."""
    pl = _frame_at(s, x_hint=(0, 0, 1))
    return pl * Pos(0, 0, 2.3) * (Sphere(1.5) & Cylinder(1.5, 1.5, align=_MIN))


def thermopile_pod(s):
    """MLX90614 in a TO-39 can: 9.2 mm base flange, 8.1 mm can 4.5 mm high, a
    4.8 mm filter window recessed 0.3 mm with a 0.4 mm metal rim, and the
    field-limiting shroud hugging the can (fused, not floating)."""
    pl = _frame_at(s)
    flange = Cylinder(4.6, 0.8, align=_MIN)                               # z 0 .. 0.8
    can = _fillet_safe(Pos(0, 0, 0.8) * Cylinder(4.05, 3.7, align=_MIN), 0.3)   # 0.8 .. 4.5
    body = flange + can
    body = body - Pos(0, 0, 4.2) * Cylinder(2.4, 0.3, align=_MIN)         # window recess
    rim_ring = Pos(0, 0, 4.5) * _ring(2.8, 2.4, 0.15)                     # metal window rim
    shroud = Pos(0, 0, 2.5) * _ring(5.4, 4.0, 5.0)                        # 2.5 .. 7.5
    return pl * (body + rim_ring + shroud)


def thermopile_pins(s):
    """Four gold-plated TO-39 leads (0.45 mm dia, 5 mm) on a 5.08 mm circle,
    out of the back of the flange."""
    pl = _frame_at(s)
    pins = None
    for k in range(4):
        a = math.radians(45 + 90 * k)
        pin = Pos(2.54 * math.cos(a), 2.54 * math.sin(a), 0) * Cylinder(0.225, 5.0, align=_MAX)
        pins = pin if pins is None else pins + pin
    return pl * pins


def electrode_disc(s):
    """v3 dry electrode: 6 mm x 0.8 mm Ag/AgCl-coated disc whose FACE is at
    the pose with its normal along the pose axis (into the skin), plus a 0.3
    mm proud 4 mm button. The disc body sits behind the face, inside the
    brow bumper or the nose pad. No arms."""
    pl = _frame_at(s)
    disc = Cylinder(ELEC_R, ELEC_T, align=_MAX)                           # behind the face
    btn = Cylinder(ELEC_BTN_R, ELEC_BTN_PROUD, align=_MIN)                # proud, into the skin
    return pl * _fillet_safe(disc + btn, 0.12, lambda e: e.filter_by(lambda ed: ed.length > 10.0))


def _ppg_face_dist(s):
    """Distance from the PPG pose (temple centreline) to its skin face: the
    temple's inner surface plus 0.25 mm proud."""
    inner_x = L.TEMPLE_ARTERY_X - L.TEMPLE_SECTION[0] / 2.0
    return (abs(s.pos[0]) - inner_x) + 0.25


def ppg_window(s):
    """MAX30102 package: 5.6 x 3.3 x 1.55 mm black body, long axis along the
    temple, with the three optical windows (two LED 1 x 0.8, one PD 1.4 x 1.4)
    recessed in the skin face. The skin face is 0.25 mm proud of the temple's
    inner surface."""
    pl = _frame_at(s, x_hint=(0, 1, 0))
    d = _ppg_face_dist(s)
    pkg = Pos(0, 0, d - 1.55) * Box(5.6, 3.3, 1.55, align=_MIN)
    for (x, y, w, h) in ((-1.6, +0.6, 1.0, 0.8), (-1.6, -0.6, 1.0, 0.8), (1.5, 0.0, 1.4, 1.4)):
        pkg = pkg - Pos(x, y, d - 0.2) * Box(w, h, 0.2, align=_MIN)
    return pl * pkg


def ppg_flex(s):
    """8 x 6 x 0.8 mm flex tab the MAX30102 is mounted on (behind the package,
    inside the temple)."""
    pl = _frame_at(s, x_hint=(0, 1, 0))
    d = _ppg_face_dist(s)
    return pl * Pos(0, 0, d - 1.55) * Box(8.0, 6.0, 0.8, align=_MAX)


def fov_cone(s, length=30.0):
    """Visualisation only: the solid angle each sensor actually sees."""
    if s.fov_deg <= 0:
        return None
    r = length * math.tan(math.radians(s.fov_deg / 2.0))
    pl = _frame_at(s)
    return pl * Cone(0.35, r, length, align=(Align.CENTER, Align.CENTER, Align.MIN))


def build():
    parts, log = [], []

    def add(label, shape):
        if shape is not None:
            parts.append(shape)
            log.append(label)

    rim_L, rim_R = lens_rim(+LENS_CX), lens_rim(-LENS_CX)
    brd = bridge()
    add("rim_L", rim_L)
    add("rim_R", rim_R)
    add("bridge", brd)
    add("nosepad_L", nose_pad(+1, rim_L))
    add("nosepad_R", nose_pad(-1, rim_R))
    add("browpad_L", brow_bumper(+1))
    add("browpad_R", brow_bumper(-1))
    add("temple_L", temple_arm(+1))
    add("temple_R", temple_arm(-1))
    add("hinge_L", hinge(+1))
    add("hinge_R", hinge(-1))

    for s in L.SENSORS:
        side = s.name.split("_")[1] if s.kind == "electrode" else s.name[-1]
        if s.kind == "camera":
            add(s.name, camera_pod(s))
            add(f"cam_fpc_{side}", camera_fpc(s, brd))
        elif s.kind == "ir_led":
            add(s.name, emitter_pod(s))
            add(f"irled_pkg_{side}", emitter_package(s))
        elif s.kind == "thermopile":
            add(s.name, thermopile_pod(s))
            add(f"thermo_pins_{side}", thermopile_pins(s))
        elif s.kind == "electrode":
            add(s.name, electrode_disc(s))
        elif s.kind == "ppg":
            add(s.name, ppg_window(s))
            add(f"ppg_flex_{side}", ppg_flex(s))

    frame = parts[0]
    for p in parts[1:]:
        frame = frame + p
    return frame, log


def build_fov():
    cones = [c for c in (fov_cone(s) for s in L.SENSORS) if c is not None]
    if not cones:
        return None
    out = cones[0]
    for c in cones[1:]:
        out = out + c
    return out


if __name__ == "__main__":
    print(f"lens centre  x = +/-{LENS_CX:.1f} mm")
    print(f"pupil centre x = +/-{L.IPD/2:.1f} mm")
    print(f"optical decentration = {DECENTRATION:.1f} mm nasal\n")

    frame, log = build()
    print(f"assembled {len(log)} bodies: {', '.join(log)}")
    print(f"frame volume = {frame.volume/1000:.2f} cm3")
    bb = frame.bounding_box()
    print(f"bounding box = {bb.size.X:.1f} x {bb.size.Y:.1f} x {bb.size.Z:.1f} mm")

    export_step(frame, str(OUT / "frame.step"))
    export_stl(frame, str(OUT / "frame.stl"))
    print(f"\nwrote {OUT/'frame.step'}")
    print(f"wrote {OUT/'frame.stl'}")

    fov = build_fov()
    if fov is not None:
        export_stl(fov, str(OUT / "fov.stl"))
        print(f"wrote {OUT/'fov.stl'}")

    print(f"\nestimated frame mass (PLA) = {frame.volume/1000*1.24:.1f} g")
