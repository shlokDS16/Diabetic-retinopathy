"""
Exploded-view animation of the NRDI glasses in Blender, driven entirely from
sim/out/parts.json (explode vectors, order, colours) and sim/out/parts_stl/.

Per-part STL import keeps the project's Z-up millimetre frame verbatim (the
glTF importer would re-map axes), so everything is scaled 0.001 to metres and
the same explode vectors serve Blender and the Three.js viewer.

Timeline (frames, 30 fps):
    0-20    assembled hold
    20-90   explode, staggered by `order` (0 first)
    90-110  exploded hold
    110-170 reassemble (mirror)

Usage (headless):
    blender -b --python sim/render/blender_explode.py -- [--engine EEVEE|CYCLES]
    (v2: leader lines + mount rings + camera-facing callout labels for every
     sensor-like part; the .blend is saved with packed images)
                                                          [--frames 0,70,100]
                                                          [--anim] [--res 1920x1080]
                                                          [--samples 128] [--out DIR]
"""
import bpy, json, math, sys, pathlib
from mathutils import Vector

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "sim" / "out"
PARTS = json.loads((OUT_DIR / "parts.json").read_text(encoding="utf-8"))
STL_DIR = OUT_DIR / "parts_stl"

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default
ENGINE  = arg("--engine", "EEVEE").upper()
FRAMES  = [int(f) for f in arg("--frames", "0,70,100").split(",")]
ANIM    = "--anim" in argv
RES     = tuple(int(v) for v in arg("--res", "1600x900").split("x"))
SAMPLES = int(arg("--samples", "64"))
RENDER_DIR = pathlib.Path(arg("--out", str(ROOT / "sim" / "render" / "out")))
RENDER_DIR.mkdir(parents=True, exist_ok=True)

MM = 0.001
FPS = 30
F_HOLD0, F_EXPL0, F_EXPL1, F_HOLD1, F_END = 0, 20, 90, 110, 170
STAGGER = 12          # frames between successive `order` groups
DUR = 40              # frames each group takes to travel

# ------------------------------------------------------------------ scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.fps = FPS
scene.frame_start, scene.frame_end = 0, F_END
scene.render.resolution_x, scene.render.resolution_y = RES
scene.render.film_transparent = False
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.view_settings.exposure = 0.0

if ENGINE == "CYCLES":
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"
    prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
else:
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = SAMPLES
    scene.eevee.use_raytracing = True

# ------------------------------------------------------------ materials ----
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import materials as M
ASSETS = ROOT / "sim" / "assets"

# ------------------------------------------------------------- import ----
rig = bpy.data.objects.new("nrdi_rig", None)
scene.collection.objects.link(rig)

objs = {}
for p in PARTS["parts"]:
    label = p["label"]
    path = STL_DIR / f"{label}.stl"
    bpy.ops.wm.stl_import(filepath=str(path), global_scale=MM,
                          forward_axis="Y", up_axis="Z")
    ob = bpy.context.selected_objects[0]
    ob.name = label
    ob.parent = rig
    ob.data.materials.append(M.material_for(p))
    for poly in ob.data.polygons:
        poly.use_smooth = True
    if hasattr(bpy.ops.object, "shade_smooth_by_angle"):
        bpy.context.view_layer.objects.active = ob
        try: bpy.ops.object.shade_smooth_by_angle(angle=math.radians(30))
        except Exception: pass
    objs[label] = (ob, p)

# --------------------------------------------------------- animation -----
def ease(t):
    return 0.5 - 0.5 * math.cos(math.pi * min(max(t, 0.0), 1.0))

for label, (ob, p) in objs.items():
    e = p["explode"]
    disp = Vector(e["dir"]) * e["dist_mm"] * MM
    k = e["order"]
    f_start = F_EXPL0 + k * STAGGER
    f_stop = min(f_start + DUR, F_EXPL1)
    # reassembly mirrors the explosion
    r_start = F_HOLD1 + k * STAGGER
    r_stop = min(r_start + DUR, F_END)
    for f in (F_HOLD0, f_start):
        ob.location = (0, 0, 0); ob.keyframe_insert("location", frame=f)
    for f in (f_stop, F_HOLD1, r_start):
        ob.location = disp; ob.keyframe_insert("location", frame=f)
    ob.location = (0, 0, 0); ob.keyframe_insert("location", frame=r_stop)
    # Blender 5 slotted actions: default Bezier auto-clamped keys already ease in/out.

# ----------------------------------------------------- lights & camera ---
bb = PARTS["assembled_bbox_mm"]
centre = Vector([(bb["min"][i] + bb["max"][i]) / 2 for i in range(3)]) * MM
centre.y += 0.02   # bias toward the front
centre.z += 0.012  # v2: the exploded PCB stack rises above the frame

target = bpy.data.objects.new("target", None)
target.location = centre
scene.collection.objects.link(target)

cam_data = bpy.data.cameras.new("cam")
cam_data.lens = 50          # v2: wider so the exploded parts + labels fit at 16:9
cam_data.dof.use_dof = True
cam_data.dof.focus_object = target
cam_data.dof.aperture_fstop = 32.0
cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam)
scene.camera = cam
cam.location = centre + Vector((0.30, 0.26, 0.17))
tr = cam.constraints.new("TRACK_TO")
tr.target = target; tr.track_axis = "TRACK_NEGATIVE_Z"; tr.up_axis = "UP_Y"
# slow orbit over the shot
for f, ang in ((0, -12), (F_END, 12)):
    a = math.radians(ang)
    off = Vector((0.30, 0.26, 0.17))
    rot = Vector((off.x * math.cos(a) - off.y * math.sin(a),
                  off.x * math.sin(a) + off.y * math.cos(a), off.z))
    cam.location = centre + rot
    cam.keyframe_insert("location", frame=f)

def area_light(name, loc, energy, size, colour=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA")
    ld.energy = energy; ld.size = size; ld.color = colour
    lo = bpy.data.objects.new(name, ld)
    lo.location = loc
    scene.collection.objects.link(lo)
    c = lo.constraints.new("TRACK_TO")
    c.target = target; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
    return lo

area_light("key",  centre + Vector((0.35, 0.25, 0.45)), 18, 0.6, (1.0, 0.97, 0.92))
area_light("fill", centre + Vector((-0.45, 0.35, 0.15)), 6, 0.9, (0.9, 0.95, 1.0))
area_light("rim",  centre + Vector((0.0, -0.5, 0.35)), 12, 0.4)

world = bpy.data.worlds.new("studio"); scene.world = world
if not world.node_tree:
    world.use_nodes = True
wn = world.node_tree
env = wn.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(str(ASSETS / "hdri" / "studio_small_09_2k.hdr"))
bg = wn.nodes["Background"]; wn.links.new(env.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = 0.8
lp = wn.nodes.new("ShaderNodeLightPath"); mix = wn.nodes.new("ShaderNodeMixShader")
dark = wn.nodes.new("ShaderNodeBackground"); dark.inputs["Color"].default_value = (0.012, 0.013, 0.016, 1)
wn.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"]); wn.links.new(bg.outputs["Background"], mix.inputs[1])
wn.links.new(dark.outputs["Background"], mix.inputs[2]); wn.links.new(mix.outputs["Shader"], wn.nodes["World Output"].inputs["Surface"])

# ground plane with soft contact shadow
bpy.ops.mesh.primitive_plane_add(size=3.0, location=(centre.x, centre.y, bb["min"][2] * MM - 0.002))
ground = bpy.context.active_object
ground.name = "ground"
gm = bpy.data.materials.new("ground_mat")
if not gm.node_tree:
    gm.use_nodes = True
gm.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.03, 0.032, 0.036, 1)
gm.node_tree.nodes["Principled BSDF"].inputs["Roughness"].default_value = 0.6
ground.data.materials.append(gm)

# ------------------------------------------- leaders, mount rings, labels ---
# For every sensor / electronics / power / actuator part (and every electrode):
#   leader   thin light-grey line from the exploded part to its MOUNT point on
#            the frame (`mount_mm` in parts.json if present, else `pos_mm`).
#            Built as a bevelled 2-point curve; point 0 is hooked to the part
#            object so it follows the part exactly through the whole timeline.
#   ring     small torus at the mount point, normal along the explode direction
#   label    callout + part number as a text object 6 mm beyond the exploded
#            part along its explode direction, always facing the camera
# All three fade in with the explode (object-colour alpha + scale keyed on the
# part's own start/stop frames) and are hidden while assembled.
# Any part not in the table below is left alone -> new parts from parts.json
# (browpad_*, elec_*_{brow_med,...}, ppg_flex_*, cam_fpc_*) are handled by the
# same predicate without edits here.
def _is_sensor_like(p):
    return p["category"] in ("sensor", "electronics", "power", "actuator") or p["label"].startswith("elec")

def _alpha_mat(name, rgb, emission):
    """Emissive unlit-looking material whose alpha comes from the object colour."""
    m = bpy.data.materials.new(name)
    if not m.node_tree:
        m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*rgb, 1)
    b.inputs["Emission Color"].default_value = (*rgb, 1)
    b.inputs["Emission Strength"].default_value = emission
    b.inputs["Roughness"].default_value = 0.6
    b.inputs["Specular IOR Level"].default_value = 0.0
    oi = nt.nodes.new("ShaderNodeObjectInfo")
    nt.links.new(oi.outputs["Alpha"], b.inputs["Alpha"])
    m.surface_render_method = "DITHERED"
    m.use_backface_culling = False
    return m

MAT_LEADER = _alpha_mat("leader", (0.82, 0.84, 0.86), 0.6)
MAT_LABEL = _alpha_mat("label", (0.92, 0.93, 0.95), 1.2)

def _no_shadow(ob):
    for attr in ("visible_shadow", "visible_glossy", "visible_diffuse", "visible_transmission", "visible_volume_scatter"):
        try: setattr(ob, attr, False)
        except Exception: pass

def _key_fade(ob, f_start, f_stop, r_start, r_stop, scale_key=True):
    """alpha (object colour) and optional scale: 0 while assembled, 1 while exploded."""
    for f, a in ((F_HOLD0, 0.0), (f_start, 0.0), (f_stop, 1.0), (F_HOLD1, 1.0), (r_start, 1.0), (r_stop, 0.0)):
        ob.color = (1, 1, 1, a); ob.keyframe_insert("color", frame=f)
        if scale_key:
            ob.scale = (max(a, 1e-3),) * 3; ob.keyframe_insert("scale", frame=f)
        ob.hide_render = a <= 0.0 and f in (F_HOLD0, r_stop)
        ob.hide_viewport = ob.hide_render
        ob.keyframe_insert("hide_render", frame=f); ob.keyframe_insert("hide_viewport", frame=f)
    # unhide one frame after the part starts moving / hide one frame after it lands
    ob.hide_render = False; ob.hide_viewport = False
    ob.keyframe_insert("hide_render", frame=f_start + 1); ob.keyframe_insert("hide_viewport", frame=f_start + 1)

anno = bpy.data.collections.new("annotations"); scene.collection.children.link(anno)
label_count = 0; label_specs = []; occupiers = []
for label, (ob, p) in objs.items():
    if not _is_sensor_like(p):
        continue
    e = p["explode"]; d = Vector(e["dir"]).normalized(); dist = e["dist_mm"]
    pos = Vector(p["pos_mm"]); mount = Vector(p.get("mount_mm", p["pos_mm"]))
    k = e["order"]; f_start = F_EXPL0 + k * STAGGER; f_stop = min(f_start + DUR, F_EXPL1)
    r_start = F_HOLD1 + k * STAGGER; r_stop = min(r_start + DUR, F_END)
    # -- leader: 2-point poly curve, bevelled to r 0.15 mm, point 0 hooked to the part
    cu = bpy.data.curves.new(f"leader_{label}", "CURVE"); cu.dimensions = "3D"
    cu.bevel_depth = 0.15 * MM; cu.bevel_resolution = 4; cu.fill_mode = "FULL"; cu.use_fill_caps = True
    sp = cu.splines.new("POLY"); sp.points.add(1)
    sp.points[0].co = (*(pos * MM), 1.0); sp.points[1].co = (*(mount * MM), 1.0)
    lead = bpy.data.objects.new(f"leader_{label}", cu); anno.objects.link(lead)
    lead.data.materials.append(MAT_LEADER)
    hook = lead.modifiers.new("follow_part", "HOOK"); hook.object = ob
    hook.matrix_inverse = ob.matrix_world.inverted() if ob.matrix_world.determinant() else hook.matrix_inverse
    hook.vertex_indices_set([0])
    _no_shadow(lead); _key_fade(lead, f_start, f_stop, r_start, r_stop, scale_key=False)
    # -- ring at the mount point, normal along the explode direction
    bpy.ops.mesh.primitive_torus_add(major_radius=1.2 * MM, minor_radius=0.12 * MM, major_segments=32, minor_segments=8,
                                     location=mount * MM)
    ring = bpy.context.active_object; ring.name = f"mount_{label}"
    for c in list(ring.users_collection): c.objects.unlink(ring)
    anno.objects.link(ring)
    ring.rotation_mode = "QUATERNION"; ring.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(d)
    ring.data.materials.append(MAT_LEADER); _no_shadow(ring)
    _key_fade(ring, f_start, f_stop, r_start, r_stop)
    # -- callout label: 6 mm beyond the exploded part along its explode direction
    bbm = p.get("bbox_mm"); half = 0.0; ext = Vector((3, 3, 3))
    if bbm:
        ext = Vector([(bbm["max"][i] - bbm["min"][i]) / 2 for i in range(3)])
        half = abs(ext.x * d.x) + abs(ext.y * d.y) + abs(ext.z * d.z)
    body = f"{p.get('callout', label)}\n{p.get('part_number', '')}".rstrip()
    label_specs.append(dict(label=label, body=body, anchor=pos + d * (dist + half + 6.0),
                            frames=(f_start, f_stop, r_start, r_stop)))
    occupiers.append((pos + d * dist, ext.length))          # exploded part centre, radius (mm)
    label_count += 1

# labels are de-overlapped in the camera plane of the exploded hold (frame 100):
# shift along camera-up until the text rectangle clears every exploded part and
# every label already placed
def _cam_basis(frame):
    a = math.radians(-12 + 24 * frame / F_END); off = Vector((0.30, 0.26, 0.17))
    cpos = centre + Vector((off.x * math.cos(a) - off.y * math.sin(a), off.x * math.sin(a) + off.y * math.cos(a), off.z))
    view = (centre - cpos).normalized(); right = view.cross(Vector((0, 0, 1))).normalized(); up = right.cross(view).normalized()
    return right, up
RIGHT, UP = _cam_basis((F_EXPL1 + F_HOLD1) // 2)
TXT = 2.2                                                  # mm
placed = []                                                # (centre_mm, half_w, half_h)
for spec in sorted(label_specs, key=lambda s: -s["anchor"].dot(UP)):
    lines = spec["body"].split("\n")
    hw = 0.5 * max(len(l) for l in lines) * 0.55 * TXT + 1.0; hh = 0.5 * len(lines) * TXT * 1.15 + 1.0
    def clear(c):
        for q, r in occupiers:
            dv = c - q
            if abs(dv.dot(RIGHT)) < hw + r * 0.8 and abs(dv.dot(UP)) < hh + r * 0.8:
                return False
        for q, w2, h2 in placed:
            dv = c - q
            if abs(dv.dot(RIGHT)) < hw + w2 and abs(dv.dot(UP)) < hh + h2:
                return False
        return True
    c = spec["anchor"].copy(); step = 0
    while not clear(c) and step < 24:
        step += 1
        c = spec["anchor"] + UP * (2.0 * hh * ((step + 1) // 2) * (1 if step % 2 else -1))
    placed.append((c, hw, hh))
    tx = bpy.data.curves.new(f"label_{spec['label']}", "FONT")
    tx.body = spec["body"]; tx.size = TXT * MM; tx.align_x = "CENTER"; tx.align_y = "CENTER"; tx.space_line = 1.15
    txo = bpy.data.objects.new(f"label_{spec['label']}", tx); anno.objects.link(txo)
    txo.location = c * MM
    txo.data.materials.append(MAT_LABEL); _no_shadow(txo)
    con = txo.constraints.new("TRACK_TO"); con.target = cam; con.track_axis = "TRACK_Z"; con.up_axis = "UP_Y"
    _key_fade(txo, *spec["frames"])
    if step:
        # thin tick from the label back to its anchor so a shifted label still reads
        cu2 = bpy.data.curves.new(f"tick_{spec['label']}", "CURVE"); cu2.dimensions = "3D"
        cu2.bevel_depth = 0.08 * MM; cu2.bevel_resolution = 3; cu2.fill_mode = "FULL"
        sp2 = cu2.splines.new("POLY"); sp2.points.add(1)
        sp2.points[0].co = (*((c - UP * hh) * MM), 1.0); sp2.points[1].co = (*(spec["anchor"] * MM), 1.0)
        tk = bpy.data.objects.new(f"tick_{spec['label']}", cu2); anno.objects.link(tk)
        tk.data.materials.append(MAT_LEADER); _no_shadow(tk); _key_fade(tk, *spec["frames"], scale_key=False)
print(f"annotations: {label_count} sensor-like parts got leader + ring + label")

# --------------------------------------------------------------- render ---
scene.render.image_settings.file_format = "PNG"
if ANIM:
    # Blender 5.2 has no FFMPEG image format: render a PNG sequence, encode with
    # sim/render/encode_mp4.py (imageio-ffmpeg) afterwards
    seq = RENDER_DIR / f"explode_seq_{ENGINE.lower()}"; seq.mkdir(exist_ok=True)
    scene.render.filepath = str(seq / "frame_")
    bpy.ops.render.render(animation=True)
    print("wrote PNG sequence", seq)
else:
    for f in FRAMES:
        scene.frame_set(f)
        scene.render.filepath = str(RENDER_DIR / f"explode_{ENGINE.lower()}_f{f:03d}.png")
        bpy.ops.render.render(write_still=True)
        print("wrote", scene.render.filepath)

# pack the HDRI so the .blend opens self-contained on the client's machine
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(RENDER_DIR / "explode.blend"))
