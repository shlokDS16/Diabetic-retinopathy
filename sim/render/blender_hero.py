"""
Hero renders of the NRDI glasses: product shots and on-face shots, Cycles/OptiX.

Shared with blender_explode.py: parts come from sim/out/parts_stl + parts.json
(project frame, Z-up, mm -> scaled to metres). Adds:
  - Poly Haven studio HDRI (CC0) world + soft area key/rim
  - acetate / glass / stainless / soldermask / silicon material set
  - curved dark cyclorama backdrop with contact shadows
  - optional head (Lee Perry-Smith scan, CC BY 3.0) aligned so the pupils sit at
    (+-IPD/2, 0, 0) of the project frame
  - shots: "product" (3/4 front macro), "face" (on-head 3/4), "detail" (nasal
    sensor cluster), "align" (orthographic front view with pupil markers, for
    checking head alignment)

Usage:
    blender -b --python sim/render/blender_hero.py -- --shot product [--head]
        [--res 1920x1080] [--samples 256] [--engine CYCLES|EEVEE] [--out DIR]
"""
import bpy, json, math, sys, pathlib
from mathutils import Vector, Euler

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "sim" / "out"
ASSETS = ROOT / "sim" / "assets"
PARTS = json.loads((OUT_DIR / "parts.json").read_text(encoding="utf-8"))
STL_DIR = OUT_DIR / "parts_stl"

argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default
SHOT = arg("--shot", "product")
HEAD = "--head" in argv or SHOT in ("face", "align")
ENGINE = arg("--engine", "CYCLES").upper()
RES = tuple(int(v) for v in arg("--res", "1920x1080").split("x"))
SAMPLES = int(arg("--samples", "256"))
RENDER_DIR = pathlib.Path(arg("--out", str(ROOT / "sim" / "render" / "out")))
RENDER_DIR.mkdir(parents=True, exist_ok=True)
MM = 0.001
IPD = PARTS["coordinate_system"].get("ipd_mm", 63.0) if isinstance(PARTS.get("coordinate_system"), dict) else 63.0

# Head alignment (project frame, mm). Tune with --shot align.
HEAD_SCALE = float(arg("--head-scale", "42.4"))       # scan units -> mm
HEAD_OFFSET = Vector([float(v) for v in arg("--head-offset", "0,-85,-79").split(",")])
HEAD_ROT_DEG = [float(v) for v in arg("--head-rot", "0,0,180").split(",")]   # scan faces -Y

# ------------------------------------------------------------------ scene ---
bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = RES
scene.view_settings.view_transform = "AgX"
scene.view_settings.look = "AgX - Medium High Contrast"
scene.render.film_transparent = False
if ENGINE == "CYCLES":
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"; prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    scene.cycles.device = "GPU"
    scene.cycles.samples = SAMPLES
    scene.cycles.use_denoising = True
    scene.cycles.max_bounces = 12
    scene.cycles.transmission_bounces = 12
    scene.cycles.transparent_max_bounces = 16
    scene.cycles.caustics_reflective = False
    scene.cycles.caustics_refractive = False
else:
    scene.render.engine = "BLENDER_EEVEE"
    scene.eevee.taa_render_samples = SAMPLES
    scene.eevee.use_raytracing = True

# ------------------------------------------------------------- materials ----
# Shared with blender_explode.py / blender_physics.py: sim/render/materials.py
# is the single material set (acetate, CR-39, stainless, soldermask, IC,
# window, IR-LED, electrode, silicone, polyimide, gold) and maps parts by label.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
from materials import _mat, material_for   # noqa: E402

# --------------------------------------------------------------- import ----
rig = bpy.data.objects.new("nrdi_rig", None)
scene.collection.objects.link(rig)
objs = {}
for p in PARTS["parts"]:
    path = STL_DIR / f"{p['label']}.stl"
    if not path.exists():
        continue
    bpy.ops.wm.stl_import(filepath=str(path), global_scale=MM, forward_axis="Y", up_axis="Z")
    ob = bpy.context.selected_objects[0]
    ob.name = p["label"]; ob.parent = rig
    ob.data.materials.append(material_for(p))
    bpy.context.view_layer.objects.active = ob
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))
    except Exception:
        for poly in ob.data.polygons:
            poly.use_smooth = True
    objs[p["label"]] = ob

# ------------------------------------------------------------------ head ----
head = None
if HEAD:
    bpy.ops.import_scene.gltf(filepath=str(ASSETS / "head" / "LeePerrySmith.glb"))
    head = [o for o in bpy.context.selected_objects if o.type == "MESH"][0]
    head.name = "head"
    # the importer parents the mesh under a rotated empty: bake that away so the
    # pose arguments below act in world space
    bpy.ops.object.select_all(action="DESELECT")
    head.select_set(True); bpy.context.view_layer.objects.active = head
    bpy.ops.object.parent_clear(type="CLEAR_KEEP_TRANSFORM")
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)
    head.rotation_mode = "XYZ"          # importer leaves QUATERNION mode
    head.rotation_euler = Euler([math.radians(a) for a in HEAD_ROT_DEG])
    head.scale = (HEAD_SCALE * MM,) * 3
    head.location = HEAD_OFFSET * MM
    # skin material: colour map + random-walk SSS
    m, b, nt = _mat("skin")
    img = nt.nodes.new("ShaderNodeTexImage")
    img.image = bpy.data.images.load(str(ASSETS / "head" / "Map-COL.jpg"))
    # the scan's UVs are glTF-convention; flip V so the map lands on the face
    uv = nt.nodes.new("ShaderNodeUVMap"); mp = nt.nodes.new("ShaderNodeMapping")
    mp.inputs["Location"].default_value = (0.0, 1.0, 0.0); mp.inputs["Scale"].default_value = (1.0, -1.0, 1.0)
    nt.links.new(uv.outputs["UV"], mp.inputs["Vector"]); nt.links.new(mp.outputs["Vector"], img.inputs["Vector"])
    nt.links.new(img.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Subsurface Weight"].default_value = 1.0
    b.inputs["Subsurface Radius"].default_value = (0.0030, 0.00075, 0.0004)   # skin dmfp (m)
    b.inputs["Subsurface Scale"].default_value = 0.5
    b.inputs["Roughness"].default_value = 0.45
    b.inputs["Coat Weight"].default_value = 0.15
    b.inputs["Coat Roughness"].default_value = 0.35
    try:
        b.subsurface_method = "RANDOM_WALK_SKIN"
    except Exception:
        pass
    head.data.materials.clear()
    head.data.materials.append(m)
    bpy.context.view_layer.objects.active = head
    try:
        bpy.ops.object.shade_smooth()
    except Exception:
        pass

# ------------------------------------------------------- world & lights ----
world = bpy.data.worlds.new("studio"); scene.world = world
if not world.node_tree:
    world.use_nodes = True
wn = world.node_tree
env = wn.nodes.new("ShaderNodeTexEnvironment")
env.image = bpy.data.images.load(str(ASSETS / "hdri" / "studio_small_09_2k.hdr"))
bg = wn.nodes["Background"]
bg.inputs["Strength"].default_value = 1.0
wn.links.new(env.outputs["Color"], bg.inputs["Color"])
# hide the HDRI from the camera: black backdrop behind, HDRI only for reflections
lp = wn.nodes.new("ShaderNodeLightPath")
mix = wn.nodes.new("ShaderNodeMixShader")
dark = wn.nodes.new("ShaderNodeBackground"); dark.inputs["Color"].default_value = (0.012, 0.013, 0.016, 1)
out = wn.nodes["World Output"]
wn.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"])
wn.links.new(bg.outputs["Background"], mix.inputs[1])
wn.links.new(dark.outputs["Background"], mix.inputs[2])
wn.links.new(mix.outputs["Shader"], out.inputs["Surface"])
mapping = wn.nodes.new("ShaderNodeMapping"); coord = wn.nodes.new("ShaderNodeTexCoord")
wn.links.new(coord.outputs["Generated"], mapping.inputs["Vector"])
wn.links.new(mapping.outputs["Vector"], env.inputs["Vector"])
mapping.inputs["Rotation"].default_value = (0, 0, math.radians(35))

bb = PARTS["assembled_bbox_mm"]
centre = Vector([(bb["min"][i] + bb["max"][i]) / 2 for i in range(3)]) * MM
front = Vector((0.0, bb["max"][1] * MM, 0.0))            # frame front, between the lenses

def area_light(name, loc, energy, size, colour=(1, 1, 1), target=None):
    ld = bpy.data.lights.new(name, "AREA"); ld.energy = energy; ld.size = size; ld.color = colour
    lo = bpy.data.objects.new(name, ld); lo.location = loc; scene.collection.objects.link(lo)
    c = lo.constraints.new("TRACK_TO"); c.target = target; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
    return lo

target = bpy.data.objects.new("target", None); scene.collection.objects.link(target)

# curved cyclorama: large plane bent up at the back (only for product shots)
if not HEAD:
    bpy.ops.mesh.primitive_plane_add(size=2.0, location=(0, -0.6, bb["min"][2] * MM - 0.002))
    cyc = bpy.context.active_object; cyc.name = "cyc"
    bpy.ops.object.mode_set(mode="EDIT"); bpy.ops.mesh.subdivide(number_cuts=40); bpy.ops.object.mode_set(mode="OBJECT")
    for v in cyc.data.vertices:
        if v.co.y < -0.6:
            t = (-0.6 - v.co.y) / 0.4
            v.co.z += 0.6 * t * t
    bpy.ops.object.shade_smooth()
    cm, cb, _ = _mat("cyc"); cb.inputs["Base Color"].default_value = (0.03, 0.032, 0.036, 1)
    cb.inputs["Roughness"].default_value = 0.7
    cyc.data.materials.append(cm)

# ----------------------------------------------------------------- shots ----
cam_data = bpy.data.cameras.new("cam"); cam = bpy.data.objects.new("cam", cam_data)
scene.collection.objects.link(cam); scene.camera = cam
tr = cam.constraints.new("TRACK_TO"); tr.target = target; tr.track_axis = "TRACK_NEGATIVE_Z"; tr.up_axis = "UP_Y"
cam_data.dof.use_dof = True; cam_data.dof.focus_object = target

if SHOT == "product":
    target.location = centre + Vector((0.0, 0.05, 0.0))
    cam_data.lens = 85; cam_data.dof.aperture_fstop = 8.0
    cam.location = target.location + Vector((0.22, 0.30, 0.13))
    area_light("key", target.location + Vector((0.3, 0.3, 0.5)), 60, 0.8, (1, 0.97, 0.93), target)
    area_light("rim", target.location + Vector((-0.3, -0.5, 0.35)), 40, 0.5, (0.85, 0.92, 1), target)
    area_light("fill", target.location + Vector((-0.5, 0.4, 0.1)), 15, 1.2, target=target)
elif SHOT == "detail":
    nasal = Vector((0.020, 0.012, 0.0))
    target.location = nasal
    cam_data.lens = 100; cam_data.dof.aperture_fstop = 4.0
    cam.location = nasal + Vector((0.10, 0.14, 0.06))
    area_light("key", nasal + Vector((0.2, 0.25, 0.3)), 25, 0.5, (1, 0.97, 0.93), target)
    area_light("rim", nasal + Vector((-0.2, -0.3, 0.2)), 20, 0.4, (0.85, 0.92, 1), target)
elif SHOT == "face":
    target.location = Vector((0.0, 0.0, 0.0))
    cam_data.lens = 85; cam_data.dof.aperture_fstop = 5.6
    cam.location = Vector((0.28, 0.42, 0.10))
    area_light("key", Vector((0.5, 0.5, 0.6)), 120, 1.0, (1, 0.96, 0.92), target)
    area_light("rim", Vector((-0.4, -0.4, 0.5)), 60, 0.6, (0.85, 0.92, 1), target)
    area_light("fill", Vector((-0.6, 0.5, 0.0)), 30, 1.5, target=target)
elif SHOT == "align":
    # orthographic front view with pupil markers at (+-IPD/2, 0, 0)
    cam_data.type = "ORTHO"; cam_data.ortho_scale = 0.26; cam_data.dof.use_dof = False
    target.location = Vector((0.0, 0.0, 0.0)); cam.location = Vector((0.0, 0.6, 0.0))
    for sx in (+1, -1):
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.003, location=(sx * IPD / 2 * MM, 0.02, 0.0))
        mk = bpy.context.active_object; mm_, mb, _ = _mat(f"marker{sx}")
        mb.inputs["Emission Color"].default_value = (1, 0, 0, 1); mb.inputs["Emission Strength"].default_value = 20
        mk.data.materials.append(mm_)
    area_light("key", Vector((0.3, 0.6, 0.3)), 100, 1.5, target=target)
    for ob in objs.values():
        ob.hide_render = ob.name not in ("rim_L", "rim_R", "bridge")

# ------------------------------------------------------------------ render ---
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(RENDER_DIR / f"hero_{SHOT}.png")
bpy.ops.render.render(write_still=True)
print("wrote", scene.render.filepath)
if head is not None:
    d = head.dimensions
    print(f"head dims (m): {d.x:.3f} x {d.y:.3f} x {d.z:.3f}  location {tuple(round(v,4) for v in head.location)}")
bpy.ops.wm.save_as_mainfile(filepath=str(RENDER_DIR / f"hero_{SHOT}.blend"))
