"""
Physics renders on the anatomical model (Cycles/OptiX): the "science" shots.

  thermal   temperature-coloured skin/cornea surface (skin_T.ply vertex colours),
            glasses ghosted as glass, thermopile FOV cone, and a horizontal cut
            plane (slice_T_z0.png) revealing the orbit temperature field.
  optical   PPG measurement-density banana as an emissive OpenVDB volume under
            the temple over the superficial temporal artery, skin ghosted.
  electrical bioimpedance lead-field density on the skin (leadfield.ply) with the
            electrode pads and current-path glow.
  thermography  the skin temperature projected on the real face (face_T.ply from
            thermography_face.py), glasses in real materials, DCI/BAA cones,
            camera-parented inferno colour bar. "Image 2" for the client deck.

Everything is placed in the project frame (mm -> m). Uses the same HDRI and
material set as blender_hero.py so the physics shots match the product shots.

Usage:
    blender -b --python sim/render/blender_physics.py -- --shot thermal
        [--res 1920x1080] [--samples 256] [--engine CYCLES|EEVEE]
"""
import bpy, json, math, sys, pathlib
import numpy as np
from mathutils import Vector

ROOT = pathlib.Path(__file__).resolve().parents[2]
OUT = ROOT / "sim" / "out"; FDIR = OUT / "fields"; ASSETS = ROOT / "sim" / "assets"
PARTS = json.loads((OUT / "parts.json").read_text(encoding="utf-8"))
LAYOUT = json.loads((OUT / "sensor_layout.json").read_text(encoding="utf-8"))
argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
def arg(name, default):
    return argv[argv.index(name) + 1] if name in argv else default
SHOT = arg("--shot", "thermal")
ENGINE = arg("--engine", "CYCLES").upper()
RES = tuple(int(v) for v in arg("--res", "1920x1080").split("x"))
SAMPLES = int(arg("--samples", "256"))
RENDER_DIR = ROOT / "sim" / "render" / "out"; RENDER_DIR.mkdir(parents=True, exist_ok=True)
MM = 0.001

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
scene.render.resolution_x, scene.render.resolution_y = RES
scene.view_settings.view_transform = "AgX"; scene.view_settings.look = "AgX - Medium High Contrast"
if ENGINE == "CYCLES":
    scene.render.engine = "CYCLES"
    prefs = bpy.context.preferences.addons["cycles"].preferences
    prefs.compute_device_type = "OPTIX"; prefs.get_devices()
    for d in prefs.devices:
        d.use = d.type == "OPTIX"
    scene.cycles.device = "GPU"; scene.cycles.samples = SAMPLES; scene.cycles.use_denoising = True
    scene.cycles.volume_bounces = 2; scene.cycles.volume_step_rate = 0.5
else:
    scene.render.engine = "BLENDER_EEVEE"; scene.eevee.taa_render_samples = SAMPLES

def _mat(name):
    m = bpy.data.materials.new(name)
    if not m.node_tree:
        m.use_nodes = True
    return m, m.node_tree.nodes["Principled BSDF"], m.node_tree

# ---------------------------------------------------------------- world ----
world = bpy.data.worlds.new("w"); scene.world = world
if not world.node_tree:
    world.use_nodes = True
wn = world.node_tree
env = wn.nodes.new("ShaderNodeTexEnvironment"); env.image = bpy.data.images.load(str(ASSETS / "hdri" / "studio_small_09_2k.hdr"))
bg = wn.nodes["Background"]; wn.links.new(env.outputs["Color"], bg.inputs["Color"]); bg.inputs["Strength"].default_value = 0.35
lp = wn.nodes.new("ShaderNodeLightPath"); mix = wn.nodes.new("ShaderNodeMixShader")
dark = wn.nodes.new("ShaderNodeBackground"); dark.inputs["Color"].default_value = (0.010, 0.011, 0.014, 1)
wn.links.new(lp.outputs["Is Camera Ray"], mix.inputs["Fac"]); wn.links.new(bg.outputs["Background"], mix.inputs[1])
wn.links.new(dark.outputs["Background"], mix.inputs[2]); wn.links.new(mix.outputs["Shader"], wn.nodes["World Output"].inputs["Surface"])

# ------------------------------------------------------------- helpers ----
def import_ply(path, name, emission=0.0, alpha=1.0):
    bpy.ops.wm.ply_import(filepath=str(path), global_scale=MM, forward_axis="Y", up_axis="Z")
    ob = bpy.context.selected_objects[0]; ob.name = name
    m, b, nt = _mat(f"mat_{name}")
    col = nt.nodes.new("ShaderNodeVertexColor")
    ca = ob.data.color_attributes
    col.layer_name = ca[0].name if len(ca) else "Col"
    nt.links.new(col.outputs["Color"], b.inputs["Base Color"])
    b.inputs["Roughness"].default_value = 0.7
    b.inputs["Specular IOR Level"].default_value = 0.2
    if emission > 0:
        nt.links.new(col.outputs["Color"], b.inputs["Emission Color"]); b.inputs["Emission Strength"].default_value = emission
    if alpha < 1:
        b.inputs["Alpha"].default_value = alpha; m.surface_render_method = "DITHERED"
    ob.data.materials.append(m)
    bpy.context.view_layer.objects.active = ob
    try:
        bpy.ops.object.shade_smooth_by_angle(angle=math.radians(40))
    except Exception:
        pass
    return ob

def glasses(ghost=True):
    rig = bpy.data.objects.new("glasses", None); scene.collection.objects.link(rig)
    m, b, nt = _mat("ghost")
    b.inputs["Base Color"].default_value = (0.9, 0.95, 1.0, 1); b.inputs["Transmission Weight"].default_value = 1.0
    b.inputs["Roughness"].default_value = 0.1; b.inputs["IOR"].default_value = 1.15; b.inputs["Alpha"].default_value = 0.22
    m.surface_render_method = "DITHERED"
    for p in PARTS["parts"]:
        path = OUT / "parts_stl" / f"{p['label']}.stl"
        if not path.exists():
            continue
        bpy.ops.wm.stl_import(filepath=str(path), global_scale=MM, forward_axis="Y", up_axis="Z")
        ob = bpy.context.selected_objects[0]; ob.name = "g_" + p["label"]; ob.parent = rig
        ob.data.materials.append(m)
    return rig

def sensor(name):
    return next(s for s in LAYOUT["sensors"] if s["name"] == name)

def cone(name, s, length_mm, colour, fov=None, alpha=0.15, emission=3.0):
    """Emissive FOV cone from a sensor pose."""
    fov = s["fov_deg"] if fov is None else fov
    r = length_mm * math.tan(math.radians(fov / 2))
    bpy.ops.mesh.primitive_cone_add(radius1=0.3 * MM, radius2=r * MM, depth=length_mm * MM)
    ob = bpy.context.active_object; ob.name = name
    ax = Vector(s["axis"]).normalized()
    ob.rotation_mode = "QUATERNION"; ob.rotation_quaternion = Vector((0, 0, 1)).rotation_difference(ax)
    ob.location = Vector(s["pos"]) * MM + ax * length_mm * MM / 2
    m, b, nt = _mat(f"mat_{name}")
    b.inputs["Base Color"].default_value = (*colour, 1); b.inputs["Emission Color"].default_value = (*colour, 1)
    b.inputs["Emission Strength"].default_value = emission; b.inputs["Alpha"].default_value = alpha
    m.surface_render_method = "DITHERED"; ob.data.materials.append(m)
    return ob

def slice_plane(png, meta, name):
    """Textured plane carrying a field slice (alpha outside tissue)."""
    u0, u1 = meta["u_range_mm"]; v0, v1 = meta["v_range_mm"]
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    ob = bpy.context.active_object; ob.name = name
    ob.scale = ((u1 - u0) * MM, (v1 - v0) * MM, 1.0)
    ax = meta["plane"]
    centre = [0, 0, 0]; centre["xyz".index(meta["u_axis"])] = (u0 + u1) / 2; centre["xyz".index(meta["v_axis"])] = (v0 + v1) / 2
    centre["xyz".index(ax)] = meta["value_mm"]
    ob.location = Vector(centre) * MM
    if ax == "x":
        ob.rotation_euler = (math.radians(90), 0, math.radians(90))
    elif ax == "y":
        ob.rotation_euler = (math.radians(90), 0, 0)
    m, b, nt = _mat(f"mat_{name}")
    img = nt.nodes.new("ShaderNodeTexImage"); img.image = bpy.data.images.load(str(png)); img.extension = "CLIP"
    nt.links.new(img.outputs["Color"], b.inputs["Base Color"]); nt.links.new(img.outputs["Color"], b.inputs["Emission Color"])
    nt.links.new(img.outputs["Alpha"], b.inputs["Alpha"]); b.inputs["Emission Strength"].default_value = 0.25
    b.inputs["Roughness"].default_value = 0.9; m.surface_render_method = "DITHERED"
    ob.data.materials.append(m)
    return ob

def vdb_volume(npy, meta, name, colour=(1.0, 0.45, 0.1)):
    import openvdb as vdb
    arr = np.load(npy).astype(np.float32)
    grid = vdb.FloatGrid(); grid.copyFromArray(arr); grid.name = "density"
    grid.transform = vdb.createLinearTransform(voxelSize=meta["res_mm"] * MM)
    path = str(FDIR / f"{name}.vdb"); vdb.write(path, grids=[grid])
    r = bpy.ops.object.volume_import(filepath=path)
    ob = bpy.context.selected_objects[0] if bpy.context.selected_objects else bpy.context.active_object
    ob.name = name; ob.location = Vector(meta["origin_mm"]) * MM
    print("VDB import:", r, ob.type, [g.name for g in ob.data.grids] if ob.type == "VOLUME" else "not a volume",
          "grid max", float(arr.max()), "voxels>0.55", int((arr > 0.55).sum()))
    m = bpy.data.materials.new(f"mat_{name}")
    if not m.node_tree:
        m.use_nodes = True
    nt = m.node_tree
    for n in list(nt.nodes):
        nt.nodes.remove(n)
    out = nt.nodes.new("ShaderNodeOutputMaterial"); pv = nt.nodes.new("ShaderNodeVolumePrincipled")
    pv.inputs["Density"].default_value = 250.0; pv.inputs["Emission Strength"].default_value = 60.0
    pv.inputs["Emission Color"].default_value = (*colour, 1); pv.inputs["Color"].default_value = (*colour, 1)
    pv.inputs["Absorption Color"].default_value = (0.2, 0.05, 0.0, 1)
    attr = nt.nodes.new("ShaderNodeAttribute"); attr.attribute_name = "density"
    ramp = nt.nodes.new("ShaderNodeMapRange"); ramp.inputs["From Min"].default_value = 0.35; ramp.inputs["From Max"].default_value = 1.0
    mul = nt.nodes.new("ShaderNodeMath"); mul.operation = "MULTIPLY"; mul.inputs[1].default_value = 250.0
    nt.links.new(attr.outputs["Fac"], ramp.inputs["Value"]); nt.links.new(ramp.outputs["Result"], mul.inputs[0])
    nt.links.new(mul.outputs["Value"], pv.inputs["Density"])          # density = 250 * remapped attribute
    em = nt.nodes.new("ShaderNodeMath"); em.operation = "MULTIPLY"; em.inputs[1].default_value = 80.0
    nt.links.new(ramp.outputs["Result"], em.inputs[0]); nt.links.new(em.outputs["Value"], pv.inputs["Emission Strength"])
    nt.links.new(pv.outputs["Volume"], out.inputs["Volume"])
    ob.data.materials.append(m)
    return ob

def camera(loc, target, lens=85, fstop=8.0):
    tgt = bpy.data.objects.new("target", None); tgt.location = target; scene.collection.objects.link(tgt)
    cd = bpy.data.cameras.new("cam"); cd.lens = lens; cd.dof.use_dof = True; cd.dof.focus_object = tgt; cd.dof.aperture_fstop = fstop
    cam = bpy.data.objects.new("cam", cd); cam.location = loc; scene.collection.objects.link(cam); scene.camera = cam
    c = cam.constraints.new("TRACK_TO"); c.target = tgt; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"
    return tgt

def light(name, loc, energy, size, target, colour=(1, 1, 1)):
    ld = bpy.data.lights.new(name, "AREA"); ld.energy = energy; ld.size = size; ld.color = colour
    lo = bpy.data.objects.new(name, ld); lo.location = loc; scene.collection.objects.link(lo)
    c = lo.constraints.new("TRACK_TO"); c.target = target; c.track_axis = "TRACK_NEGATIVE_Z"; c.up_axis = "UP_Y"

palette = json.loads((FDIR / "palette.json").read_text(encoding="utf-8"))

# ----------------------------------------------------------------- shots ----
if SHOT == "thermal":
    skin = import_ply(FDIR / "skin_T.ply", "skin_T", emission=0.35)
    # sagittal cut through the pupil axis: remove the lateral half (x > 31.5 mm) so the
    # temperature slice through the globe and orbit is exposed in profile
    # window: delete skin faces lateral of the pupil axis within y in [-50, 8], z in [-32, 32]
    # (a boolean on an open surface leaves a cap coincident with the slice plane)
    import bmesh
    bm = bmesh.new(); bm.from_mesh(skin.data)
    kill = [f for f in bm.faces if (f.calc_center_median().x > 0.0315 and -0.050 < f.calc_center_median().y < 0.008
                                     and -0.032 < f.calc_center_median().z < 0.032)]
    bmesh.ops.delete(bm, geom=kill, context="FACES"); bm.to_mesh(skin.data); bm.free()
    slice_plane(FDIR / "slice_T_x31.png", palette["slices"]["slice_T_x31"], "slice_x31")
    glasses()
    cone("fov_thermo_DCI", sensor("thermo_L"), 14.0, (1.0, 0.55, 0.15), fov=5.0)
    tgt = camera(Vector((0.26, 0.20, 0.12)), Vector((0.030, -0.018, 0.0)), lens=70, fstop=11)
    light("key", Vector((0.3, 0.15, 0.3)), 25, 0.6, tgt, (1, 0.96, 0.9)); light("rim", Vector((-0.1, -0.35, 0.25)), 15, 0.5, tgt, (0.8, 0.9, 1))
elif SHOT == "optical":
    skin = import_ply(FDIR / "skin_T.ply", "skin", alpha=0.05)
    m = skin.data.materials[0]; b = m.node_tree.nodes["Principled BSDF"]
    for l in list(m.node_tree.links):
        if l.to_socket == b.inputs["Base Color"]:
            m.node_tree.links.remove(l)
    b.inputs["Base Color"].default_value = (0.35, 0.22, 0.18, 1)
    meta = json.loads((FDIR / "banana_880.json").read_text(encoding="utf-8"))
    vdb_volume(FDIR / "banana_880.npy", meta, "banana_880")
    glasses()
    s = sensor("ppg_L"); p = Vector(s["pos"]) * MM
    tgt = camera(p + Vector((0.16, 0.10, 0.08)), p + Vector((-0.004, 0.0, 0.0)), lens=85, fstop=8.0)
    light("key", p + Vector((0.2, 0.1, 0.25)), 20, 0.5, tgt); light("rim", p + Vector((-0.1, -0.3, 0.2)), 15, 0.4, tgt, (0.8, 0.9, 1))
elif SHOT == "electrical":
    skin = import_ply(FDIR / "leadfield.ply", "leadfield", emission=0.5)
    glasses()
    tgt = camera(Vector((0.20, 0.17, 0.10)), Vector((0.03, -0.01, 0.0)), lens=70, fstop=11)
    light("key", Vector((0.25, 0.2, 0.35)), 40, 0.6, tgt, (1, 0.96, 0.9)); light("rim", Vector((-0.2, -0.3, 0.3)), 25, 0.5, tgt, (0.8, 0.9, 1))

elif SHOT == "thermography":
    # "image 2": the Pennes skin temperature projected on the real face (face_T.ply from
    # thermography_face.py), glasses in their real materials, thermopile DCI + BAA cones,
    # camera-parented colour bar. Same head pose as blender_hero.py --shot face.
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
    import materials as M
    face = import_ply(FDIR / "face_T.ply", "face_T", emission=0.5)
    fb = face.data.materials[0].node_tree.nodes["Principled BSDF"]
    fb.inputs["Roughness"].default_value = 0.55; fb.inputs["Specular IOR Level"].default_value = 0.3
    fb.inputs["Subsurface Weight"].default_value = 0.25          # light SSS so the colour field looks like skin, not paint
    fb.inputs["Subsurface Radius"].default_value = (0.0012, 0.0006, 0.0003); fb.inputs["Subsurface Scale"].default_value = 0.4
    try:
        fb.subsurface_method = "RANDOM_WALK"
    except Exception:
        pass
    # glasses with the real material set (not the ghost)
    rig = bpy.data.objects.new("glasses", None); scene.collection.objects.link(rig)
    for p in PARTS["parts"]:
        path = OUT / "parts_stl" / f"{p['label']}.stl"
        if not path.exists():
            continue
        bpy.ops.wm.stl_import(filepath=str(path), global_scale=MM, forward_axis="Y", up_axis="Z")
        ob = bpy.context.selected_objects[0]; ob.name = p["label"]; ob.parent = rig
        ob.data.materials.append(M.material_for(p))
        bpy.context.view_layer.objects.active = ob
        try:
            bpy.ops.object.shade_smooth_by_angle(angle=math.radians(35))
        except Exception:
            pass
    cone("fov_thermo_DCI", sensor("thermo_L"), 14.0, (1.0, 0.55, 0.15), fov=5.0)
    cone("fov_thermo_BAA", sensor("thermo_L"), 14.0, (1.0, 0.65, 0.35), fov=90.0, alpha=0.06, emission=1.0)
    bg.inputs["Strength"].default_value = 0.18                    # HDRI dimmed: the face is self-lit by the field
    tgt = camera(Vector((0.39, 0.60, 0.13)), Vector((0.0, 0.0, 0.0)), lens=100, fstop=16.0)   # 0.73 m out so the 0.6 m colour bar sits in front of the face
    scene.camera.data.dof.use_dof = False                        # keep the camera-parented colour bar sharp
    light("key", Vector((0.5, 0.5, 0.6)), 14, 1.0, tgt, (1, 0.96, 0.92)); light("rim", Vector((-0.4, -0.4, 0.5)), 10, 0.6, tgt, (0.85, 0.92, 1))
    light("fill", Vector((-0.6, 0.5, 0.0)), 4, 1.5, tgt)      # low key: the field is the light source, AgX must not wash the inferno to pink
    # ---- colour bar, parented to the camera at z = -0.6 m (camera space) ----
    cam = scene.camera; hw = 0.6 * 18.0 / 100.0; hh = hw * RES[1] / RES[0]      # half frame width/height at 0.6 m
    bar_w, bar_h = 0.050, 0.0045
    bx, by = -hw + 0.020 + bar_w / 2, -hh + 0.014
    def cam_child(ob, x, y):
        ob.parent = cam; ob.matrix_parent_inverse.identity(); ob.location = (x, y, -0.6); ob.rotation_euler = (0, 0, 0)
        for attr in ("visible_shadow", "visible_glossy", "visible_diffuse"):
            try: setattr(ob, attr, False)
            except Exception: pass
    bpy.ops.mesh.primitive_plane_add(size=1.0)
    bar = bpy.context.active_object; bar.name = "colorbar"; bar.scale = (bar_w, bar_h, 1.0)
    bm_, bb_, bnt = _mat("mat_colorbar")
    img = bnt.nodes.new("ShaderNodeTexImage"); img.image = bpy.data.images.load(str(FDIR / "colorbar_inferno.png"))
    bnt.links.new(img.outputs["Color"], bb_.inputs["Base Color"]); bnt.links.new(img.outputs["Color"], bb_.inputs["Emission Color"])
    bb_.inputs["Emission Strength"].default_value = 1.0; bb_.inputs["Roughness"].default_value = 1.0; bb_.inputs["Specular IOR Level"].default_value = 0.0
    bar.data.materials.append(bm_); cam_child(bar, bx, by)
    tm, tb, _ = _mat("mat_bartext"); tb.inputs["Base Color"].default_value = (0.92, 0.93, 0.95, 1)
    tb.inputs["Emission Color"].default_value = (0.92, 0.93, 0.95, 1); tb.inputs["Emission Strength"].default_value = 1.0
    def bar_text(name, body, x, y, align="CENTER", size=0.0038):
        tx = bpy.data.curves.new(name, "FONT"); tx.body = body; tx.size = size; tx.align_x = align; tx.align_y = "CENTER"
        ob = bpy.data.objects.new(name, tx); scene.collection.objects.link(ob); ob.data.materials.append(tm); cam_child(ob, x, y)
        return ob
    bar_text("bar_lo", "32.5 °C", bx - bar_w / 2 - 0.003, by, "RIGHT")
    bar_text("bar_hi", "35.8 °C", bx + bar_w / 2 + 0.003, by, "LEFT")
    bar_text("bar_title", "skin temperature, 3D Pennes model (thermal3d_full)", bx, by + 0.0075, "CENTER", 0.0030)
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = str(RENDER_DIR / f"physics_{SHOT}.png")
bpy.ops.render.render(write_still=True)
print("wrote", scene.render.filepath)
bpy.ops.file.pack_all()
bpy.ops.wm.save_as_mainfile(filepath=str(RENDER_DIR / f"physics_{SHOT}.blend"))
